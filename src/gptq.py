from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn.modules.conv import _ConvNd

from src import dist_utils, model_utils, linalg_utils, quant_utils


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(
        self,
        layer: nn.Module,
        perchannel: bool = True,
        group_size: Optional[int] = None,
        sym: bool = False,
        rel_damp: float = 1e-2,
        block_size: int = None,
        act_order: bool = False,
        is_distributed: bool = False,
    ):
        self._validate_layer(layer)
        self.layer = layer
        self.W = self.layer.weight
        self.d_row, self.d_col = model_utils.get_number_of_rows_and_cols(layer)
        # Quantizer hyperparameters
        self.quantizer = quant_utils.Quantizer()
        self.sym = sym
        self.perchannel = perchannel
        self.group_size = group_size
        # FastOBQ hyperparameters
        self.rel_damp = rel_damp
        self.block_size = block_size or self.d_col
        self.act_order = act_order
        self.group_size = group_size
        # backup layer properties
        self.W_device = self.W.device
        self.W_dtype = self.W.dtype
        self.W_shape = self.W.shape
        # init hessian
        self.H = None
        self.num_samples = 0
        self.tokens_collected = 0
        self.is_distributed = is_distributed
        # Flags indicating issues
        self.issue_zero_samples = False
        self.issue_nan_hessian = False
        self.issue_non_invertible = False

    @staticmethod
    def _validate_layer(layer):
        assert isinstance(layer, (nn.Linear, _ConvNd)), "OBC supports only linear and convolutional layers."

    def has_hessian_issues(self) -> bool:
        return any([self.issue_zero_samples, self.issue_nan_hessian, self.issue_non_invertible])

    # preparatory methods
    @torch.no_grad()
    def update(self, input: Tensor) -> None:
        """
        Update the estimate of Hessian matrix from a batch of data.

        Args:
            input: batch of layer inputs
        """
        # get batch size
        batch_size = input.shape[0]
        # init hessian
        if self.H is None:
            self.H = torch.zeros((self.d_col, self.d_col), device=input.device, dtype=torch.float32)
        # input reshaping
        if isinstance(self.layer, nn.Linear):
            input = input.reshape(-1, input.shape[-1])
        else:
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            # output size (batch_size, channels * \prod kernel_size, num_patches)
            input = unfold(input)
            input = input.transpose(1, 2).flatten(0, 1)
        input = input.float()
        # hessian update
        beta = self.num_samples / (self.num_samples + batch_size)
        alpha = 2.0 / (self.num_samples + batch_size)
        self.H.addmm_(input.T, input, beta=beta, alpha=alpha)
        # update number of collected samples
        self.num_samples += batch_size
        self.tokens_collected += input.shape[0]

    def reset(self) -> None:
        self.W = self.layer.weight
        self.H = None
        self.num_samples = 0
        torch.cuda.empty_cache()

    @torch.no_grad()
    def quantization_pre_step(self) -> None:
        """
        Preparatory step with hessian regularization and weight reshaping.
        """
        # 1) Hessian preparation
        if self.H is None:
            self.H = torch.eye(self.d_col, device=self.W_device, dtype=torch.float32)
            self.issue_zero_samples = True
        # synchronize Hessians
        if self.is_distributed and dist_utils.is_dist_available_and_initialized():
            dist.all_reduce(self.H, op=dist.ReduceOp.AVG)
        # Replace matrix by identity in case of NaNs
        if torch.isnan(self.H).any().item():
            self.H = torch.eye(self.d_col, device=self.W_device, dtype=torch.float32)
            self.issue_nan_hessian = True
        # get ids of pruned channels
        pruned_ids = torch.diag(self.H) == 0
        self.H[pruned_ids, pruned_ids] = 1
        # Hessian regularization
        damp = self.rel_damp * torch.diag(self.H).mean()
        self.H[range(self.d_col), range(self.d_col)] += damp
        # 2) Weight preparation
        # copy weight, flatten
        self.W = self.W.clone().float()
        if isinstance(self.layer, _ConvNd):
            self.W = self.W.flatten(1, -1)
        self.W[:, pruned_ids] = 0
        # flag pre step as completed
        self.pre_step_completed = True

    @torch.no_grad()
    def step(self, bits: int) -> Tensor:
        # 1) define constants and chunk
        d_row, d_col, block_size, device, dtype = self.d_row, self.d_col, self.block_size, self.W_device, self.W_dtype
        # get quantization group size
        group_size = self.group_size or d_col
        num_groups = d_col // group_size

        perm = None
        if self.act_order:
            perm = torch.argsort(torch.diag(self.H), descending=True)
            self.W.data = self.W[:, perm]
            self.H.data = self.H[perm, :][:, perm]

        # TODO a nicer way to implement this?
        is_main_gptq_process = dist_utils.is_main() or not self.is_distributed

        if is_main_gptq_process:
            # prepare weight and Cholesky of H^{-1}
            w, H_inv_cho= self._prepare()

            qweight = torch.empty(d_row, d_col, device=device, dtype=torch.uint8)
            scale = torch.empty(d_row, num_groups, device=device, dtype=dtype)
            zero = torch.empty(d_row, num_groups, device=device, dtype=dtype)

            # Configure quantizer
            quantizer = self.quantizer
            quantizer.configure(bits=bits, perchannel=self.perchannel, sym=self.sym)

            if not self.group_size:
                quantizer.find_params(w, weight=True)
                self.scale = quantizer.scale.to(dtype)
                self.zero = quantizer.zero.to(zero.dtype)

            # iterate over columns
            for c1 in range(0, d_col, block_size):
                c2 = min(c1 + block_size, d_col)
                ncols = c2 - c1  # number of columns
                w_blk = w[:, c1:c2].clone()  # column-wise weight slice
                errs = torch.zeros_like(w_blk)
                losses_blk = torch.zeros_like(w_blk)
                H_inv_cho_blk = H_inv_cho[c1:c2, c1:c2]
                # 2) iterate over block
                for i in range(ncols):
                    w_ci = w_blk[:, i]
                    d = H_inv_cho_blk[i, i]

                    if self.group_size and (c1 + i) % group_size == 0:
                        quantizer.find_params(w[:, (c1 + i) : (c1 + i + group_size)], weight=True)
                        scale[:, (c1 + i) // group_size] = quantizer.scale.flatten()
                        zero[:, (c1 + i) // group_size] = quantizer.zero.flatten()

                    q = quantizer.quantize(w_ci.unsqueeze(1))
                    w_q = quantizer.dequantize(q).flatten()

                    qweight[:, c1 + i] = q.flatten()
                    err = (w_ci - w_q) / d
                    losses_blk[:, i] = err**2

                    w[:, c1 + i] = w_q
                    w_blk[:, i:].addr_(err, H_inv_cho_blk[i, i:], alpha=-1)
                    errs[:, i] = err
                # 3) update the weights after block
                w[:, c2:].addmm_(errs, H_inv_cho[c1:c2, c2:], alpha=-1)
        else:
            qweight = torch.empty(d_row, d_col, device=device, dtype=torch.uint8)
            scale = torch.empty(d_row, num_groups, device=device, dtype=dtype)
            zero = torch.empty(d_row, num_groups, device=device, dtype=dtype)

        # Restore Hessian order
        if self.act_order:
            invperm = torch.argsort(perm)
            self.H = self.H[invperm, :][:, invperm]

        if self.is_distributed and dist_utils.is_dist_available_and_initialized():
            dist.barrier()
            dist.broadcast(qweight, src=0)
            dist.broadcast(scale, src=0)
            dist.broadcast(zero, src=0)

        return qweight, scale, zero, perm

    def quantize(self, bitwidth_options) -> Tensor:
        self.quantization_pre_step()
        return self.step(bitwidth_options)

    @torch.no_grad()
    def _prepare(self):
        w = self.W
        # get columns with all zeros
        zero_cols = torch.nonzero(w.eq(0).all(dim=0))
        H = self.H
        # mask rows with zero input channels
        H[zero_cols, :] = 0
        H[:, zero_cols] = 0
        H[zero_cols, zero_cols] = 1
        # invert
        try:
            H = linalg_utils.inv_sym(H)
            H_inv_cho = torch.linalg.cholesky(H, upper=True)
        except:
            H_inv_cho = torch.eye(self.d_col, device=H.device, dtype=torch.float32)
            self.issue_non_invertible = True
        return w, H_inv_cho
    