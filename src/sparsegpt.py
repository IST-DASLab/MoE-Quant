from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn.modules.conv import _ConvNd

from src import dist_utils, model_utils, linalg_utils


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseGPT:

    def __init__(
        self,
        layer: nn.Module,
        rel_damp: float = 1e-2,
        block_size: int = None,
        is_distributed: bool = False,
        tied_sparsegpt_handle: Optional["SparseGPT"] = None
    ):
        self._validate_layer(layer)
        self.layer = layer
        self.W = self.layer.weight
        self.d_row, self.d_col = model_utils.get_number_of_rows_and_cols(layer)
        # SparseGPT hyperparameters
        self.rel_damp = rel_damp
        self.block_size = block_size or self.d_col
        # backup layer properties
        self.W_device = self.W.device
        self.W_dtype = self.W.dtype
        self.W_shape = self.W.shape
        # init hessian
        self.H = None
        self.num_samples = 0
        self.is_distributed = is_distributed
        self.tied_sparsegpt_handle = tied_sparsegpt_handle
        self.num_tied_handles = 0
        if tied_sparsegpt_handle is not None:
            tied_sparsegpt_handle.num_tied_handles += 1
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
        # get number of samples (tokens) in batch
        num_new_samples = input.shape[0]
        # hessian update
        beta = self.num_samples / (self.num_samples + num_new_samples)
        alpha = 2.0 / (self.num_samples + num_new_samples)
        self.H.addmm_(input.T, input, beta=beta, alpha=alpha)
        # update number of collected samples
        self.num_samples += num_new_samples

    @property
    def tokens_collected(self) -> int:
        return self.num_samples

    def reset(self) -> None:
        self.W = self.layer.weight
        if self.num_tied_handles == 0:
            self.H = None
        elif self.tied_sparsegpt_handle:
            self.tied_sparsegpt_handle.num_tied_handles -= 1
            if self.tied_sparsegpt_handle.num_tied_handles == 0:
                self.tied_sparsegpt_handle.H = None
        self.num_samples = 0
        torch.cuda.empty_cache()

    @torch.no_grad()
    def sparsification_pre_step(self) -> None:
        """
        Preparatory step with hessian regularization and weight reshaping.
        """
        # 1) Hessian preparation
        reduce_if_needed = True
        if self.H is None:
            if self.tied_sparsegpt_handle:
                self.H = self.tied_sparsegpt_handle.H
            else:
                self.H = torch.eye(self.d_col, device=self.W_device, dtype=torch.float32)
                self.issue_zero_samples = True
            # No need to reduce
            reduce_if_needed = False
        # synchronize Hessians
        if self.is_distributed and reduce_if_needed and dist_utils.is_dist_available_and_initialized():
            dist.all_reduce(self.H, op=dist.ReduceOp.AVG)
        # Replace matrix by identity in case of NaNs
        if torch.isnan(self.H).any().item():
            self.H = torch.eye(self.d_col, device=self.W_device, dtype=torch.float32)
            self.issue_nan_hessian = True
        # get ids of pruned channels
        pruned_ids = torch.diag(self.H) == 0
        self.H[pruned_ids, pruned_ids] = 1
        # 2) Weight preparation
        # copy weight, flatten
        self.W = self.W.clone().float()
        if isinstance(self.layer, _ConvNd):
            self.W = self.W.flatten(1, -1)
        self.W[:, pruned_ids] = 0
        # flag pre step as completed
        self.pre_step_completed = True

    @torch.no_grad()
    def _prune(self, n: int = 2, m: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prune the layer according to the given sparsity.
        """
        # 1) Define constants and chunk
        d_row, d_col, block_size, device, dtype = self.d_row, self.d_col, self.block_size, self.W_device, self.W_dtype

        is_main_sparsegpt_process = dist_utils.is_main() or not self.is_distributed

        if is_main_sparsegpt_process:
            w = self.W
            # Get hessian inverse
            hessian_inv = self._get_hessian_inverse()  
             # Get hessian inverse
            for c1 in range(0, d_col, block_size):
                c2 = min(c1 + block_size, d_col)
                ncols = c2 - c1  # number of columns
                w_blk = w[:, c1:c2].clone()  # column-wise weight slice
                res = torch.zeros_like(w_blk)
                errs = torch.zeros_like(w_blk)
                losses_blk = torch.zeros_like(w_blk)
                hessian_inv_blk = hessian_inv[c1:c2, c1:c2]
                mask = torch.zeros_like(w_blk, dtype=torch.bool)
                # 2) iterate over block
                for i in range(ncols):
                    if i % m == 0:
                        scores = w_blk[:, i: (i + m)].pow(2) / hessian_inv_blk.diag()[i: (i + m)].view(1, -1).pow(2)
                        thr, _ = torch.kthvalue(scores, k=n, dim=-1, keepdim=True)
                        mask[:, i: (i + m)] = scores > thr

                    w_ci = w_blk[:, i]
                    d = hessian_inv_blk[i, i]

                    q = w_ci.clone()
                    q[~mask[:, i]] = 0

                    res[:, i] = q
                    err = (w_ci - q) / d
                    losses_blk[:, i] = err ** 2

                    w_blk[:, i:].addr_(err, hessian_inv_blk[i, i:], alpha=-1)
                    errs[:, i] = err
                # 3) update the weights after block
                w[:, c1:c2] = res
                w[:, c2:].addmm_(errs, hessian_inv[c1:c2, c2:], alpha=-1)

            sweight = w.to(dtype=dtype)
        else:
            sweight = torch.empty(d_row, d_col, device=device, dtype=dtype)
        
        if self.is_distributed and dist_utils.is_dist_available_and_initialized():
            dist.barrier()
            dist.broadcast(sweight, src=0)

        return sweight

    def prune(self, n: int = 2, m: int = 4) -> Tensor:
        self.sparsification_pre_step()
        return self._prune(n, m)

    @torch.no_grad()
    def _get_hessian_inverse(self):
        w = self.W
        # Get columns with all zeros
        zero_cols = torch.nonzero(w.eq(0).all(dim=0))
        H = self.H
        # Regularize Hessian before sparsification
        if not self.tied_sparsegpt_handle:
            # Mask rows with zero input channels
            H[zero_cols, :] = 0
            H[:, zero_cols] = 0
            H[zero_cols, zero_cols] = 1
            # Hessian regularization
            damp = self.rel_damp * torch.diag(self.H).mean()
            self.H[range(self.d_col), range(self.d_col)] += damp
        # Invert
        try:
            H = linalg_utils.inv_sym(H)
            H_inv_cho = torch.linalg.cholesky(H, upper=True)
        except:
            H_inv_cho = torch.eye(self.d_col, device=H.device, dtype=torch.float32)
        # Divide Hessian inverse by diagonal (in order to not divide on it later)
        H_inv_cho.div_(H_inv_cho.diag()[:, None])
        return H_inv_cho
