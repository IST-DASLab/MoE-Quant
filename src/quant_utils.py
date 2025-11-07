import math
from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F
import triton
from triton import language as tl
from transformers import AutoConfig
from compressed_tensors.compressors import unpack_from_int32


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


FP8_GROUP_SIZE = 128
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz)


class QuantizationScale(Enum):
    ABSMAX = "absmax"
    MSE = "mse"


@triton.jit
def tl_pow(x, a):
    return (x.abs().log() * a).exp()  # TODO: triton does not have x.pow(a) or x ** a?


@triton.jit
def tl_round(x):
    return (
        x + 0.5
    ).floor()  # TODO: triton does not have round()? We might want to change to round to even number here.


@triton.jit
def tl_round_fp(x, dtype):
    return x if dtype is None else x.cast(dtype, fp_downcast_rounding="rtne").cast(x.dtype)


@triton.jit
def tl_quantize(x, scale, qzero, maxq):
    return tl.clamp(tl_round(x / scale + qzero), 0.0, maxq)


@triton.jit
def tl_dequantize(qx, scale, qzero, dtype):
    return tl_round_fp((qx - qzero) * scale, dtype)


@triton.jit
def tl_dequantize_quantized(x, scale, qzero, maxq, dtype):
    return tl_dequantize(tl_quantize(x, scale, qzero, maxq), scale, qzero, dtype)


def round_fp(x: torch.Tensor, dtype: torch.dtype = None) -> torch.Tensor:
    return x if dtype is None else x.to(dtype=dtype).to(x.dtype)


def quantize(x: torch.Tensor, scale: torch.Tensor, qzero: torch.Tensor, maxq: torch.Tensor) -> torch.Tensor:
    return (x / scale + qzero).round().clamp(torch.zeros_like(maxq), maxq)


def dequantize(qx: torch.Tensor, scale: torch.Tensor, qzero: torch.Tensor, dtype: torch.dtype = None) -> torch.Tensor:
    return round_fp((qx - qzero) * scale, dtype)


def dequantize_quantized(
    x: torch.Tensor, scale: torch.Tensor, qzero: torch.Tensor, maxq: torch.Tensor, dtype: torch.dtype = None
) -> torch.Tensor:
    return dequantize(quantize(x, scale, qzero, maxq), scale, qzero, dtype)


def find_quantization_meta(
    x: torch.Tensor,
    bit_width: int,
    symmetric: bool = False,
    dtype: torch.dtype = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find quantization metadata over dim=-1
    x: (..., C), weight
    bit_width: int
    symmetric: bool, whether to set qzero to the middle
    dtype: torch.dtype, target scale dtype, fp16 or bf16
    """
    epsilon: float = 1e-12
    maxq = torch.tensor(2**bit_width - 1, dtype=x.dtype, device=x.device)  # ()

    x_min = x.amax(dim=-1)
    x_max = x.amin(dim=-1)

    if symmetric:
        scale = (2.0 / maxq) * torch.maximum(x_min.abs(), x_max.abs())
        scale = round_fp(scale + epsilon, dtype)  # (...)
        qzero = torch.full_like(scale, ((maxq + 1.0) * 0.5).item())  # (...)
    else:
        scale = round_fp((x_max - x_min) / maxq + epsilon, dtype)  # (...)
        qzero = (-x_min / scale).round().clamp(0, maxq)  # (...)
    return scale, qzero, maxq


@triton.jit
def mse_scale_triton_kernel(
    x_ptr,
    p_ptr,
    scale_ptr,
    qzero_ptr,
    maxq_ptr,
    dtype_ptr,
    norm: float,
    p_size: int,
    group_size: int,
    batch_size: int,
    BLOCK_SIZE_P: tl.constexpr,
    BLOCK_SIZE_G: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b_offsets = pid * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)  # (R)
    b_mask = b_offsets < batch_size  # (R)
    x_offsets = b_offsets[:, None] * group_size + tl.arange(0, BLOCK_SIZE_G)  # (R, C)
    x_mask = b_mask[:, None] & (tl.arange(0, BLOCK_SIZE_G) < group_size)  # (R, C)
    p_offsets = tl.arange(0, BLOCK_SIZE_P)  # (P)
    p_mask = p_offsets < p_size  # (P)
    scale_ptrs = scale_ptr + b_offsets  # (R)

    x = tl.load(x_ptr + x_offsets, mask=x_mask)[:, None, :]  # (R, 1, C)
    p = tl.load(p_ptr + p_offsets, mask=p_mask)  # (P)
    scale = tl.load(scale_ptrs, mask=b_mask)  # (R)
    qzero = tl.load(qzero_ptr + b_offsets, mask=b_mask)[:, None, None]  # (R, 1, 1)
    maxq = tl.load(maxq_ptr)  # ()
    dtype = None if dtype_ptr is None else tl.load(dtype_ptr).dtype

    scale_p = tl_round_fp(scale[:, None] * p, dtype)[:, :, None]  # (R, P, 1)
    q = tl_dequantize_quantized(x, scale_p, qzero, maxq, dtype)  # (R, P, C)
    best_idx = tl.argmin(tl.sum(tl_pow(q - x, norm), axis=-1), axis=-1, tie_break_left=False)  # (R)

    scale = tl_round_fp(scale * tl.load(p_ptr + best_idx), dtype)  # (R)  # TODO: replace with tl.gather()
    tl.store(scale_ptrs, scale, mask=b_mask)  # (R)


def mse_scale(
    x: torch.Tensor,
    p: torch.Tensor,
    scale: torch.Tensor,
    qzero: torch.Tensor,
    maxq: torch.Tensor,
    dtype: torch.dtype = None,
    norm: float = 2.4,
) -> torch.Tensor:
    """
    Find the optimal scale for quantization with respect to the MSE loss
    x: (..., C), weight
    p: (P), shrinkage factors
    scale: (...), initial scale, modified in-place and returned
    qzero: (...), zero points
    maxq: ()
    dtype: torch.dtype, target scale dtype, fp16 or bf16
    norm: float, norm for the loss
    debug_mode: bool, whether to use the baseline implementation without Triton
    """

    assert (
        x.is_contiguous()
        and p.is_contiguous()
        and scale.is_contiguous()
        and qzero.is_contiguous()
        and maxq.is_contiguous()
    )
    batch_size: int = torch.tensor(x.shape[:-1]).prod().item()
    previous_device: torch.device = torch.device(f"cuda:{torch.cuda.current_device()}")
    torch.cuda.set_device(x.device)
    grid = lambda meta: (triton.cdiv(batch_size, meta["BLOCK_SIZE_B"]),)
    mse_scale_triton_kernel[grid](
        x,
        p,
        scale,
        qzero,
        maxq,
        torch.empty(0, dtype=dtype) if dtype is not None else None,
        norm,
        p.size(-1),
        x.size(-1),
        batch_size,
        BLOCK_SIZE_P=torch.tensor(p.size(-1)).log2().ceil().exp2().int().item(),
        BLOCK_SIZE_G=torch.tensor(x.size(-1)).log2().ceil().exp2().int().item(),
        BLOCK_SIZE_B=1,
    )
    torch.cuda.set_device(previous_device)
    return scale


@torch.no_grad()
def get_quantization_grid(
    weight: torch.Tensor,
    group_size: int,
    bits: int,
    symmetric: bool = False,
    dtype: torch.dtype = None,
    quantization_scale: QuantizationScale = QuantizationScale.ABSMAX,
    quant_max_shrink: float = 0.2,
    quant_n_grid: int = 100,
    quant_norm: float = 2.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get the quantization grid for the weight matrix
    weight: (..., (R), C)
    scale: (..., (R), C)
    qzero: (..., (R), C)
    maxq: ()
    """
    weight = weight.unflatten(dim=-1, sizes=(-1, group_size))  # (..., G, gs)

    scale, qzero, maxq = find_quantization_meta(
        x=weight,
        bit_width=bits,
        symmetric=symmetric,
        dtype=dtype,
    )  # (..., G), (..., G), ()
    if quantization_scale == QuantizationScale.MSE:
        search_points = torch.linspace(1, quant_max_shrink, quant_n_grid, dtype=weight.dtype, device=weight.device)
        mse_scale(
            x=weight.contiguous(),  # (..., G, gs)
            p=search_points,  # (..., P)
            scale=scale,  # (..., G)
            qzero=qzero,  # (..., G)
            maxq=maxq,  # ()
            dtype=dtype,
            norm=quant_norm,
        )

    scale = scale.repeat_interleave(group_size, dim=-1)  # (..., C)
    qzero = qzero.repeat_interleave(group_size, dim=-1)  # (..., C)

    weight = weight.flatten(start_dim=-2)  # (..., C)

    assert weight.shape == scale.shape == qzero.shape and maxq.shape == ()
    return scale, qzero, maxq  # (..., (R), C), (..., (R), C), ()


def dequantize_linear_weight(
    qweight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    perm: Optional[torch.Tensor] = None,
):
    scale = scale.view(qweight.shape[0], -1, 1)
    zero = zero.view(qweight.shape[0], -1, 1)
    num_groups = scale.shape[1]
    weight = dequantize(qweight.view(qweight.shape[0], num_groups, -1), scale, zero).view_as(qweight)
    if perm is not None:
        invperm = perm.argsort()
        weight = weight[:, invperm]
    return weight


def get_relative_mse_error(q: torch.Tensor, w: torch.Tensor, H: Optional[torch.Tensor] = None):
    delta = q - w
    if H is None:
        return delta.pow(2).mean() / w.pow(2).mean()
    else:
        return (delta).mm(H).mul(delta).mean() / (w.mm(H).mul(w).mean() + 1e-6)


def dequantize_weight_from_fp8(W, s):
    g = FP8_GROUP_SIZE
    # Dequantize weight
    d_out, d_in = W.shape
    # Pad weight if needed
    pad_out = math.ceil(d_out / g) * g - d_out
    pad_in = math.ceil(d_in / g) * g - d_in
    W = F.pad(W, (0, pad_in, 0, pad_out))
    d_out_pad, d_in_pad = W.shape

    W = W.view(d_out_pad // g, g, d_in_pad // g, g)
    s = s.view(d_out_pad // g, 1, d_in_pad // g, 1)
    W = (W * s).view(d_out_pad, d_in_pad)

    # Remove padding
    W = W[:d_out, :d_in]
    return W


def dequantize_state_dict(
    state_dict: dict[str, torch.Tensor], 
    dtype: torch.dtype = torch.float16,
    quantization_format: str = "fp8"
) -> None:
    assert quantization_format in ["fp8", "int4"]
    state_dict_keys = list(state_dict.keys())

    # Original DeepSeek packing
    if quantization_format == "fp8":
        # Dequantize
        for k in state_dict_keys:
            if k.endswith("scale_inv"):
                layer_name, _ = k.rsplit(".", 1)

                W = state_dict[f"{layer_name}.weight"].to(dtype)
                s = state_dict[f"{layer_name}.weight_scale_inv"].to(dtype)

                state_dict[f"{layer_name}.weight"] = dequantize_weight_from_fp8(W, s)
                del state_dict[f"{layer_name}.weight_scale_inv"]

    # KIMI-K2 compressed tensors packing
    elif quantization_format == "int4":
        # Dequantize
        for k in state_dict_keys:
            if k.endswith("weight_packed"):
                layer_name, _ = k.rsplit(".", 1)

                weight_shape = state_dict[f"{layer_name}.weight_shape"]

                qweight_packed = state_dict[f"{layer_name}.weight_packed"]
                qweight = unpack_from_int32(qweight_packed, num_bits=4, shape=weight_shape)

                scale = state_dict[f"{layer_name}.weight_scale"]
                
                W = (qweight.view(*scale.shape, -1) * scale[..., None]).view_as(qweight).to(dtype)

                state_dict[f"{layer_name}.weight"] = W
                del state_dict[f"{layer_name}.weight_packed"]
                del state_dict[f"{layer_name}.weight_shape"]
                del state_dict[f"{layer_name}.weight_scale"]


def can_dequantize(
    state_dict: dict[str, torch.Tensor],
    quantization_format: str = "fp8"
) -> bool:
    assert quantization_format in ["fp8", "int4"]
     # Original DeepSeek packing
    if quantization_format == "fp8":
        for k, v in state_dict.items():
            if v.dtype in FP8_DTYPES and f"{k}_scale_inv" not in state_dict:
                return False
            
    # KIMI-K2 compressed tensors packing
    elif quantization_format == "int4":
        for k, v in state_dict.items():
            if k.endswith("weight_packed"):
                layer_name, _ = k.rsplit(".", 1)
                if f"{layer_name}.weight_shape" not in state_dict or f"{layer_name}.weight_scale" not in state_dict:
                    return False
    return True


def infer_quantization_format(config: AutoConfig) -> str:
    quant_method = config.quantization_config["quant_method"]
    # DeepSeek format
    if quant_method == "fp8":
        return "fp8"
    # KIMI-K2 format
    elif quant_method == "compressed-tensors":
        if config.quantization_config["config_groups"]["group_0"]["weights"]["num_bits"] == 4:
            return "int4"
        else:
            raise ValueError("Only 4-bit quantization is supported.")
    else:
        raise ValueError("Unknown or unsupported quantization method.")
