import torch
import triton
from triton import language as tl

from src.quant_utils import tl_quantize, tl_dequantize

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


@triton.jit
def sparse_quantize_error_triton_kernel(
    x_ptr,
    qx_ptr,
    error_ptr,
    scale_ptr,
    qzero_ptr,
    maxq_ptr,
    dtype_ptr,
    n_elements: int,
    sparse_n: tl.constexpr,
    sparse_m: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    scale = tl.load(scale_ptr + offsets, mask=mask)
    qzero = tl.load(qzero_ptr + offsets, mask=mask)
    maxq = tl.load(maxq_ptr)
    dtype = None if dtype_ptr is None else tl.load(dtype_ptr).dtype

    # Prune
    if sparse_m > 0:
        sparse_offsets = tl.arange(0, sparse_m)
        sx = x.reshape(BLOCK_SIZE // sparse_m, sparse_m)
        score = sx.abs()
        sparse_mask = tl.zeros(score.shape, dtype=tl.int1)
        for i in range(sparse_n):
            max_idx = tl.argmin(score, axis=1, keep_dims=True)
            sparse_mask_i = sparse_offsets == max_idx
            sparse_mask = sparse_mask | sparse_mask_i
            score = tl.where(sparse_mask_i, float("inf"), score)
        sx = tl.where(sparse_mask, 0, sx)
        sx = sx.reshape(BLOCK_SIZE)
    else:
        sx = x

    # Quantize
    qx = tl_quantize(sx, scale, qzero, maxq)
    y = tl_dequantize(qx, scale, qzero, dtype)
    error = y - x

    tl.store(x_ptr + offsets, y, mask=mask)
    tl.store(qx_ptr + offsets, qx, mask=mask)
    tl.store(error_ptr + offsets, error, mask=mask)


def sparse_quantize_error_triton(
    x: torch.Tensor,
    qx: torch.Tensor,
    error: torch.Tensor,
    scale: torch.Tensor,
    qzero: torch.Tensor,
    maxq: torch.Tensor,
    sparse_n: int = 0,
    sparse_m: int = 0,
    dtype: torch.dtype = None,
) -> None:
    if sparse_m > 0:
        assert 0 < sparse_n < sparse_m, "sparse_n must be in (0, sparse_m)"

    n_elements: int = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    sparse_quantize_error_triton_kernel[grid](
        x,
        qx,
        error,
        scale,
        qzero,
        maxq,
        torch.empty(0, dtype=dtype) if dtype is not None else None,
        n_elements,
        sparse_n,
        sparse_m,
        BLOCK_SIZE=128,
    )


@triton.jit
def addvv_triton_kernel(
    vec_a_ptr,
    vec_b_ptr,
    mat_c_ptr,
    size_a: int,
    size_b: int,
    BLOCK_SIZE_B: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset_a = pid % size_a
    offsets_b = pid // size_a * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    mask = offsets_b < size_b
    c_ptrs = mat_c_ptr + offset_a * size_b + offsets_b

    a = tl.load(vec_a_ptr + offset_a)
    b = tl.load(vec_b_ptr + offsets_b, mask=mask)
    c = tl.load(c_ptrs, mask=mask)
    c = tl.fma(a, b, c)

    tl.store(c_ptrs, c, mask=mask)


def addvv_triton(
    vec_a: torch.Tensor,
    vec_b: torch.Tensor,
    mat_c: torch.Tensor,
) -> None:
    size_a, size_b = mat_c.shape
    grid = lambda meta: (size_a * triton.cdiv(size_b, meta["BLOCK_SIZE_B"]),)
    addvv_triton_kernel[grid](
        vec_a,
        vec_b,
        mat_c,
        size_a,
        size_b,
        BLOCK_SIZE_B=256,
    )


def gptq_loop_graph(
    weight: torch.Tensor,
    hessian_inv: torch.Tensor,
    scale: torch.Tensor,
    qzero: torch.Tensor,
    maxq: torch.Tensor,
    qweight: torch.Tensor = None,
    error_block: torch.Tensor = None,
    dtype: torch.dtype = None,
    gptq_block_size: int = 128,
    sparse_n: int = 0,
    sparse_m: int = 0,
    direct: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CUDA Graph wrapper for GPTQ loops
    """
    n_columns, n_rows = weight.shape
    w_dtype: torch.dtype = weight.dtype
    device: torch.device = weight.device

    if direct:
        if qweight is None:
            qweight: torch.Tensor = torch.empty_like(weight)
        if error_block is None:
            error_block: torch.Tensor = torch.empty(gptq_block_size, n_rows, dtype=w_dtype, device=device)
        assert (
            weight.is_contiguous()
            and hessian_inv.is_contiguous()
            and scale.is_contiguous()
            and qzero.is_contiguous()
            and maxq.is_contiguous()
            and qweight.is_contiguous()
            and error_block.is_contiguous()
        )
        for i1 in range(0, n_columns, gptq_block_size):
            i2: int = min(i1 + gptq_block_size, n_columns)
            for j in range(i1, i2):
                sparse_quantize_error_triton(
                    weight[j], qweight[j], error_block[j - i1], scale[j], qzero[j], maxq, sparse_n, sparse_m, dtype,
                )
                addvv_triton(hessian_inv[j, j + 1 : i2], error_block[j - i1], weight[j + 1 : i2])
            weight[i2:].addmm_(hessian_inv[i1:i2, i2:].t(), error_block[: i2 - i1], beta=1, alpha=1)
        return qweight, weight

    previous_device: torch.device = torch.device(f"cuda:{torch.cuda.current_device()}")
    torch.cuda.set_device(weight.device)
    if not hasattr(gptq_loop_graph, "graph_info"):
        gptq_loop_graph.graph_info = {}
    graph_key: tuple = n_columns, n_rows, w_dtype, dtype, gptq_block_size, device
    if graph_key not in gptq_loop_graph.graph_info:
        graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
        graph_tensors: dict[str, torch.Tensor] = {
            "weight": torch.empty_like(weight.contiguous()),
            "hessian_inv": torch.empty_like(hessian_inv.contiguous()),
            "scale": torch.empty_like(scale.contiguous()),
            "qzero": torch.empty_like(qzero.contiguous()),
            "maxq": torch.empty_like(maxq.contiguous()),
            "qweight": torch.empty_like(weight.contiguous()),
            "error_block": torch.empty(gptq_block_size, n_rows, dtype=w_dtype, device=device),
        }
        n_warmups: int = 5
        s: torch.cuda.Stream = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmups):
                gptq_loop_graph(**graph_tensors, dtype=dtype, gptq_block_size=gptq_block_size, sparse_n=sparse_n, sparse_m=sparse_m, direct=True)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(graph):
            gptq_loop_graph(**graph_tensors, dtype=dtype, gptq_block_size=gptq_block_size, sparse_n=sparse_n, sparse_m=sparse_m, direct=True)
        gptq_loop_graph.graph_info[graph_key] = {"graph": graph, "tensors": graph_tensors}

    graph, graph_tensors = (
        gptq_loop_graph.graph_info[graph_key]["graph"],
        gptq_loop_graph.graph_info[graph_key]["tensors"],
    )
    graph_tensors["weight"].copy_(weight)
    graph_tensors["hessian_inv"].copy_(hessian_inv)
    graph_tensors["scale"].copy_(scale)
    graph_tensors["qzero"].copy_(qzero)
    graph_tensors["maxq"].copy_(maxq)
    graph.replay()
    weight.copy_(graph_tensors["weight"])
    torch.cuda.set_device(previous_device)
    return graph_tensors["qweight"], weight


def gptq_loop(
    weight: torch.Tensor,
    hessian_inv: torch.Tensor,
    scale: torch.Tensor,
    qzero: torch.Tensor,
    maxq: torch.Tensor,
    dtype: torch.dtype,
    gptq_block_size: int = 128,
    sparse_n: int = 0,
    sparse_m: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weight tensor with GPTQ algorithm
    weight: (C, R), transposed weight tensor to quantize, modified in-place and returned
    hessian_inv: (C, C), inverse of Hessian matrix
    scale: (C, R), transposed scale tensor for quantization
    qzero: (C, R), transposed zero-point tensor for quantization
    maxq: (), maximum quantized value
    dtype: target scale dtype, fp16 or bf16
    gptq_block_size: block size for GPTQ loop, this is independent of the quantization group size
    """
    if gptq_block_size <= 0:
        gptq_block_size = weight.size(-2)

    qweight, _ = gptq_loop_graph(
        weight=weight,
        hessian_inv=hessian_inv,
        scale=scale,
        qzero=qzero,
        maxq=maxq,
        dtype=dtype,
        gptq_block_size=gptq_block_size,
        sparse_n=sparse_n,
        sparse_m=sparse_m,
        direct=False,
    )
    return qweight # (C, R)
