"""
FP8 Blockwise/Rowwise Triton Kernels

This module provides Triton kernels for FP8 matrix multiplication with per-block
and per-row scaling, enabling native FP8 matmul instead of dequantize-fallback.

Key kernels:
- fp8_gemm_blockwise: 2D blockwise FP8 matmul (weights have [N//bs, K//bs] scales)
- fp8_gemm_rowwise: Rowwise FP8 matmul (weights have [N] scales)
- fp8_act_quant: Activation quantization to FP8 blockwise

Based on INT8 kernel patterns from int8_kernels.py, adapted for FP8:
- FP8 data loaded → cast to FP32 → accumulate → apply scales → output

torch._scaled_mm only supports scalar (tensorwise) scales, hence these custom kernels.
"""

import torch
import logging
from typing import Tuple

# Try to import Triton
try:
    import triton
    import triton.language as tl
    from triton import Config

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
    logging.info("FP8 kernels: Triton not available, will use dequantize fallback")


def _check_triton_available() -> bool:
    """Check if Triton is available for FP8 kernels."""
    return _HAS_TRITON


if _HAS_TRITON:
    # ==============================================================================
    # FP8 Activation Quantization Kernels
    # ==============================================================================

    @triton.jit
    def fp8_act_quant_kernel(
        x_ptr,
        y_ptr,
        s_ptr,
        BLOCK_SIZE: tl.constexpr,
        FP8_MAX: tl.constexpr,
    ):
        """
        Quantizes activation tensor to FP8 with per-block scaling.

        Each program handles one block of BLOCK_SIZE elements.
        """
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        # Load and compute per-block max
        x = tl.load(x_ptr + offs).to(tl.float32)
        amax = tl.max(tl.abs(x))

        # Compute scale (dequant scale = amax / FP8_MAX)
        # Store the dequant scale for later use
        scale = amax / FP8_MAX
        scale = tl.maximum(scale, 1e-12)  # Prevent division by zero

        # Quantize: x_fp8 = x / scale, clamped to [-FP8_MAX, FP8_MAX]
        # Note: We store dequant scale, so quant is x * (FP8_MAX / amax)
        quant_scale = FP8_MAX / tl.maximum(amax, 1e-12)
        y = x * quant_scale
        y = tl.minimum(tl.maximum(y, -FP8_MAX), FP8_MAX)

        # Store FP8 quantized values and scale
        tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty))
        tl.store(s_ptr + pid, scale)

    def fp8_act_quant(
        x: torch.Tensor, block_size: int = 128, dtype: torch.dtype = torch.float8_e4m3fn
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantizes activation tensor to FP8 with blockwise scaling.

        Args:
            x: Input tensor [..., K] where K is divisible by block_size
            block_size: Block size for quantization
            dtype: FP8 dtype (float8_e4m3fn or float8_e5m2)

        Returns:
            Tuple of (FP8 quantized tensor, scale tensor [..., K//block_size])
        """
        assert x.is_contiguous(), "Input must be contiguous"
        assert (
            x.size(-1) % block_size == 0
        ), f"Last dim {x.size(-1)} not divisible by {block_size}"

        fp8_max = torch.finfo(dtype).max

        y = torch.empty_like(x, dtype=dtype)
        s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)

        num_programs = s.numel()
        grid = (num_programs,)

        fp8_act_quant_kernel[grid](
            x,
            y,
            s,
            BLOCK_SIZE=block_size,
            FP8_MAX=fp8_max,
        )
        return y, s

    # ==============================================================================
    # FP8 Blockwise GEMM Kernel
    # ==============================================================================

    fp8_gemm_configs = [
        Config(
            {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
            num_stages=num_stages,
            num_warps=8,
        )
        for block_m in [64, 128]
        for block_n in [64, 128]
        for num_stages in [3, 4]
    ]

    @triton.jit
    def fp8_gemm_blockwise_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        a_s_ptr,
        b_s_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        input_block_size: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        """
        FP8 blockwise GEMM kernel.

        Computes C = A @ B.T where:
        - A is FP8 [..., K] with scales [..., K//input_block_size]
        - B is FP8 [N, K] with scales [N//input_block_size, K//input_block_size]
        - C is output [..., N] in float16/bfloat16

        The kernel:
        1. Loads FP8 tiles from A and B
        2. Casts to FP32 for accumulation
        3. Applies per-block scales from both A and B
        4. Accumulates the scaled products
        """
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        k_blocks = tl.cdiv(K, BLOCK_SIZE_K)

        # Compute offsets
        offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        # Pointers to FP8 data
        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

        # Activation scale pointers: shape [..., K//input_block_size]
        # For each M row, we iterate through K blocks
        a_s_k_blocks = tl.cdiv(K, input_block_size)
        a_s_ptrs = a_s_ptr + offs_m * a_s_k_blocks

        # Weight scale pointers: shape [N//input_block_size, K//input_block_size]
        b_s_k_blocks = tl.cdiv(K, input_block_size)
        b_s_rows = offs_n // input_block_size

        # Accumulator
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # Main loop over K dimension
        for k_idx in range(k_blocks):
            k_start = k_idx * BLOCK_SIZE_K

            # Load FP8 tiles and cast to float32
            mask_k = offs_k < K - k_start
            a_fp8 = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b_fp8 = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)

            # Cast to float32 for computation
            a_f32 = a_fp8.to(tl.float32)
            b_f32 = b_fp8.to(tl.float32)

            # Matrix multiply tile
            dot_result = tl.dot(a_f32, b_f32)

            # Compute scale index for this K block
            # Each BLOCK_SIZE_K may span multiple input_block_size blocks
            k_scale_idx = k_start // input_block_size

            # Load scales
            a_s = tl.load(a_s_ptrs + k_scale_idx)  # [BLOCK_SIZE_M]
            b_s = tl.load(b_s_ptr + b_s_rows * b_s_k_blocks + k_scale_idx)

            # Apply scales: result = dot * a_scale[:, None] * b_scale
            accumulator += dot_result * a_s[:, None] * b_s[None, :]

            # Advance pointers
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K

        # Store result
        c = accumulator.to(c_ptr.dtype.element_ty)
        offs_m_actual = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n_actual = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_m_actual[:, None] * N + offs_n_actual[None, :]
        mask = (offs_m_actual[:, None] < M) & (offs_n_actual[None, :] < N)
        tl.store(c_ptrs, c, mask=mask)

    def fp8_gemm_blockwise(
        a: torch.Tensor,
        a_s: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
        input_block_size: int = 128,
    ) -> torch.Tensor:
        """
        FP8 blockwise matrix multiplication.

        Args:
            a: FP8 activations [..., K]
            a_s: Activation scales [..., K//input_block_size]
            b: FP8 weights [N, K]
            b_s: Weight scales [N//input_block_size, K//input_block_size]
            input_block_size: Block size used for quantization

        Returns:
            Output tensor [..., N] in default float dtype
        """
        assert (
            a.is_contiguous() and b.is_contiguous()
        ), "Input tensors must be contiguous"
        assert (
            a_s.is_contiguous() and b_s.is_contiguous()
        ), "Scale tensors must be contiguous"
        assert b.dim() == 2, f"Weight must be 2D, got {b.dim()}D"

        K = a.size(-1)
        M = a.numel() // K
        N = b.shape[0]
        batch_shape = a.shape[:-1]

        assert b.size(1) == K, f"Shape mismatch: a has K={K}, b has shape {b.shape}"

        # Output tensor
        c = a.new_empty(*batch_shape, N, dtype=torch.float16)

        # Grid
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = input_block_size

        grid = (
            triton.cdiv(M, BLOCK_SIZE_M),
            triton.cdiv(N, BLOCK_SIZE_N),
        )

        fp8_gemm_blockwise_kernel[grid](
            a,
            b,
            c,
            a_s,
            b_s,
            M,
            N,
            K,
            input_block_size,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        return c

    # ==============================================================================
    # FP8 Blockwise GEMM with Bias (addmm)
    # ==============================================================================

    @triton.jit
    def fp8_addmm_blockwise_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        bias_ptr,
        a_s_ptr,
        b_s_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        input_block_size: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        """
        FP8 blockwise GEMM kernel with optional bias addition.
        """
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        k_blocks = tl.cdiv(K, BLOCK_SIZE_K)

        # Compute offsets
        offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        # Pointers to FP8 data
        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

        # Scale pointers
        a_s_k_blocks = tl.cdiv(K, input_block_size)
        a_s_ptrs = a_s_ptr + offs_m * a_s_k_blocks
        b_s_k_blocks = tl.cdiv(K, input_block_size)
        b_s_rows = offs_n // input_block_size

        # Accumulator
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # Main loop
        for k_idx in range(k_blocks):
            k_start = k_idx * BLOCK_SIZE_K

            mask_k = offs_k < K - k_start
            a_fp8 = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b_fp8 = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)

            a_f32 = a_fp8.to(tl.float32)
            b_f32 = b_fp8.to(tl.float32)

            dot_result = tl.dot(a_f32, b_f32)

            k_scale_idx = k_start // input_block_size
            a_s = tl.load(a_s_ptrs + k_scale_idx)
            b_s = tl.load(b_s_ptr + b_s_rows * b_s_k_blocks + k_scale_idx)

            accumulator += dot_result * a_s[:, None] * b_s[None, :]

            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K

        # Add bias if present
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
            accumulator += bias[None, :]

        # Store result
        c = accumulator.to(c_ptr.dtype.element_ty)
        offs_m_actual = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n_actual = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_m_actual[:, None] * N + offs_n_actual[None, :]
        mask = (offs_m_actual[:, None] < M) & (offs_n_actual[None, :] < N)
        tl.store(c_ptrs, c, mask=mask)

    def fp8_addmm_blockwise(
        a: torch.Tensor,
        a_s: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
        bias: torch.Tensor = None,
        input_block_size: int = 128,
    ) -> torch.Tensor:
        """
        FP8 blockwise matrix multiplication with optional bias.

        Args:
            a: FP8 activations [..., K]
            a_s: Activation scales [..., K//input_block_size]
            b: FP8 weights [N, K]
            b_s: Weight scales [N//input_block_size, K//input_block_size]
            bias: Optional bias [N]
            input_block_size: Block size used for quantization

        Returns:
            Output tensor [..., N] in default float dtype
        """
        assert (
            a.is_contiguous() and b.is_contiguous()
        ), "Input tensors must be contiguous"
        assert (
            a_s.is_contiguous() and b_s.is_contiguous()
        ), "Scale tensors must be contiguous"

        K = a.size(-1)
        M = a.numel() // K
        N = b.shape[0]
        batch_shape = a.shape[:-1]

        c = a.new_empty(*batch_shape, N, dtype=torch.float16)

        has_bias = bias is not None
        if has_bias:
            assert bias.is_contiguous() and bias.dim() == 1 and bias.size(0) == N
            bias_ptr = bias
        else:
            bias_ptr = c  # Dummy, won't be used

        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = input_block_size

        grid = (
            triton.cdiv(M, BLOCK_SIZE_M),
            triton.cdiv(N, BLOCK_SIZE_N),
        )

        fp8_addmm_blockwise_kernel[grid](
            a,
            b,
            c,
            bias_ptr,
            a_s,
            b_s,
            M,
            N,
            K,
            input_block_size,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            HAS_BIAS=has_bias,
        )
        return c

    # ==============================================================================
    # FP8 Rowwise GEMM Kernel
    # ==============================================================================

    @triton.jit
    def fp8_gemm_rowwise_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        a_s_ptr,
        b_s_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        input_block_size: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        """
        FP8 rowwise GEMM kernel.

        Similar to blockwise, but weight scales are per-row: shape [N]
        """
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        k_blocks = tl.cdiv(K, BLOCK_SIZE_K)

        offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

        # Activation scales (blockwise along K)
        a_s_k_blocks = tl.cdiv(K, input_block_size)
        a_s_ptrs = a_s_ptr + offs_m * a_s_k_blocks

        # Weight scales are per-row: [N], so load once per N tile
        b_s = tl.load(b_s_ptr + offs_n)  # [BLOCK_SIZE_N]

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_idx in range(k_blocks):
            k_start = k_idx * BLOCK_SIZE_K

            mask_k = offs_k < K - k_start
            a_fp8 = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b_fp8 = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)

            a_f32 = a_fp8.to(tl.float32)
            b_f32 = b_fp8.to(tl.float32)

            dot_result = tl.dot(a_f32, b_f32)

            k_scale_idx = k_start // input_block_size
            a_s = tl.load(a_s_ptrs + k_scale_idx)

            # For rowwise: b_s is per output row, applied after full K accumulation
            # But for efficiency, we apply a_s per K block and b_s at the end
            accumulator += dot_result * a_s[:, None]

            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K

        # Apply weight scales (per-row, so broadcast across M)
        accumulator *= b_s[None, :]

        c = accumulator.to(c_ptr.dtype.element_ty)
        offs_m_actual = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n_actual = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_m_actual[:, None] * N + offs_n_actual[None, :]
        mask = (offs_m_actual[:, None] < M) & (offs_n_actual[None, :] < N)
        tl.store(c_ptrs, c, mask=mask)

    def fp8_gemm_rowwise(
        a: torch.Tensor,
        a_s: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
        input_block_size: int = 128,
    ) -> torch.Tensor:
        """
        FP8 rowwise matrix multiplication.

        Args:
            a: FP8 activations [..., K]
            a_s: Activation scales [..., K//input_block_size]
            b: FP8 weights [N, K]
            b_s: Weight scales [N] (per-row)
            input_block_size: Block size for activation quantization

        Returns:
            Output tensor [..., N] in default float dtype
        """
        assert a.is_contiguous() and b.is_contiguous()
        assert a_s.is_contiguous() and b_s.is_contiguous()
        assert b_s.dim() == 1 and b_s.size(0) == b.size(0), "Rowwise: b_s must be [N]"

        K = a.size(-1)
        M = a.numel() // K
        N = b.shape[0]
        batch_shape = a.shape[:-1]

        c = a.new_empty(*batch_shape, N, dtype=torch.float16)

        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = input_block_size

        grid = (
            triton.cdiv(M, BLOCK_SIZE_M),
            triton.cdiv(N, BLOCK_SIZE_N),
        )

        fp8_gemm_rowwise_kernel[grid](
            a,
            b,
            c,
            a_s,
            b_s,
            M,
            N,
            K,
            input_block_size,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        return c

else:
    # Fallback stubs when Triton is not available
    def fp8_act_quant(x, block_size=128, dtype=torch.float8_e4m3fn):
        raise RuntimeError("Triton not available for FP8 kernels")

    def fp8_gemm_blockwise(a, a_s, b, b_s, input_block_size=128):
        raise RuntimeError("Triton not available for FP8 kernels")

    def fp8_addmm_blockwise(a, a_s, b, b_s, bias=None, input_block_size=128):
        raise RuntimeError("Triton not available for FP8 kernels")

    def fp8_gemm_rowwise(a, a_s, b, b_s, input_block_size=128):
        raise RuntimeError("Triton not available for FP8 kernels")
