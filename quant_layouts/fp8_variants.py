"""
FP8 Variant Layouts (Row-wise and Block-wise scaling)

These layouts extend ComfyUI's base TensorCoreFP8Layout with finer-grained scaling:
- RowWiseFP8Layout: One scale per row (M scales for MxN weight)
- BlockWiseFP8Layout: One scale per 2D block (MxN blocks)

When Triton is available, these layouts use native FP8 matmul kernels.
Otherwise, they fall back to dequantization.
"""

import torch
import logging
from dataclasses import dataclass
from typing import Tuple, Optional

from ..utils.logging_utils import get_one_time_logger

logger = get_one_time_logger(__name__)

# Import from ComfyUI core (which re-exports from comfy_kitchen)
from comfy.quant_ops import QuantizedTensor, register_layout_op

# Import base layout types from comfy_kitchen
try:
    from comfy_kitchen.tensor import QuantizedLayout, BaseLayoutParams
except ImportError:
    # Fallback for older versions
    from comfy.quant_ops import QuantizedLayout
    BaseLayoutParams = object  # Will fail at runtime if used

# Check comfy-kitchen triton backend first, then independent check
def _should_use_fp8_kernels():
    """Check FP8 Triton kernel availability using ck or independent check."""
    try:
        from .. import is_ck_triton_available
        if is_ck_triton_available():
            return True
    except ImportError:
        pass

    # Fall back to independent check
    try:
        from ..kernels.fp8_kernels import _check_triton_available
        return _check_triton_available()
    except ImportError:
        return False


# Try to import FP8 Triton kernels
try:
    from ..kernels.fp8_kernels import (
        _check_triton_available,
        fp8_act_quant,
        fp8_gemm_blockwise,
        fp8_addmm_blockwise,
        fp8_gemm_rowwise,
    )

    _HAS_FP8_KERNELS = _should_use_fp8_kernels()
except ImportError:
    _HAS_FP8_KERNELS = False
    logger.debug("FP8 Triton kernels not available, using dequantize fallback")


class RowWiseFP8Layout(QuantizedLayout):
    """
    Row-wise FP8 quantization layout.

    Storage format:
    - qdata: FP8 tensor (torch.float8_e4m3fn)
    - scale: Per-row scaling factors, shape (out_features,) - stored as dequant scale
    - orig_dtype: Original dtype before quantization
    """

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        """Row-wise FP8 layout parameters. Inherits scale, orig_dtype, orig_shape."""
        pass  # No additional fields needed - BaseLayoutParams has all we need

    @classmethod
    def quantize(
        cls, tensor, scale=None, dtype=torch.float8_e4m3fn, **kwargs
    ) -> Tuple[torch.Tensor, "RowWiseFP8Layout.Params"]:
        """
        Quantize a 2D tensor with row-wise scaling.
        """
        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if tensor.ndim != 2:
            raise ValueError(
                f"RowWiseFP8Layout requires 2D tensor, got shape {tensor.shape}"
            )

        M, N = tensor.shape
        fp8_max = torch.finfo(dtype).max

        if scale is None:
            # Compute per-row absolute maximum
            row_max = tensor.abs().amax(dim=1, keepdim=True)  # (M, 1)
            quant_scale = fp8_max / row_max.clamp_min(1e-12)  # (M, 1)
        else:
            # scale is provided as dequant scale, convert to quant scale
            quant_scale = (
                (1.0 / scale).unsqueeze(1) if scale.ndim == 1 else (1.0 / scale)
            )

        # Apply scale per-row
        tensor_scaled = tensor * quant_scale

        # Clamp and convert
        tensor_scaled = torch.clamp(tensor_scaled, -fp8_max, fp8_max)
        qdata = tensor_scaled.to(dtype)

        # Store dequant scale (reciprocal of quant scale)
        dequant_scale = (1.0 / quant_scale).squeeze(1)  # (M,)

        params = cls.Params(
            scale=dequant_scale.to(torch.float32),
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
        )
        return qdata, params

    @staticmethod
    def dequantize(qdata, params) -> torch.Tensor:
        """Dequantize FP8 tensor with row-wise scaling."""
        scale = params.scale
        orig_dtype = params.orig_dtype

        # Convert to target dtype (matching core ComfyUI pattern)
        plain_tensor = torch.ops.aten._to_copy.default(qdata, dtype=orig_dtype)
        # Cast scale to orig_dtype before in-place multiply to preserve output dtype
        scale_broadcast = scale.to(
            dtype=orig_dtype, device=plain_tensor.device
        ).unsqueeze(1)  # (M, 1)
        plain_tensor.mul_(scale_broadcast)
        return plain_tensor

    @classmethod
    def get_plain_tensors(cls, qtensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract raw tensors for computation."""
        return qtensor._qdata, qtensor._params.scale


class BlockWiseFP8Layout(QuantizedLayout):
    """
    True 2D block-wise FP8 quantization layout.

    Storage format:
    - qdata: FP8 tensor (torch.float8_e4m3fn)
    - scale: Per-block scaling factors, shape (M//block_size, N//block_size)
    - block_size: Size of quantization blocks
    - orig_dtype: Original dtype before quantization
    """

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        """Block-wise FP8 layout parameters."""
        block_size: int = 64

    @classmethod
    def quantize(
        cls, tensor, scale=None, block_size=64, dtype=torch.float8_e4m3fn, **kwargs
    ) -> Tuple[torch.Tensor, "BlockWiseFP8Layout.Params"]:
        """
        Quantize a 2D tensor with 2D block-wise scaling.
        """
        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if tensor.ndim != 2:
            raise ValueError(
                f"BlockWiseFP8Layout requires 2D tensor, got shape {tensor.shape}"
            )

        M, N = tensor.shape

        if M % block_size != 0 or N % block_size != 0:
            raise ValueError(
                f"BlockWiseFP8Layout requires dimensions divisible by block_size={block_size}. "
                f"Got shape ({M}, {N})"
            )

        fp8_max = torch.finfo(dtype).max

        # Reshape to 2D blocks
        tensor_blocked = tensor.reshape(
            M // block_size, block_size, N // block_size, block_size
        )
        tensor_blocked = tensor_blocked.permute(0, 2, 1, 3)  # (M//bs, N//bs, bs, bs)

        if scale is None:
            # Compute per-block absolute maximum
            block_max = tensor_blocked.abs().amax(dim=(2, 3))  # (M//bs, N//bs)
            quant_scale = fp8_max / block_max.clamp_min(1e-12)
        else:
            quant_scale = 1.0 / scale

        # Apply scale per-block
        scale_broadcast = quant_scale.unsqueeze(-1).unsqueeze(-1)
        tensor_scaled = tensor_blocked * scale_broadcast

        # Clamp and convert
        tensor_scaled = torch.clamp(tensor_scaled, -fp8_max, fp8_max)
        qdata_blocked = tensor_scaled.to(dtype)

        # Reshape back
        qdata = qdata_blocked.permute(0, 2, 1, 3).reshape(M, N)
        dequant_scale = 1.0 / quant_scale

        params = cls.Params(
            scale=dequant_scale.to(torch.float32),
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
            block_size=block_size,
        )
        return qdata, params

    @staticmethod
    def dequantize(qdata, params) -> torch.Tensor:
        """Dequantize FP8 tensor with 2D block-wise scaling."""
        scale = params.scale
        block_size = params.block_size
        orig_dtype = params.orig_dtype

        M, N = qdata.shape

        # Reshape to blocks
        qdata_blocked = qdata.reshape(
            M // block_size, block_size, N // block_size, block_size
        )
        qdata_blocked = qdata_blocked.permute(0, 2, 1, 3)

        # Convert to target dtype (matching core ComfyUI pattern)
        dequantized = torch.ops.aten._to_copy.default(qdata_blocked, dtype=orig_dtype)

        # Cast scale to orig_dtype before in-place multiply to preserve output dtype
        scale_broadcast = (
            scale.to(dtype=orig_dtype, device=dequantized.device)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        dequantized.mul_(scale_broadcast)

        # Reshape back
        dequantized = dequantized.permute(0, 2, 1, 3).reshape(M, N)
        return dequantized

    @classmethod
    def get_plain_tensors(cls, qtensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Extract raw tensors for computation."""
        return (
            qtensor._qdata,
            qtensor._params.scale,
            qtensor._params.block_size,
        )


# ==============================================================================
# Operation Handlers (dequant-fallback for both layouts)
# ==============================================================================


@register_layout_op(torch.ops.aten.linear.default, RowWiseFP8Layout)
def rowwise_fp8_linear(func, args, kwargs):
    """Row-wise FP8 linear operation with native kernel support."""
    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None

    # Try native FP8 kernel if weight is quantized and Triton available
    if _HAS_FP8_KERNELS and isinstance(weight, QuantizedTensor):
        w_qdata, w_scale = RowWiseFP8Layout.get_plain_tensors(weight)

        # Check if weight is on CUDA (required for Triton)
        if w_qdata.is_cuda:
            orig_dtype = weight._params.orig_dtype
            # Use a reasonable block size for activation quantization
            act_block_size = 128

            # Input needs to be quantized for rowwise kernel
            if input_tensor.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                try:
                    # Quantize input to FP8 blockwise
                    a_qdata, a_scale = fp8_act_quant(
                        input_tensor.to(device=w_qdata.device),
                        block_size=act_block_size,
                        dtype=w_qdata.dtype,
                    )

                    logger.debug(
                        f"FP8 rowwise: Native kernel (dynamic quant), "
                        f"input={a_qdata.shape}, weight={w_qdata.shape}"
                    )

                    # For rowwise, bias needs manual addition (no fused kernel yet)
                    result = fp8_gemm_rowwise(
                        a_qdata,
                        a_scale,
                        w_qdata,
                        w_scale,
                        input_block_size=act_block_size,
                    )

                    if bias is not None:
                        result = result + bias.to(
                            device=result.device, dtype=result.dtype
                        )

                    return result.to(orig_dtype)
                except Exception as e:
                    logger.warning(f"FP8 rowwise native kernel failed: {e}")

    # Fallback: dequantize
    logger.debug("FP8 rowwise: Using dequant fallback")
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    return torch.nn.functional.linear(input_tensor, weight, bias)


@register_layout_op(torch.ops.aten.mm.default, RowWiseFP8Layout)
def rowwise_fp8_mm(func, args, kwargs):
    """Row-wise FP8 matrix multiplication (dequant-fallback)."""
    input_tensor = args[0]
    weight = args[1]

    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    return func(input_tensor, weight)


@register_layout_op(torch.ops.aten.addmm.default, RowWiseFP8Layout)
def rowwise_fp8_addmm(func, args, kwargs):
    """Row-wise FP8 addmm operation (dequant-fallback)."""
    bias = args[0]
    input_tensor = args[1]
    weight = args[2]

    if isinstance(bias, QuantizedTensor):
        bias = bias.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()

    return func(bias, input_tensor, weight, **kwargs)


@register_layout_op(torch.ops.aten.view.default, RowWiseFP8Layout)
@register_layout_op(torch.ops.aten.t.default, RowWiseFP8Layout)
def rowwise_fp8_func(func, args, kwargs):
    """Handle view/transpose for row-wise FP8 tensors."""
    input_tensor = args[0]
    if isinstance(input_tensor, QuantizedTensor):
        plain_input, scale = RowWiseFP8Layout.get_plain_tensors(input_tensor)
        ar = list(args)
        ar[0] = plain_input
        # Use _copy_with to preserve params
        return input_tensor._copy_with(qdata=func(*ar, **kwargs))
    return func(*args, **kwargs)


@register_layout_op(torch.ops.aten.linear.default, BlockWiseFP8Layout)
def blockwise_fp8_linear(func, args, kwargs):
    """Block-wise FP8 linear operation with native kernel support."""
    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None

    # Try native FP8 kernel if both are quantized and Triton available
    if _HAS_FP8_KERNELS and isinstance(weight, QuantizedTensor):
        w_qdata, w_scale, w_block_size = BlockWiseFP8Layout.get_plain_tensors(weight)

        # Check if weight is on CUDA (required for Triton)
        if w_qdata.is_cuda:
            orig_dtype = weight._params.orig_dtype

            # If input is already quantized FP8, use it directly
            if isinstance(input_tensor, QuantizedTensor):
                a_qdata, a_scale, a_block_size = BlockWiseFP8Layout.get_plain_tensors(
                    input_tensor
                )

                logger.debug(
                    f"FP8 blockwise: Native kernel (both quantized), "
                    f"input={a_qdata.shape}, weight={w_qdata.shape}, block_size={w_block_size}"
                )

                try:
                    if bias is not None:
                        result = fp8_addmm_blockwise(
                            a_qdata,
                            a_scale,
                            w_qdata,
                            w_scale,
                            bias=bias.to(device=a_qdata.device),
                            input_block_size=w_block_size,
                        )
                    else:
                        result = fp8_gemm_blockwise(
                            a_qdata,
                            a_scale,
                            w_qdata,
                            w_scale,
                            input_block_size=w_block_size,
                        )
                    return result.to(orig_dtype)
                except Exception as e:
                    logger.warning(
                        f"FP8 native kernel failed, falling back to dequant: {e}"
                    )

            # Input is not quantized - quantize it dynamically
            elif input_tensor.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                try:
                    # Quantize input to FP8
                    a_qdata, a_scale = fp8_act_quant(
                        input_tensor.to(device=w_qdata.device),
                        block_size=w_block_size,
                        dtype=w_qdata.dtype,
                    )

                    logger.debug(
                        f"FP8 blockwise: Native kernel (dynamic quant), "
                        f"input={a_qdata.shape}, weight={w_qdata.shape}"
                    )

                    if bias is not None:
                        result = fp8_addmm_blockwise(
                            a_qdata,
                            a_scale,
                            w_qdata,
                            w_scale,
                            bias=bias.to(device=a_qdata.device),
                            input_block_size=w_block_size,
                        )
                    else:
                        result = fp8_gemm_blockwise(
                            a_qdata,
                            a_scale,
                            w_qdata,
                            w_scale,
                            input_block_size=w_block_size,
                        )
                    return result.to(orig_dtype)
                except Exception as e:
                    logger.warning(
                        f"FP8 dynamic quant failed, falling back to dequant: {e}"
                    )

    # Fallback: dequantize
    logger.debug("FP8 blockwise: Using dequant fallback")
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    return torch.nn.functional.linear(input_tensor, weight, bias)


@register_layout_op(torch.ops.aten.mm.default, BlockWiseFP8Layout)
def blockwise_fp8_mm(func, args, kwargs):
    """Block-wise FP8 matrix multiplication (dequant-fallback)."""
    input_tensor = args[0]
    weight = args[1]

    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    return func(input_tensor, weight)


@register_layout_op(torch.ops.aten.addmm.default, BlockWiseFP8Layout)
def blockwise_fp8_addmm(func, args, kwargs):
    """Block-wise FP8 addmm operation (dequant-fallback)."""
    bias = args[0]
    input_tensor = args[1]
    weight = args[2]

    if isinstance(bias, QuantizedTensor):
        bias = bias.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()

    return func(bias, input_tensor, weight, **kwargs)


@register_layout_op(torch.ops.aten.view.default, BlockWiseFP8Layout)
@register_layout_op(torch.ops.aten.t.default, BlockWiseFP8Layout)
def blockwise_fp8_func(func, args, kwargs):
    """Handle view/transpose for block-wise FP8 tensors."""
    input_tensor = args[0]
    if isinstance(input_tensor, QuantizedTensor):
        plain_input, scale, block_size = BlockWiseFP8Layout.get_plain_tensors(
            input_tensor
        )
        ar = list(args)
        ar[0] = plain_input
        # Use _copy_with to preserve params
        return input_tensor._copy_with(qdata=func(*ar, **kwargs))
    return func(*args, **kwargs)
