"""
Block-wise INT8 Quantization Layout

Provides BlockWiseINT8Layout with optional Triton kernel acceleration.
Triton is lazy-imported - if user selects Triton backend but it's not installed,
a clear error message instructs them to install it.
"""

import torch
import os
import logging
from dataclasses import dataclass
from typing import Tuple, Optional

# Import from ComfyUI core (which re-exports from comfy_kitchen)
from comfy.quant_ops import QuantizedTensor, register_layout_op

# Import base layout types from comfy_kitchen
try:
    from comfy_kitchen.tensor import QuantizedLayout, BaseLayoutParams
except ImportError:
    # Fallback for older versions
    from comfy.quant_ops import QuantizedLayout
    BaseLayoutParams = object

# Lazy Triton import state
_triton_int8_available = None
_triton_functions = {}


def _check_triton_available():
    """
    Check and cache Triton INT8 kernel availability.
    
    Priority:
    1. Check comfy-kitchen triton backend (if ck is available)
    2. Fall back to independent Triton import check
    """
    global _triton_int8_available, _triton_functions

    if _triton_int8_available is not None:
        return _triton_int8_available

    # Check comfy-kitchen triton backend first
    try:
        from .. import is_ck_triton_available
        if is_ck_triton_available():
            # ck triton is available - try to load our kernels
            pass  # Fall through to kernel import below
        else:
            # ck says triton unavailable - still try our independent check
            pass
    except ImportError:
        # ck not available - use independent check
        pass

    # Try to import our Triton kernels
    try:
        from ..kernels.int8_kernels import (
            act_quant as triton_act_quant,
            act_dequant as triton_act_dequant,
            weight_quant as triton_weight_quant,
            weight_dequant as triton_weight_dequant,
            int8_gemm as triton_int8_gemm,
            int8_addmm as triton_int8_addmm,
        )

        _triton_functions["act_quant"] = triton_act_quant
        _triton_functions["act_dequant"] = triton_act_dequant
        _triton_functions["weight_quant"] = triton_weight_quant
        _triton_functions["weight_dequant"] = triton_weight_dequant
        _triton_functions["int8_gemm"] = triton_int8_gemm
        _triton_functions["int8_addmm"] = triton_int8_addmm
        _triton_int8_available = True
        logging.info("Triton INT8 kernels loaded successfully")
    except ImportError as e:
        _triton_int8_available = False
        logging.info(f"Triton INT8 kernels not available: {e}")

    return _triton_int8_available


def _get_triton_function(name):
    """Get a Triton function, raising helpful error if not available."""
    if not _check_triton_available():
        raise RuntimeError(
            f"Triton INT8 kernels not available. "
            f"To use Triton backend, install triton:\n"
            f"  Linux:   pip install triton\n"
            f"  Windows: pip install triton-windows\n"
            f"Or select 'pytorch' as kernel_backend."
        )
    return _triton_functions[name]


class BlockWiseINT8Layout(QuantizedLayout):
    """
    Block-wise INT8 quantization layout with optional Triton acceleration.

    Storage format:
    - qdata: INT8 tensor (torch.int8)
    - scale: Per-block scaling factors (float32)
    - block_size: Size of quantization blocks (default 128)
    - orig_dtype: Original dtype before quantization
    - is_weight: Whether this is a weight tensor

    Supports both Triton kernels (when available) and PyTorch fallback.
    """

    # Class-level setting for kernel backend
    use_triton = False  # Set via loader node

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        """Block-wise INT8 layout parameters."""
        block_size: int = 128
        is_weight: bool = False

    @classmethod
    def set_backend(cls, backend: str):
        """Set the kernel backend ('triton' or 'pytorch')."""
        if backend == "triton":
            if not _check_triton_available():
                raise RuntimeError(
                    "Triton INT8 kernels not available. "
                    "Install triton or use 'pytorch' backend."
                )
            cls.use_triton = True
        else:
            cls.use_triton = False

    @classmethod
    def quantize(
        cls, tensor, scale=None, block_size=128, is_weight=False, **kwargs
    ) -> Tuple[torch.Tensor, "BlockWiseINT8Layout.Params"]:
        """
        Quantize a tensor to INT8 with block-wise scaling.
        """
        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        if is_weight:
            assert (
                tensor.dim() == 2
            ), f"Weight tensor must be 2D, got shape {tensor.shape}"
            M, N = tensor.shape
            assert (
                M % block_size == 0 and N % block_size == 0
            ), f"Dimensions must be divisible by block_size={block_size}, got shape {tensor.shape}"

            # Use Triton if available and enabled
            if cls.use_triton and scale is None and tensor.is_cuda:
                try:
                    weight_quant = _get_triton_function("weight_quant")
                    qdata, scale = weight_quant(tensor, block_size=block_size)
                except Exception as e:
                    logging.warning(
                        f"Triton weight_quant failed: {e}, falling back to PyTorch"
                    )
                    qdata, scale = cls._weight_quantize_pytorch(
                        tensor, block_size, scale
                    )
            else:
                qdata, scale = cls._weight_quantize_pytorch(tensor, block_size, scale)
        else:
            K = tensor.shape[-1]
            assert (
                K % block_size == 0
            ), f"Last dimension must be divisible by block_size={block_size}, got {K}"

            if cls.use_triton and tensor.is_cuda:
                try:
                    act_quant = _get_triton_function("act_quant")
                    qdata, scale = act_quant(tensor, block_size=block_size)
                except Exception as e:
                    logging.warning(
                        f"Triton act_quant failed: {e}, falling back to PyTorch"
                    )
                    qdata, scale = cls._activation_quantize_pytorch(tensor, block_size)
            else:
                qdata, scale = cls._activation_quantize_pytorch(
                    tensor, block_size, scale
                )

        params = cls.Params(
            scale=scale.to(torch.float32),
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
            block_size=block_size,
            is_weight=is_weight,
        )

        return qdata, params

    @staticmethod
    def _weight_quantize_pytorch(tensor, block_size, scale=None):
        """PyTorch fallback for weight quantization."""
        M, N = tensor.shape
        tensor_blocked = tensor.reshape(
            M // block_size, block_size, N // block_size, block_size
        )
        tensor_blocked = tensor_blocked.permute(0, 2, 1, 3)

        if scale is None:
            amax = tensor_blocked.abs().amax(dim=(-2, -1))
            scale = amax / 127.0
            scale = torch.maximum(
                scale, torch.tensor(1e-8, device=scale.device, dtype=scale.dtype)
            )

        scale_broadcast = scale.unsqueeze(-1).unsqueeze(-1)
        tensor_scaled = tensor_blocked / scale_broadcast
        tensor_scaled = torch.clamp(tensor_scaled, -127.0, 127.0)
        qdata = tensor_scaled.to(torch.int8)
        qdata = qdata.permute(0, 2, 1, 3).reshape(M, N)
        return qdata, scale

    @staticmethod
    def _activation_quantize_pytorch(tensor, block_size, scale=None):
        """PyTorch fallback for activation quantization."""
        K = tensor.shape[-1]
        batch_shape = tensor.shape[:-1]
        tensor_blocked = tensor.reshape(*batch_shape, K // block_size, block_size)

        if scale is None:
            amax = tensor_blocked.abs().amax(dim=-1)
            scale = amax / 127.0
            scale = torch.maximum(
                scale, torch.tensor(1e-8, device=scale.device, dtype=scale.dtype)
            )

        scale_broadcast = scale.unsqueeze(-1)
        tensor_scaled = tensor_blocked / scale_broadcast
        tensor_scaled = torch.clamp(tensor_scaled, -127.0, 127.0)
        qdata = tensor_scaled.to(torch.int8).reshape(tensor.shape)
        return qdata, scale

    @staticmethod
    def dequantize(qdata, params) -> torch.Tensor:
        """Dequantize INT8 tensor back to original precision."""
        scale = params.scale
        block_size = params.block_size
        is_weight = params.is_weight
        orig_dtype = params.orig_dtype
        
        if not qdata.is_contiguous():
            qdata = qdata.contiguous()
        if not scale.is_contiguous():
            scale = scale.contiguous()

        output_dt = orig_dtype

        if is_weight:
            # Try Triton if enabled
            if BlockWiseINT8Layout.use_triton and qdata.dim() == 2 and qdata.is_cuda:
                try:
                    weight_dequant = _get_triton_function("weight_dequant")
                    return weight_dequant(
                        qdata, scale, block_size=block_size, output_dtype=output_dt
                    )
                except Exception as e:
                    logging.warning(
                        f"Triton weight_dequant failed: {e}, falling back to PyTorch"
                    )

            # PyTorch fallback
            M, N = qdata.shape
            expected_scale_shape = (M // block_size, N // block_size)
            if scale.shape != expected_scale_shape:
                expected_numel = (M // block_size) * (N // block_size)
                if scale.numel() == expected_numel:
                    scale = scale.reshape(expected_scale_shape)
                else:
                    raise RuntimeError(
                        f"Weight scale shape mismatch: scale.shape={scale.shape}, expected {expected_scale_shape}"
                    )
            qdata_blocked = qdata.reshape(
                M // block_size, block_size, N // block_size, block_size
            )
            qdata_blocked = qdata_blocked.permute(0, 2, 1, 3)
            scale_broadcast = scale.unsqueeze(-1).unsqueeze(-1)
            dequant = qdata_blocked.to(output_dt) * scale_broadcast
            dequant = dequant.permute(0, 2, 1, 3).reshape(M, N)
        else:
            # Activation dequantization
            if BlockWiseINT8Layout.use_triton and qdata.is_cuda:
                try:
                    act_dequant = _get_triton_function("act_dequant")
                    return act_dequant(
                        qdata, scale, block_size=block_size, output_dtype=output_dt
                    )
                except Exception as e:
                    logging.warning(
                        f"Triton act_dequant failed: {e}, falling back to PyTorch"
                    )

            # PyTorch fallback
            batch_shape = qdata.shape[:-1]
            K = qdata.shape[-1]
            expected_scale_shape = (*batch_shape, K // block_size)
            if scale.shape != expected_scale_shape:
                expected_numel = 1
                for dim in expected_scale_shape:
                    expected_numel *= dim
                if scale.numel() == expected_numel:
                    scale = scale.reshape(expected_scale_shape)
                else:
                    raise RuntimeError(
                        f"Activation scale shape mismatch: scale.shape={scale.shape}, expected {expected_scale_shape}"
                    )
            qdata_blocked = qdata.reshape(*batch_shape, K // block_size, block_size)
            scale_broadcast = scale.unsqueeze(-1)
            dequant = qdata_blocked.to(output_dt) * scale_broadcast
            dequant = dequant.reshape(qdata.shape)

        return dequant

    @classmethod
    def get_plain_tensors(cls, qtensor):
        """Extract raw tensors for computation."""
        return (
            qtensor._qdata,
            qtensor._params.scale,
            qtensor._params.block_size,
            qtensor._params.is_weight,
        )


# ==============================================================================
# Logging Helpers
# ==============================================================================

# Call counters to avoid log spam
_int8_path_counts = {"NATIVE_TRITON": 0, "PYTORCH_BOTH_QUANT": 0, "DEQUANT_FALLBACK": 0, "DYNAMIC_ACT_QUANT": 0}
_LOG_LIMIT = 3  # Log first N occurrences per path, then summary


def _log_int8_path(path_type, input_shape, weight_shape, reason=None):
    """Log which INT8 code path is being used (always, but rate-limited)."""
    global _int8_path_counts

    count = _int8_path_counts.get(path_type, 0)
    _int8_path_counts[path_type] = count + 1

    if count < _LOG_LIMIT:
        if path_type == "NATIVE_TRITON":
            logging.info(
                f"INT8: Native Triton matmul - input={input_shape}, weight={weight_shape}"
            )
        elif path_type == "PYTORCH_BOTH_QUANT":
            logging.info(
                f"INT8: PyTorch fallback (both quantized) - input={input_shape}, weight={weight_shape}"
            )
        elif path_type == "DYNAMIC_ACT_QUANT":
            logging.info(
                f"INT8: Dynamic activation quant - input={input_shape}, weight={weight_shape}"
            )
        elif path_type == "DEQUANT_FALLBACK":
            logging.warning(
                f"INT8: Dequant fallback ({reason}) - input={input_shape}, weight={weight_shape}. "
                f"Native INT8 matmul NOT used - only memory savings, no compute speedup."
            )
    elif count == _LOG_LIMIT:
        logging.info(
            f"INT8: Suppressing further '{path_type}' logs (limit={_LOG_LIMIT})"
        )


# ==============================================================================
# Operation Handlers
# ==============================================================================


def _int8_gemm_pytorch_fallback(
    a_int8, a_scale, b_int8, b_scale, block_size, bias=None
):
    """PyTorch fallback for INT8 matmul: dequantize and use standard matmul."""
    K = a_int8.shape[-1]
    batch_shape = a_int8.shape[:-1]
    N = b_int8.shape[0]

    # Dequantize activations
    expected_scale_shape = (*batch_shape, K // block_size)
    if a_scale.shape != expected_scale_shape:
        expected_numel = 1
        for dim in expected_scale_shape:
            expected_numel *= dim
        if a_scale.numel() == expected_numel:
            a_scale = a_scale.reshape(expected_scale_shape)

    a_blocked = a_int8.reshape(*batch_shape, K // block_size, block_size)
    a_scale_broadcast = a_scale.unsqueeze(-1)
    a_fp32 = a_blocked.to(torch.float32) * a_scale_broadcast
    a_fp32 = a_fp32.reshape(*batch_shape, K)

    # Dequantize weights
    expected_weight_scale_shape = (N // block_size, K // block_size)
    if b_scale.shape != expected_weight_scale_shape:
        expected_weight_numel = (N // block_size) * (K // block_size)
        if b_scale.numel() == expected_weight_numel:
            b_scale = b_scale.reshape(expected_weight_scale_shape)

    b_blocked = b_int8.reshape(N // block_size, block_size, K // block_size, block_size)
    b_blocked = b_blocked.permute(0, 2, 1, 3)
    b_scale_broadcast = b_scale.unsqueeze(-1).unsqueeze(-1)
    b_fp32 = b_blocked.to(torch.float32) * b_scale_broadcast
    b_fp32 = b_fp32.permute(0, 2, 1, 3).reshape(N, K)

    # Bias may arrive in bfloat16 when previous layers (e.g. TensorWiseINT8Layout)
    # output bfloat16 and cast_bias_weight matches the input dtype.
    # The INT8 fallback computes in float32, so bias must match.
    if bias is not None and bias.dtype != a_fp32.dtype:
        bias = bias.to(a_fp32.dtype)

    output = torch.nn.functional.linear(a_fp32, b_fp32, bias)
    return output


@register_layout_op(torch.ops.aten.linear.default, BlockWiseINT8Layout)
def int8_linear(func, args, kwargs):
    """Block-wise INT8 linear operation."""
    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None

    # Case 1: Both input and weight are quantized
    if isinstance(input_tensor, QuantizedTensor) and isinstance(
        weight, QuantizedTensor
    ):
        a_int8, a_scale, a_block_size, _ = BlockWiseINT8Layout.get_plain_tensors(
            input_tensor
        )
        b_int8, b_scale, b_block_size, _ = BlockWiseINT8Layout.get_plain_tensors(weight)

        out_dtype = input_tensor._params.orig_dtype

        # Try Triton if enabled
        if BlockWiseINT8Layout.use_triton and a_int8.is_cuda:
            try:
                int8_addmm = (
                    _get_triton_function("int8_addmm")
                    if bias is not None
                    else _get_triton_function("int8_gemm")
                )
                a_2d = a_int8.reshape(-1, a_int8.shape[-1]).contiguous()
                a_scale_2d = a_scale.reshape(-1, a_scale.shape[-1]).contiguous()

                if bias is not None:
                    result = int8_addmm(
                        a_2d,
                        a_scale_2d,
                        b_int8.contiguous(),
                        b_scale.contiguous(),
                        bias,
                    )
                else:
                    result = _get_triton_function("int8_gemm")(
                        a_2d, a_scale_2d, b_int8.contiguous(), b_scale.contiguous()
                    )

                if result.dtype != out_dtype:
                    result = result.to(out_dtype)

                _log_int8_path("NATIVE_TRITON", a_int8.shape, b_int8.shape)
                return result.reshape(*input_tensor.shape[:-1], weight.shape[0])
            except Exception as e:
                logging.warning(
                    f"Triton INT8 matmul failed: {e}, falling back to PyTorch"
                )

        # PyTorch fallback (both quantized, but using dequant path)
        _log_int8_path("PYTORCH_BOTH_QUANT", a_int8.shape, b_int8.shape)
        output = _int8_gemm_pytorch_fallback(
            a_int8, a_scale, b_int8, b_scale, a_block_size, bias
        )
        if output.dtype != out_dtype:
            output = output.to(out_dtype)
        return output

    # Case 2: Weight is quantized, input is NOT - dynamically quantize input
    if isinstance(weight, QuantizedTensor) and not isinstance(input_tensor, QuantizedTensor):
        b_int8, b_scale, b_block_size, _ = BlockWiseINT8Layout.get_plain_tensors(weight)
        out_dtype = weight._params.orig_dtype
        
        # Ensure input is contiguous and on same device
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()
        if input_tensor.device != b_int8.device:
            input_tensor = input_tensor.to(b_int8.device)
        
        # Dynamically quantize input activation to INT8
        orig_shape = input_tensor.shape
        K = input_tensor.shape[-1]
        
        # Check if K is divisible by block_size, pad if needed
        if K % b_block_size != 0:
            # Fall back to dequant if dimensions don't align
            _log_int8_path(
                "DEQUANT_FALLBACK",
                input_tensor.shape,
                b_int8.shape,
                reason=f"K={K} not divisible by block_size={b_block_size}"
            )
            weight_dequant = weight.dequantize()
            return torch.nn.functional.linear(input_tensor, weight_dequant, bias)
        
        # Quantize activation on-the-fly
        a_int8, a_params = BlockWiseINT8Layout.quantize(
            input_tensor,
            block_size=b_block_size,
            is_weight=False
        )
        a_scale = a_params.scale
        
        _log_int8_path("DYNAMIC_ACT_QUANT", a_int8.shape, b_int8.shape)
        
        # Try Triton path
        if BlockWiseINT8Layout.use_triton and a_int8.is_cuda:
            try:
                int8_addmm = (
                    _get_triton_function("int8_addmm")
                    if bias is not None
                    else _get_triton_function("int8_gemm")
                )
                a_2d = a_int8.reshape(-1, a_int8.shape[-1]).contiguous()
                a_scale_2d = a_scale.reshape(-1, a_scale.shape[-1]).contiguous()

                if bias is not None:
                    result = int8_addmm(
                        a_2d,
                        a_scale_2d,
                        b_int8.contiguous(),
                        b_scale.contiguous(),
                        bias,
                    )
                else:
                    result = _get_triton_function("int8_gemm")(
                        a_2d, a_scale_2d, b_int8.contiguous(), b_scale.contiguous()
                    )

                if result.dtype != out_dtype:
                    result = result.to(out_dtype)

                return result.reshape(*orig_shape[:-1], weight.shape[0])
            except Exception as e:
                logging.warning(
                    f"Triton INT8 matmul failed: {e}, using PyTorch fallback"
                )
        
        # PyTorch fallback for dynamic quant path
        output = _int8_gemm_pytorch_fallback(
            a_int8, a_scale, b_int8, b_scale, b_block_size, bias
        )
        if output.dtype != out_dtype:
            output = output.to(out_dtype)
        return output.reshape(*orig_shape[:-1], weight.shape[0])

    # Case 3: Fallback - dequantize (rare - usually one of above cases applies)
    _log_int8_path(
        "DEQUANT_FALLBACK",
        input_tensor.shape if hasattr(input_tensor, "shape") else None,
        weight.shape
        if hasattr(weight, "shape")
        else weight._qdata.shape
        if isinstance(weight, QuantizedTensor)
        else None,
        reason="unexpected case",
    )

    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    return torch.nn.functional.linear(input_tensor, weight, bias)


@register_layout_op(torch.ops.aten.mm.default, BlockWiseINT8Layout)
def int8_mm(func, args, kwargs):
    """Block-wise INT8 matmul."""
    input_tensor = args[0]
    weight = args[1]

    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    return func(input_tensor, weight)


@register_layout_op(torch.ops.aten.addmm.default, BlockWiseINT8Layout)
def int8_addmm(func, args, kwargs):
    """Block-wise INT8 addmm."""
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


@register_layout_op(torch.ops.aten.view.default, BlockWiseINT8Layout)
@register_layout_op(torch.ops.aten.t.default, BlockWiseINT8Layout)
def int8_func(func, args, kwargs):
    """Handle view/transpose for INT8 tensors."""
    input_tensor = args[0]
    if isinstance(input_tensor, QuantizedTensor):
        qdata = input_tensor._qdata
        ar = list(args)
        ar[0] = qdata
        new_qdata = func(*ar, **kwargs)
        # Use _copy_with to preserve params
        return input_tensor._copy_with(qdata=new_qdata)
    return func(*args, **kwargs)
