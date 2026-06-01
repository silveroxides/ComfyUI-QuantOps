import torch
import logging
from comfy.quant_ops import register_layout_op, QuantizedTensor

try:
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout, tensorwise_int8_linear as orig_linear
    CK_AVAILABLE = True
except ImportError:
    from comfy.quant_ops import QuantizedLayout, BaseLayoutParams
    from dataclasses import dataclass
    
    class TensorWiseINT8Layout(QuantizedLayout):
        """Fallback Tensor-wise INT8 quantization when comfy_kitchen is not available."""
        @dataclass(frozen=True)
        class Params(BaseLayoutParams):
            weight_scale: torch.Tensor
            
        @classmethod
        def quantize(cls, plain_tensor: torch.Tensor, device: torch.device = None) -> QuantizedTensor:
            raise NotImplementedError("Fallback TensorWiseINT8Layout does not support on-the-fly quantization.")

        @classmethod
        def dequantize(cls, qdata: torch.Tensor, params: Params) -> torch.Tensor:
            scale = params.weight_scale
            # match output dtype logic if needed, typically bf16 or float16
            return qdata.to(scale.dtype) * scale

        @classmethod
        def get_plain_tensors(cls, qtensor: QuantizedTensor):
            return qtensor.tensor, qtensor.params.weight_scale, None, None

        @classmethod
        def state_dict_tensors(cls, qdata: torch.Tensor, params: Params) -> dict[str, torch.Tensor]:
            return {"weight": qdata, "weight_scale": params.weight_scale}

        @classmethod
        def supports_fast_matmul(cls) -> bool:
            return True
            
    CK_AVAILABLE = False
    orig_linear = None

from .int8_layout import _get_triton_function

@register_layout_op(torch.ops.aten.linear.default, "TensorWiseINT8Layout")
def tensorwise_int8_linear_patched(func, args, kwargs):
    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None
    
    # Import locally to avoid circular dependency with __init__.py
    triton_avail = False
    try:
        from .. import is_ck_triton_available
        triton_avail = is_ck_triton_available()
    except Exception:
        triton_avail = False
        
    logging.debug(f"Executing tensorwise_int8_linear_patched, CK_AVAILABLE={CK_AVAILABLE}, triton_avail={triton_avail}")

    # Determine if we should use our fallback.
    # We use our fallback if:
    # 1. CK is not available (obviously)
    # 2. CK triton backend is NOT available (fallback is better than eager)
    # 3. Scale is per-channel (CK might not support per-channel, even if triton is enabled)
    
    if isinstance(weight, QuantizedTensor):
        plain_weight, scale_b, _, _ = TensorWiseINT8Layout.get_plain_tensors(weight)
        
        force_fallback = not CK_AVAILABLE or not triton_avail or scale_b.numel() > 1
        logging.debug(f"Weight is QuantizedTensor, scale_b numel={scale_b.numel()}, force_fallback={force_fallback}")
        
        if force_fallback:
            try:
                # Use the reliable loader from sibling int8_layout
                int8_linear_per_channel = _get_triton_function("int8_linear_per_channel")
                logging.debug("Attempting to use int8_linear_per_channel fallback")
                plain_input = input_tensor.dequantize() if isinstance(input_tensor, QuantizedTensor) else input_tensor
                # Ensure we pass the device and use correct output dtype (default is input dtype)
                return int8_linear_per_channel(plain_input, plain_weight, scale_b, bias)
            except Exception as e:
                logging.warning(f"Fallback INT8 linear failed: {e}")
                plain_input = input_tensor.dequantize() if isinstance(input_tensor, QuantizedTensor) else input_tensor
                return torch.nn.functional.linear(plain_input, weight.dequantize(), bias)

    if orig_linear is not None:
        logging.debug("Dispatching to orig_linear from comfy_kitchen")
        return orig_linear(func, args, kwargs)
    else:
        # Should not reach here for quantized weight, but just in case
        logging.debug("Dispatching to basic torch.nn.functional.linear dequantize fallback")
        plain_input = input_tensor.dequantize() if isinstance(input_tensor, QuantizedTensor) else input_tensor
        plain_weight = weight.dequantize() if isinstance(weight, QuantizedTensor) else weight
        return torch.nn.functional.linear(plain_input, plain_weight, bias)

if CK_AVAILABLE:
    logging.info("ComfyUI-QuantOps: Patched TensorWiseINT8Layout for comprehensive fallback")
else:
    logging.info("ComfyUI-QuantOps: Registered standalone fallback TensorWiseINT8Layout")
