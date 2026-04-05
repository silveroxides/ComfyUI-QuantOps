"""
Unified Custom Operations for Quantized Models.

This module provides a single UnifiedQuantOps class that automatically handles
any mix of INT8, FP8, MXFP8, and NVFP4 quantized layers in the same model.
It relies on per-tensor layout parameters from comfy_quant metadata and uses 
QuantizedTensor dispatch to avoid dequantization whenever possible.
"""

import json
import torch
import logging
import torch.nn.functional as F
from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
from comfy.quant_ops import QuantizedTensor, QUANT_ALGOS, get_layout_class
from comfy.model_patcher import LowVramPatch
from unifiedefficientloader import tensor_to_dict

# Try to import INT8 layouts
try:
    from comfy_kitchen.tensor.int8 import BlockWiseINT8Layout
    _HAS_INT8_LAYOUT = True
except ImportError:
    try:
        from .quant_layouts.int8_layout import BlockWiseINT8Layout
        _HAS_INT8_LAYOUT = True
    except ImportError:
        _HAS_INT8_LAYOUT = False
        logging.warning("INT8 blockwise layout not available")

try:
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
    _HAS_TENSORWISE_INT8_LAYOUT = True
except ImportError:
    _HAS_TENSORWISE_INT8_LAYOUT = False
    logging.warning("INT8 tensorwise layout not available from comfy_kitchen")


class UnifiedQuantOps(manual_cast):
    """
    Unified operations class that handles INT8, FP8, MXFP8, and NVFP4 formats.
    """

    class Linear(manual_cast.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.scale_weight = None
            self.block_size = None
            self.is_quantized = False
            self.layout_type = None
            self.quant_format = None

        def reset_parameters(self):
            return None

        def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            weight_key = prefix + "weight"

            # 1. Safely pop all possible scale keys
            scale_weight_key_old = prefix + "scale_weight"
            scale_weight_key_new = prefix + "weight_scale"
            
            scale = state_dict.pop(scale_weight_key_old, None)
            if scale is None:
                scale = state_dict.pop(scale_weight_key_new, None)

            scale_2 = state_dict.pop(prefix + "weight_scale_2", None)
            scalar = state_dict.pop(prefix + "weight_scalar", None)

            # Clean up other scales not used for weight
            state_dict.pop(prefix + "input_scale", None)
            state_dict.pop(prefix + "scale_input", None)

            # 2. Parse comfy_quant metadata
            comfy_quant_tensor = state_dict.pop(prefix + "comfy_quant", None)
            layer_conf = {}

            if comfy_quant_tensor is not None:
                try:
                    layer_conf = json.loads(comfy_quant_tensor.numpy().tobytes())
                except Exception as e:
                    # Fallback to tensor_to_dict
                    layer_conf = tensor_to_dict(comfy_quant_tensor)

            self.quant_format = layer_conf.get("format", None)
            self.block_size = layer_conf.get("group_size", None)

            # 3. Load weight and initialize QuantizedTensor based on dtype
            weight_tensor = state_dict.pop(weight_key, None)

            if weight_tensor is not None:
                is_nvfp4 = (
                    self.quant_format == "nvfp4" or 
                    (weight_tensor.dtype == torch.uint8 and scale_2 is not None)
                )

                # --- NVFP4 ---
                if is_nvfp4:
                    self.is_quantized = True
                    self.layout_type = "TensorCoreNVFP4Layout"
                    self.block_size = 16  # NVFP4 uses 16x16 blocks
                    
                    from comfy.quant_ops import TensorCoreNVFP4Layout
                    
                    orig_dtype_str = layer_conf.get("orig_dtype", "torch.bfloat16") if layer_conf else "torch.bfloat16"
                    DTYPE_MAP = {
                        "torch.bfloat16": torch.bfloat16,
                        "torch.float16": torch.float16,
                        "torch.float32": torch.float32,
                    }
                    orig_dtype = DTYPE_MAP.get(orig_dtype_str, torch.bfloat16)
                    
                    if layer_conf and "orig_shape" in layer_conf:
                        orig_shape = tuple(layer_conf["orig_shape"])
                    else:
                        orig_shape = (weight_tensor.shape[0], weight_tensor.shape[1] * 2)
                    
                    layout_params = TensorCoreNVFP4Layout.Params(
                        scale=scale_2.to(torch.float32) if scale_2 is not None else torch.tensor(1.0),
                        orig_dtype=orig_dtype,
                        orig_shape=orig_shape,
                        block_scale=scale,
                    )
                    
                    self.weight = torch.nn.Parameter(
                        QuantizedTensor(weight_tensor, self.layout_type, layout_params),
                        requires_grad=False,
                    )
                    
                # --- INT8 ---
                elif weight_tensor.dtype == torch.int8:
                    self.is_quantized = True
                    self.scale_weight = scale
                    
                    if self.block_size is None:
                        self.block_size = 128
                        
                    is_tensorwise = self.quant_format == 'int8_tensorwise' or (self.quant_format is None and (scale is not None and (scale.ndim == 0 or (scale.ndim == 1 and scale.shape[0] == 1))))
                    
                    if is_tensorwise and _HAS_TENSORWISE_INT8_LAYOUT:
                        self.layout_type = "TensorWiseINT8Layout"
                        layout_params = TensorWiseINT8Layout.Params(
                            scale=scale.to(torch.float32) if scale is not None else None,
                            orig_dtype=torch.bfloat16,
                            orig_shape=tuple(weight_tensor.shape),
                            is_weight=True,
                        )
                        self.weight = torch.nn.Parameter(
                            QuantizedTensor(weight_tensor, self.layout_type, layout_params),
                            requires_grad=False
                        )
                    elif not is_tensorwise and _HAS_INT8_LAYOUT:
                        self.layout_type = "BlockWiseINT8Layout"
                        layout_params = BlockWiseINT8Layout.Params(
                            scale=scale.to(torch.float32) if scale is not None else None,
                            orig_dtype=torch.bfloat16,
                            orig_shape=tuple(weight_tensor.shape),
                            block_size=self.block_size,
                            is_weight=True,
                        )
                        self.weight = torch.nn.Parameter(
                            QuantizedTensor(weight_tensor, self.layout_type, layout_params),
                            requires_grad=False
                        )
                    else:
                        self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)

                # --- FP8 / MXFP8 ---
                elif weight_tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    self.is_quantized = True
                    self.scale_weight = scale

                    if self.quant_format is not None:
                        qconfig = QUANT_ALGOS.get(self.quant_format, {})
                        self.layout_type = qconfig.get("comfy_tensor_layout", "TensorCoreFP8Layout")
                        if self.block_size is None:
                            self.block_size = qconfig.get("group_size", None)
                    else:
                        if scale is not None:
                            if scale.ndim == 0 or (scale.ndim == 1 and scale.numel() == 1):
                                self.layout_type = "TensorCoreFP8Layout"
                            elif scale.ndim == 1 and scale.numel() == weight_tensor.shape[0]:
                                self.layout_type = "RowWiseFP8Layout"
                            elif scale.ndim == 2:
                                self.layout_type = "BlockWiseFP8Layout"
                                if self.block_size is None:
                                    M, N = weight_tensor.shape
                                    scale_M, scale_N = scale.shape
                                    if M % scale_M == 0 and N % scale_N == 0:
                                        self.block_size = M // scale_M
                            else:
                                self.layout_type = "TensorCoreFP8Layout"
                        else:
                            self.layout_type = "TensorCoreFP8Layout"

                    try:
                        get_layout_class(self.layout_type)
                    except KeyError:
                        self.layout_type = "TensorCoreFP8Layout"

                    if self.layout_type in ["TensorCoreMXFP8Layout", "HybridMXFP8Layout"]:
                        orig_dtype_str = layer_conf.get("orig_dtype", "torch.bfloat16") if layer_conf else "torch.bfloat16"
                        DTYPE_MAP = {
                            "torch.bfloat16": torch.bfloat16,
                            "torch.float16": torch.float16,
                            "torch.float32": torch.float32,
                        }
                        orig_dtype = DTYPE_MAP.get(orig_dtype_str, torch.bfloat16)
                        orig_shape = tuple(layer_conf.get("orig_shape", list(weight_tensor.shape))) if layer_conf else tuple(weight_tensor.shape)
                        
                        if scale is not None and scale.dtype == torch.uint8:
                            scale = scale.view(torch.float8_e8m0fnu)

                        if self.layout_type == "HybridMXFP8Layout":
                            from comfy_kitchen.tensor import HybridMXFP8Layout
                            layout_params = HybridMXFP8Layout.Params(
                                scale=scale, orig_dtype=orig_dtype, orig_shape=orig_shape, scalar=scalar,
                            )
                        else:
                            from comfy_kitchen.tensor import TensorCoreMXFP8Layout
                            layout_params = TensorCoreMXFP8Layout.Params(
                                scale=scale, orig_dtype=orig_dtype, orig_shape=orig_shape,
                            )
                    elif self.layout_type == "BlockWiseFP8Layout":
                        from .quant_layouts.fp8_variants import BlockWiseFP8Layout
                        block_size = self.block_size if self.block_size is not None else 64
                        layout_params = BlockWiseFP8Layout.Params(
                            scale=scale.to(torch.float32) if scale is not None else None,
                            orig_dtype=torch.bfloat16, orig_shape=tuple(weight_tensor.shape), block_size=block_size,
                        )
                    elif self.layout_type == "RowWiseFP8Layout":
                        from .quant_layouts.fp8_variants import RowWiseFP8Layout
                        layout_params = RowWiseFP8Layout.Params(
                            scale=scale.to(torch.float32) if scale is not None else None,
                            orig_dtype=torch.bfloat16, orig_shape=tuple(weight_tensor.shape),
                        )
                    else:
                        from comfy.quant_ops import TensorCoreFP8Layout
                        layout_params = TensorCoreFP8Layout.Params(
                            scale=scale.to(torch.float32) if scale is not None else None,
                            orig_dtype=torch.bfloat16, orig_shape=tuple(weight_tensor.shape),
                        )

                    self.weight = torch.nn.Parameter(
                        QuantizedTensor(weight_tensor, self.layout_type, layout_params),
                        requires_grad=False,
                    )
                else:
                    self.is_quantized = False
                    self.scale_weight = None
                    self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
            else:
                missing_keys.append(weight_key)

            bias_key = prefix + "bias"
            bias_tensor = state_dict.pop(bias_key, None)
            if bias_tensor is not None:
                self.bias = torch.nn.Parameter(bias_tensor, requires_grad=False)
            else:
                self.bias = None


        def forward_comfy_cast_weights(self, input):
            """Forward pass for QuantizedTensors or raw quantified formats."""
            weight = self.weight
            if isinstance(weight, torch.nn.Parameter):
                weight = weight.data

            input_dtype = input.dtype

            is_quantized_fast_path = isinstance(weight, QuantizedTensor)
            cast_dtype = weight.dtype if is_quantized_fast_path else None
            cast_bias_dtype = input_dtype if is_quantized_fast_path else None
            
            weight, bias, offload_stream = cast_bias_weight(
                self,
                input,
                dtype=cast_dtype,
                bias_dtype=cast_bias_dtype,
                offloadable=True,
            )

            if isinstance(weight, QuantizedTensor):
                if hasattr(weight, "_params"):
                    object.__setattr__(weight._params, "orig_dtype", input_dtype)

                if self.layout_type == "TensorCoreMXFP8Layout":
                    input_shape = input.shape
                    tensor_3d = input.ndim == 3
                    
                    if tensor_3d:
                        input = input.reshape(-1, input_shape[2])
                    
                    if input.ndim == 2:
                        if input.dtype == torch.float32:
                            orig_dtype = getattr(weight._params, "orig_dtype", torch.bfloat16)
                            q_input = input.to(orig_dtype)
                        else:
                            q_input = input
                            
                        q_input = QuantizedTensor.from_float(q_input, "TensorCoreMXFP8Layout")
                        out = torch.nn.functional.linear(q_input, weight, bias)
                        if tensor_3d:
                            out = out.reshape(input_shape[0], input_shape[1], -1)
                        if input.dtype == torch.float32:
                            out = out.to(torch.float32)
                    else:
                        out = torch.nn.functional.linear(input.reshape(input_shape), weight.dequantize(), bias)

                elif self.layout_type == "TensorCoreNVFP4Layout":
                    input_shape = input.shape
                    tensor_3d = input.ndim == 3
                    
                    if tensor_3d:
                        input = input.reshape(-1, input_shape[2])
                    
                    if input.ndim == 2:
                        if input.dtype == torch.float32:
                            orig_dtype = getattr(weight._params, "orig_dtype", torch.bfloat16)
                            q_input = input.to(orig_dtype)
                        else:
                            q_input = input

                        q_input = QuantizedTensor.from_float(q_input, "TensorCoreNVFP4Layout")
                        out = torch.nn.functional.linear(q_input, weight, bias)
                        if tensor_3d:
                            out = out.reshape(input_shape[0], input_shape[1], -1)
                        if input.dtype == torch.float32:
                            out = out.to(torch.float32)
                    else:
                        out = torch.nn.functional.linear(input.reshape(input_shape), weight.dequantize(), bias)

                elif self.layout_type in ["TensorCoreFP8Layout", "TensorCoreFP8E4M3Layout", "TensorCoreFP8E5M2Layout"]:
                    input_shape = input.shape
                    tensor_3d = input.ndim == 3
                    
                    if tensor_3d:
                        input = input.reshape(-1, input_shape[2])
                    
                    if input.ndim == 2:
                        if input.dtype == torch.float32:
                            orig_dtype = getattr(weight._params, "orig_dtype", torch.bfloat16)
                            q_input = input.to(orig_dtype)
                        else:
                            q_input = input

                        q_input = QuantizedTensor.from_float(q_input, self.layout_type, scale=getattr(self, 'input_scale', None))
                        out = torch.nn.functional.linear(q_input, weight, bias)
                        if tensor_3d:
                            out = out.reshape(input_shape[0], input_shape[1], -1)
                        if input.dtype == torch.float32:
                            out = out.to(torch.float32)
                    else:
                        out = torch.nn.functional.linear(input.reshape(input_shape), weight.dequantize(), bias)

                else:
                    # Default trigger for QuantizedTensor dispatch -> layout-specific handler
                    out = torch.nn.functional.linear(input, weight, bias)

            else:
                out = torch.nn.functional.linear(input, weight, bias)

            uncast_bias_weight(self, weight, bias, offload_stream)
            return out


        def forward_fused_lora(self, input):
            """
            Memory-efficient LoRA forward pass for INT8 models.
            Instead of dequantizing the full weight, we run native INT8 matmul
            for the base model and compute LoRA contribution separately.
            """
            weight = self.weight
            if isinstance(weight, torch.nn.Parameter):
                weight = weight.data
            
            input_dtype = input.dtype
            
            if not hasattr(UnifiedQuantOps.Linear, '_fused_lora_log_count'):
                UnifiedQuantOps.Linear._fused_lora_log_count = 0
            if UnifiedQuantOps.Linear._fused_lora_log_count < 3:
                logging.info(f"INT8: Using fused LoRA path - input={input.shape}, weight={weight.shape if hasattr(weight, 'shape') else getattr(weight, '_qdata', weight).shape}")
                UnifiedQuantOps.Linear._fused_lora_log_count += 1
            
            if isinstance(weight, QuantizedTensor):
                if weight.device != input.device:
                    weight = weight.to(device=input.device)
                if hasattr(weight, '_params'):
                    object.__setattr__(weight._params, 'orig_dtype', input_dtype)
                
                base_out = torch.nn.functional.linear(input, weight, None)
            else:
                base_out = F.linear(input.to(weight.dtype), weight, None)
            
            lora_out = None
            for patch_fn in self.weight_function:
                if isinstance(patch_fn, LowVramPatch):
                    patches = patch_fn.patches.get(patch_fn.key, [])
                    for patch_data in patches:
                        strength_patch = patch_data[0]
                        adapter = patch_data[1]
                        strength_model = patch_data[2]
                        
                        if hasattr(adapter, 'weights') and adapter.weights is not None:
                            weights = adapter.weights
                            mat1 = weights[0]
                            mat2 = weights[1]
                            alpha = weights[2] if weights[2] is not None else 1.0
                            rank = mat2.shape[0]
                            scale = strength_patch * strength_model * (alpha / rank)
                            
                            mat1 = mat1.to(device=input.device, dtype=input_dtype)
                            mat2 = mat2.to(device=input.device, dtype=input_dtype)
                            
                            temp = F.linear(input, mat2)
                            lora_contrib = F.linear(temp, mat1) * scale
                            
                            if lora_out is None:
                                lora_out = lora_contrib
                            else:
                                lora_out = lora_out + lora_contrib
                        else:
                            logging.warning(f"INT8 Fused LoRA: Falling back to dequant for non-LoRA adapter")
                            if isinstance(self.weight.data, QuantizedTensor):
                                weight_fp = self.weight.data.dequantize().to(input.device)
                            else:
                                weight_fp = self.weight.data.to(device=input.device, dtype=input_dtype)
                            patched_weight = patch_fn(weight_fp)
                            lora_contrib = F.linear(input, patched_weight - weight_fp, None)
                            if lora_out is None:
                                lora_out = lora_contrib
                            else:
                                lora_out = lora_out + lora_contrib
                else:
                    logging.warning(f"INT8 Fused LoRA: Unknown patch function type, falling back")
                    if isinstance(self.weight.data, QuantizedTensor):
                        weight_fp = self.weight.data.dequantize().to(input.device)
                    else:
                        weight_fp = self.weight.data.to(device=input.device, dtype=input_dtype)
                    patched_weight = patch_fn(weight_fp)
                    lora_contrib = F.linear(input, patched_weight - weight_fp, None)
                    if lora_out is None:
                        lora_out = lora_contrib
                    else:
                        lora_out = lora_out + lora_contrib
            
            out = base_out
            if lora_out is not None:
                out = out + lora_out
            
            if self.bias is not None:
                bias = self.bias.to(device=input.device, dtype=input_dtype)
                out = out + bias
            
            return out


        def forward(self, *args, **kwargs):
            weight = self.weight
            if isinstance(weight, torch.nn.Parameter):
                weight = weight.data

            has_lora = len(self.weight_function) > 0
            
            # Use fused LoRA only if it's an INT8 quantized tensor
            is_int8 = isinstance(weight, QuantizedTensor) and getattr(self, 'layout_type', None) in ["BlockWiseINT8Layout", "TensorWiseINT8Layout"]
            
            if has_lora and is_int8:
                return self.forward_fused_lora(*args, **kwargs)
            elif (
                self.comfy_cast_weights
                or has_lora
                or len(self.bias_function) > 0
                or isinstance(weight, QuantizedTensor)
            ):
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

        def convert_weight(self, weight, inplace=False, **kwargs):
            if isinstance(weight, QuantizedTensor):
                return weight.dequantize()
            return weight

        def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
            if getattr(self, 'layout_type', None) is not None:
                weight = QuantizedTensor.from_float(
                    weight, 
                    self.layout_type, 
                    scale="recalculate", 
                    stochastic_rounding=seed if seed else 0,
                    inplace_ops=True
                )
                if hasattr(self.weight, 'dtype'):
                    weight = weight.to(self.weight.dtype)
            else:
                weight = weight.to(self.weight.dtype)

            if return_weight:
                return weight
            
            assert inplace_update is False
            self.weight = torch.nn.Parameter(weight, requires_grad=False)


    class GroupNorm(manual_cast.GroupNorm): pass
    class LayerNorm(manual_cast.LayerNorm): pass
    class RMSNorm(manual_cast.RMSNorm): pass
    class Conv1d(manual_cast.Conv1d): pass
    class Conv2d(manual_cast.Conv2d): pass
    class Conv3d(manual_cast.Conv3d): pass
    class ConvTranspose1d(manual_cast.ConvTranspose1d): pass
    class ConvTranspose2d(manual_cast.ConvTranspose2d): pass
    class Embedding(manual_cast.Embedding): pass

    @classmethod
    def conv_nd(cls, dims, *args, **kwargs):
        if dims == 2:
            return cls.Conv2d(*args, **kwargs)
        elif dims == 3:
            return cls.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")
