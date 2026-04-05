"""
Hybrid BNB 4-bit Operations for NF4/FP4 quantized models.

This module provides custom ops that handle bitsandbytes-compatible 4-bit quantized
models (NF4/FP4 format) without requiring bitsandbytes as a runtime dependency.

State dict format (per quantized weight):
    {prefix}weight: Packed 4-bit indices, shape [numel/2, 1], dtype uint8
    {prefix}weight.absmax: Per-block scales, shape [num_blocks], dtype float32
    {prefix}weight.quant_map: Codebook, shape [16], dtype float32
    {prefix}weight.quant_state.bitsandbytes__nf4: JSON metadata as uint8

The JSON metadata contains:
    - dtype: Original weight dtype (e.g., "bfloat16")
    - shape: Original weight shape (e.g., [3072, 3072])
    - blocksize: Elements per quantization block (e.g., 64)
    - quant_type: "nf4" or "fp4"
"""

import json
import torch
import torch.nn.functional as F
import logging
from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
from unifiedefficientloader import tensor_to_dict


# NF4 (Normal Float 4-bit) quantization table
# These are 16 values derived from the normal distribution, normalized to [-1, 1].
# Source: QLoRA paper (https://arxiv.org/abs/2305.14314)
NF4_QUANT_MAP = torch.tensor([
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
], dtype=torch.float32)

# FP4 (Floating Point 4-bit) quantization table
# Normalized E2M1 floating point representation.
FP4_QUANT_MAP = torch.tensor([
    0.0,
    0.00520833,
    0.16666667,
    0.25,
    0.33333333,
    0.5,
    0.66666667,
    1.0,
    -0.0,
    -0.00520833,
    -0.16666667,
    -0.25,
    -0.33333333,
    -0.5,
    -0.66666667,
    -1.0,
], dtype=torch.float32)


def get_quant_map(quant_type: str, device: torch.device) -> torch.Tensor:
    """Get the quantization codebook for NF4 or FP4."""
    if quant_type == "nf4":
        return NF4_QUANT_MAP.to(device)
    elif quant_type == "fp4":
        return FP4_QUANT_MAP.to(device)
    else:
        logging.warning(f"Unknown quant_type '{quant_type}', defaulting to NF4")
        return NF4_QUANT_MAP.to(device)


def preprocess_bnb_state_dict(state_dict: dict) -> dict:
    """
    Preprocess a BNB 4-bit quantized state dict for model detection.

    ComfyUI's model detection examines weight tensor shapes to determine model
    architecture. BNB 4-bit packed weights have shape [N*K/2, 1] instead of
    original [N, K], causing detection to fail.

    For Flux2 and similar models, detection code has hardcoded defaults
    (e.g., hidden_size=3072, in_channels=16) that are used when weight keys
    are absent. This function simply OMITS packed weight keys from the
    detection state dict, allowing those defaults to be used.

    Args:
        state_dict: Original state dict with packed BNB 4-bit weights

    Returns:
        New state dict with packed weight keys omitted (auxiliary keys kept)
    """
    new_sd = {}
    packed_weight_keys = set()

    # First pass: identify all BNB 4-bit packed weight keys
    for key in state_dict.keys():
        if '.quant_state.bitsandbytes__nf4' in key or '.quant_state.bitsandbytes__fp4' in key:
            # Extract weight key: "layer.weight.quant_state..." -> "layer.weight"
            weight_key = key.rsplit('.quant_state.', 1)[0]
            packed_weight_keys.add(weight_key)

    # Second pass: copy all keys EXCEPT the packed weight keys themselves
    # (Keep auxiliary keys like .absmax, .quant_map, .quant_state for loading)
    for key, value in state_dict.items():
        if key in packed_weight_keys:
            # Skip packed weight - detection will use defaults
            logging.debug(f"Omitting packed weight {key} for detection (defaults will be used)")
            continue
        # Keep everything else
        new_sd[key] = value

    logging.info(f"BNB preprocess: omitted {len(packed_weight_keys)} packed weight keys for detection")
    return new_sd


def get_original_shape(state_dict: dict, weight_key: str) -> tuple:
    """
    Get original shape of a BNB 4-bit quantized weight from its quant_state.

    BNB stores original shape in the quant_state JSON metadata:
    - key.quant_state.bitsandbytes__nf4 or __fp4 contains {"shape": [N, K], ...}

    Args:
        state_dict: State dict with BNB 4-bit weights
        weight_key: Key of the weight (without .quant_state suffix)

    Returns:
        Tuple of original shape, or None if not found
    """
    # Try NF4 first, then FP4
    for suffix in ['.quant_state.bitsandbytes__nf4', '.quant_state.bitsandbytes__fp4']:
        qs_key = weight_key + suffix
        if qs_key in state_dict:
            try:
                qs = tensor_to_dict(state_dict[qs_key])
                shape = qs.get('shape', None)
                if shape:
                    return tuple(shape)
            except Exception as e:
                logging.warning(f"Failed to parse quant_state for {weight_key}: {e}")
    return None


def dequantize_bnb_4bit(
    packed: torch.Tensor,
    absmax: torch.Tensor,
    quant_map: torch.Tensor,
    blocksize: int,
    original_shape: tuple,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Dequantize BNB 4-bit packed weights to full precision.

    Args:
        packed: Packed 4-bit indices, shape [numel/2, 1], dtype uint8
        absmax: Per-block absolute maximum, shape [num_blocks], dtype float32
        quant_map: 16-element codebook, dtype float32
        blocksize: Elements per quantization block
        original_shape: Target output shape
        target_dtype: Target output dtype

    Returns:
        Dequantized weight tensor with original_shape and target_dtype
    """
    device = packed.device

    # Ensure quant_map is on the right device
    quant_map = quant_map.to(device)
    absmax = absmax.to(device)

    # Flatten packed tensor
    packed_flat = packed.flatten()

    # Unpack nibbles: each byte has 2 indices (low 4 bits, high 4 bits)
    low_indices = (packed_flat & 0x0F).to(torch.long)
    high_indices = (packed_flat >> 4).to(torch.long)

    # Interleave to get original order
    indices = torch.stack([low_indices, high_indices], dim=-1).flatten()

    # Look up values in quant_map
    values = quant_map[indices]

    # Calculate dimensions
    n_blocks = absmax.numel()
    n_elements = values.numel()
    values_per_block = n_elements // n_blocks

    # Reshape to blocks and scale by absmax
    if values_per_block * n_blocks <= n_elements:
        values_blocked = values[:n_blocks * values_per_block].view(n_blocks, values_per_block)
        dequantized = values_blocked * absmax.view(-1, 1).to(values.dtype)
    else:
        # Fallback if dimensions don't match
        dequantized = values * absmax.repeat_interleave(values_per_block)[:n_elements].to(values.dtype)

    # Flatten and truncate to original size
    original_numel = 1
    for s in original_shape:
        original_numel *= s

    dequantized_flat = dequantized.flatten()[:original_numel]

    # Reshape and cast
    return dequantized_flat.view(original_shape).to(target_dtype)


class HybridBNB4bitOps(manual_cast):
    """
    Hybrid BNB 4-bit operations class for NF4/FP4 quantized models.

    Handles:
    - Loading from bitsandbytes-format state dicts
    - Dequantization during forward pass
    - Falls back to standard path for non-quantized layers
    """

    class Linear(manual_cast.Linear):
        def __init__(self, in_features, out_features, *args, **kwargs):
            # Force CPU device to reduce memory during init
            # BNB layers will have weights replaced in _load_from_state_dict
            # Non-BNB layers will keep these CPU weights (moved to GPU in forward)
            kwargs['device'] = 'cpu'
            super().__init__(in_features, out_features, *args, **kwargs)
            # 4-bit quantization state
            self.is_bnb_4bit = False
            self.packed_weight = None
            self.absmax = None
            self.quant_map = None
            self.blocksize = 64
            self.original_shape = None
            self.original_dtype = torch.bfloat16
            self.quant_type = "nf4"

        def reset_parameters(self):
            return None




        def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            """
            Custom state dict loading that handles BNB 4-bit format.

            Expected keys:
                {prefix}weight: Packed uint8 tensor
                {prefix}weight.absmax: Per-block scales
                {prefix}weight.quant_map: 16-element codebook
                {prefix}weight.quant_state.bitsandbytes__nf4 (or __fp4): JSON metadata
            """
            weight_key = prefix + 'weight'

            # Check for BNB 4-bit format by looking for quant_state key
            quant_state_key_nf4 = prefix + 'weight.quant_state.bitsandbytes__nf4'
            quant_state_key_fp4 = prefix + 'weight.quant_state.bitsandbytes__fp4'

            quant_state_tensor = state_dict.pop(quant_state_key_nf4, None)
            if quant_state_tensor is not None:
                self.quant_type = "nf4"
            else:
                quant_state_tensor = state_dict.pop(quant_state_key_fp4, None)
                if quant_state_tensor is not None:
                    self.quant_type = "fp4"

            # If we found a quant_state, this is a BNB 4-bit layer
            if quant_state_tensor is not None:
                self.is_bnb_4bit = True

                # Parse quant_state JSON
                try:
                    quant_state = tensor_to_dict(quant_state_tensor)
                    self.blocksize = quant_state.get('blocksize', 64)
                    self.original_shape = tuple(quant_state.get('shape', []))
                    dtype_str = quant_state.get('dtype', 'bfloat16')
                    self.original_dtype = getattr(torch, dtype_str, torch.bfloat16)
                    logging.debug(f"BNB 4-bit layer {weight_key}: {self.quant_type}, shape={self.original_shape}, blocksize={self.blocksize}")
                except Exception as e:
                    logging.warning(f"Failed to parse quant_state for {weight_key}: {e}")
                    self.blocksize = 64
                    self.original_shape = None

                # Load packed weight
                self.packed_weight = state_dict.pop(weight_key, None)
                if self.packed_weight is not None:
                    self.packed_weight = self.packed_weight.to(torch.uint8)

                # Load absmax
                absmax_key = prefix + 'weight.absmax'
                self.absmax = state_dict.pop(absmax_key, None)
                if self.absmax is not None:
                    self.absmax = self.absmax.to(torch.float32)

                # Load quant_map (or use default)
                quant_map_key = prefix + 'weight.quant_map'
                loaded_quant_map = state_dict.pop(quant_map_key, None)
                if loaded_quant_map is not None:
                    self.quant_map = loaded_quant_map.to(torch.float32)
                else:
                    self.quant_map = get_quant_map(self.quant_type, torch.device('cpu'))

                # Set dummy weight to satisfy module structure
                # Actual dequantization happens in forward
                self.weight = torch.nn.Parameter(
                    torch.empty(1, dtype=torch.float32),
                    requires_grad=False
                )

            else:
                # Not a BNB 4-bit layer, use standard loading
                self.is_bnb_4bit = False
                weight_tensor = state_dict.pop(weight_key, None)
                if weight_tensor is not None:
                    self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
                else:
                    missing_keys.append(weight_key)

            # Handle bias
            bias_key = prefix + 'bias'
            bias_tensor = state_dict.pop(bias_key, None)
            if bias_tensor is not None:
                self.bias = torch.nn.Parameter(bias_tensor, requires_grad=False)
            else:
                self.bias = None

        def _dequantize_weight(self, input_dtype: torch.dtype) -> torch.Tensor:
            """Dequantize 4-bit weight to the specified dtype."""
            if not self.is_bnb_4bit:
                return self.weight.to(input_dtype)

            if self.packed_weight is None or self.absmax is None:
                raise RuntimeError("BNB 4-bit layer missing packed_weight or absmax")

            # Infer original shape if not stored
            if self.original_shape is None or len(self.original_shape) == 0:
                # Try to infer from absmax and blocksize
                n_blocks = self.absmax.numel()
                n_elements = self.packed_weight.numel() * 2  # 2 values per byte
                # Assume 2D weight
                out_features = n_blocks * self.blocksize // (n_elements // n_blocks)
                in_features = n_elements // out_features if out_features > 0 else n_elements
                self.original_shape = (out_features, in_features)
                logging.warning(f"Inferred shape {self.original_shape} for BNB 4-bit layer")

            return dequantize_bnb_4bit(
                self.packed_weight,
                self.absmax,
                self.quant_map,
                self.blocksize,
                self.original_shape,
                input_dtype,
            )

        def forward_comfy_cast_weights(self, input):
            """Forward pass with BNB 4-bit dequantization."""
            if self.is_bnb_4bit:
                # Move quantization data to input device
                device = input.device
                if self.packed_weight.device != device:
                    self.packed_weight = self.packed_weight.to(device)
                if self.absmax.device != device:
                    self.absmax = self.absmax.to(device)
                if self.quant_map.device != device:
                    self.quant_map = self.quant_map.to(device)

                # Dequantize weight
                weight = self._dequantize_weight(input.dtype)

                # Handle bias
                bias = self.bias
                if bias is not None:
                    bias = bias.to(device=device, dtype=input.dtype)

                return F.linear(input, weight, bias)

            # Standard manual_cast path for non-BNB layers
            weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            out = F.linear(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return out

        def forward(self, *args, **kwargs):
            if self.is_bnb_4bit or self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            return super().forward(*args, **kwargs)

        def convert_weight(self, weight, inplace=False, **kwargs):
            """Convert weight for LoRA patching - dequantize BNB 4-bit."""
            if self.is_bnb_4bit:
                return self._dequantize_weight(torch.float32)
            return weight

        def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
            """Set weight after LoRA patching."""
            if return_weight:
                return weight

            if inplace_update and not self.is_bnb_4bit:
                self.weight.data.copy_(weight)
            else:
                self.weight = torch.nn.Parameter(weight, requires_grad=False)

            # After patching, no longer in 4-bit mode
            self.is_bnb_4bit = False
            self.packed_weight = None
            self.absmax = None

    # Normalization layers - use standard manual_cast versions
    class GroupNorm(manual_cast.GroupNorm):
        pass

    class LayerNorm(manual_cast.LayerNorm):
        pass

    class RMSNorm(manual_cast.RMSNorm):
        pass

    # Convolution layers - use standard manual_cast versions
    class Conv1d(manual_cast.Conv1d):
        pass

    class Conv2d(manual_cast.Conv2d):
        pass

    class Conv3d(manual_cast.Conv3d):
        pass

    class ConvTranspose1d(manual_cast.ConvTranspose1d):
        pass

    class ConvTranspose2d(manual_cast.ConvTranspose2d):
        pass

    class Embedding(manual_cast.Embedding):
        pass

    @classmethod
    def conv_nd(cls, dims, *args, **kwargs):
        if dims == 2:
            return cls.Conv2d(*args, **kwargs)
        elif dims == 3:
            return cls.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")
