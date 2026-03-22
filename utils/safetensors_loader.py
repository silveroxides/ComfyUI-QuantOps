"""
Safetensors loader utilities for QuantOps.

Provides memory-efficient loading of safetensors files with guaranteed
float32 scale conversion for comfy_kitchen compatibility.

Copied and adapted from ComfyUI getkeys.py.
"""

import mmap
import json
import torch
import struct
import re
from typing import Dict, Any, Optional, Tuple


def tensor_to_dict(tensor_data: torch.Tensor) -> dict:
    """Convert uint8 tensor to dictionary."""
    byte_data = bytes(tensor_data.tolist())
    json_str = byte_data.decode("utf-8")
    return json.loads(json_str)


class MemoryEfficientSafeOpen:
    """Memory-efficient safetensors file reader."""

    def __init__(self, filename: str, device: str = "cpu", mmap_mode: bool = False):
        self.filename = filename
        self.device = device
        self.mmap_mode = mmap_mode
        self.header, self.header_size = self._read_header()
        self.file = open(filename, "rb")
        self.mmap_obj = None

        if self.mmap_mode:
            self.mmap_obj = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mmap_obj:
            self.mmap_obj.close()
        self.file.close()

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def get_tensor(self, key: str) -> torch.Tensor:
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]

        if self.mmap_mode and self.mmap_obj:
            if offset_start != offset_end:
                start = self.header_size + 8 + offset_start
                end = self.header_size + 8 + offset_end
                tensor_bytes = memoryview(self.mmap_obj)[start:end]
            else:
                tensor_bytes = None
        else:
            tensor_bytes = None
            if offset_start != offset_end:
                self.file.seek(self.header_size + 8 + offset_start)
                tensor_bytes = self.file.read(offset_end - offset_start)

        return self._deserialize_tensor(tensor_bytes, metadata)

    def get_tensor_as_dict(self, key: str) -> dict:
        """Get a uint8 tensor and convert it to a dictionary."""
        tensor = self.get_tensor(key)
        metadata = self.header[key]

        if metadata["dtype"] != "U8":
            raise ValueError(f"Tensor '{key}' has dtype {metadata['dtype']}, expected U8 (uint8)")

        return tensor_to_dict(tensor)

    def _read_header(self) -> Tuple[dict, int]:
        with open(self.filename, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def _deserialize_tensor(self, tensor_bytes, metadata) -> torch.Tensor:
        dtype_str = metadata["dtype"]
        shape = metadata["shape"]
        dtype = self._get_torch_dtype(dtype_str)

        if tensor_bytes is None:
            byte_tensor = torch.empty(0, dtype=torch.uint8)
        else:
            byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)

        if dtype_str in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, dtype_str, shape)

        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str: str) -> torch.dtype:
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn

        dtype = dtype_map.get(dtype_str)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        return dtype

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str: str, shape) -> torch.Tensor:
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            raise ValueError(f"Unsupported float8 type: {dtype_str}")


def load_quantized_state_dict(
    filepath: str,
    device: str = "cpu",
    force_scale_float32: bool = True,
) -> Tuple[Dict[str, torch.Tensor], str, Optional[dict]]:
    """
    Load a safetensors file with format detection.

    Combines format detection and loading into a single efficient pass.

    Args:
        filepath: Path to safetensors file
        device: Device to load tensors to
        force_scale_float32: If True, convert all scale tensors to float32

    Returns:
        Tuple of (state_dict, detected_format, metadata)
        - state_dict: Dict of tensors with scales guaranteed float32
        - detected_format: Detected quantization format string
        - metadata: File metadata if present
    """
    state_dict = {}
    metadata = None
    detected_format = "unknown"

    with MemoryEfficientSafeOpen(filepath, device=device) as f:
        # Get metadata if present
        if "__metadata__" in f.header:
            metadata = f.header["__metadata__"]

            # Try to detect format from global metadata
            quant_meta_str = metadata.get("_quantization_metadata")
            if quant_meta_str:
                try:
                    quant_meta = json.loads(quant_meta_str)
                    algo = quant_meta.get("algorithm", "").lower()
                    if algo in ("int8_tensorwise", "int8_tw", "w8a8"):
                        detected_format = "int8_tensorwise"
                    elif algo in ("int8_blockwise", "int8_bw", "int8"):
                        detected_format = "int8_blockwise"
                    elif algo in ("fp8_e4m3", "float8_e4m3fn", "fp8"):
                        detected_format = "float8_e4m3fn"
                    elif algo in ("fp8_blockwise", "float8_e4m3fn_blockwise"):
                        detected_format = "float8_e4m3fn_blockwise"
                    elif algo in ("fp8_rowwise", "float8_e4m3fn_rowwise"):
                        detected_format = "float8_e4m3fn_rowwise"
                    elif algo in ("mxfp8",):
                        detected_format = "mxfp8"
                    elif algo in ("nvfp4",):
                        detected_format = "nvfp4"
                    elif quant_meta:
                        # Fallback for mixed precision where `algorithm` might not be at the root or correctly mapped
                        # Check if formats like nvfp4 or mxfp8 are explicitly named in the individual layer formats
                        layers_dict = quant_meta.get("layers", quant_meta)
                        for layer_key, layer_info in layers_dict.items():
                            if isinstance(layer_info, dict):
                                fmt = layer_info.get("format", None)
                                if fmt == "int8_tensorwise":
                                    detected_format = "int8_tensorwise"
                                elif fmt == "int8_blockwise" or fmt == "int8":
                                    detected_format = "int8_blockwise"
                                elif fmt == "float8_e4m3fn_blockwise":
                                    detected_format = "float8_e4m3fn_blockwise"
                                elif fmt == "float8_e4m3fn_rowwise":
                                    detected_format = "float8_e4m3fn_rowwise"
                                elif fmt == "mxfp8":
                                    detected_format = "mxfp8"
                                elif fmt == "nvfp4":
                                    detected_format = "nvfp4"
                                elif fmt in ["float8_e4m3fn", "float8_e5m2"]:
                                    detected_format = "float8_e4m3fn"
                                if detected_format != "unknown":
                                    break
                except (json.JSONDecodeError, TypeError):
                    pass

        keys = f.keys()

        # If not detected from global metadata, try per-layer comfy_quant
        if detected_format == "unknown":
            comfy_quant_keys = [k for k in keys if k.endswith("comfy_quant")]
            if comfy_quant_keys:
                try:
                    layer_meta = f.get_tensor_as_dict(comfy_quant_keys[0])
                    fmt = layer_meta.get("format", "")
                    if "tensorwise" in fmt or "tw" in fmt:
                        detected_format = "int8_tensorwise"
                    elif "blockwise" in fmt or "bw" in fmt:
                        if "int8" in fmt:
                            detected_format = "int8_blockwise"
                        elif "fp8" in fmt or "float8" in fmt:
                            detected_format = "float8_e4m3fn_blockwise"
                except (ValueError, json.JSONDecodeError):
                    pass

        # Fallback: check weight dtype and scale shape
        if detected_format == "unknown":
            weight_keys = [k for k in keys if k.endswith(".weight")]
            scale_keys = [k for k in keys if "weight_scale" in k or "scale_weight" in k]

            if weight_keys:
                first_weight_meta = f.header.get(weight_keys[0], {})
                dtype = first_weight_meta.get("dtype", "")

                if dtype == "I8":
                    if scale_keys:
                        scale_meta = f.header.get(scale_keys[0], {})
                        scale_shape = scale_meta.get("shape", [])
                        # Scalar or 1-element = tensorwise, 2D = blockwise
                        if not scale_shape or (len(scale_shape) == 1 and scale_shape[0] == 1):
                            detected_format = "int8_tensorwise"
                        else:
                            detected_format = "int8_blockwise"
                    else:
                        detected_format = "int8_tensorwise"
                elif dtype in ("F8_E4M3", "F8_E5M2"):
                    detected_format = "float8_e4m3fn"

        # Load all tensors
        for key in keys:
            tensor = f.get_tensor(key)

            # Check if this is a scale tensor that needs float32 conversion
            if force_scale_float32 and _is_scale_tensor(key):
                if tensor.dtype in (torch.float16, torch.bfloat16):
                    tensor = tensor.to(torch.float32)

            state_dict[key] = tensor

    return state_dict, detected_format, metadata


# Legacy alias for backwards compatibility
def load_fp8_state_dict(
    filepath: str,
    device: str = "cpu",
    force_scale_float32: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Optional[dict]]:
    """Legacy function - use load_quantized_state_dict instead."""
    sd, _, metadata = load_quantized_state_dict(filepath, device, force_scale_float32)
    return sd, metadata


def _is_scale_tensor(key: str) -> bool:
    """Check if a tensor key is a scale parameter."""
    scale_patterns = [
        "weight_scale",
        "scale_weight",
        "input_scale",
        "scale_input",
        "weight_scale_2",  # NVFP4 tensor scale
    ]
    return any(pattern in key for pattern in scale_patterns)


def get_layer_metadata(
    filepath: str,
    layer_prefix: str,
) -> Optional[dict]:
    """
    Get comfy_quant metadata for a specific layer.

    Args:
        filepath: Path to safetensors file
        layer_prefix: Layer prefix (e.g., "model.layers.0.attn.qkv.")

    Returns:
        Dict with layer metadata or None if not found
    """
    comfy_quant_key = f"{layer_prefix}comfy_quant"

    with MemoryEfficientSafeOpen(filepath, device="cpu") as f:
        if comfy_quant_key in f.keys():
            try:
                return f.get_tensor_as_dict(comfy_quant_key)
            except (ValueError, json.JSONDecodeError):
                pass
    return None


def get_quantization_metadata(filepath: str) -> Optional[dict]:
    """
    Read _quantization_metadata from safetensors header without loading tensors.

    Args:
        filepath: Path to safetensors file

    Returns:
        Dict with quantization metadata or None if not found
    """
    try:
        # Only read header, not tensors
        with open(filepath, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode("utf-8")

        header = json.loads(header_json)
        metadata = header.get("__metadata__", {})

        # Look for _quantization_metadata (stored as JSON string)
        quant_meta_str = metadata.get("_quantization_metadata")
        if quant_meta_str:
            return json.loads(quant_meta_str)

        return None
    except Exception:
        return None


def detect_quant_format(filepath: str) -> str:
    """
    Detect quantization format from safetensors file metadata.

    Checks:
    1. Global _quantization_metadata in __metadata__
    2. First comfy_quant tensor if exists
    3. Weight dtypes as fallback

    Args:
        filepath: Path to safetensors file

    Returns:
        Detected format string: "int8_tensorwise", "int8_blockwise",
        "float8_e4m3fn", "float8_e4m3fn_blockwise", etc.
        Returns "unknown" if format cannot be detected.
    """
    # Try global metadata first (fastest, header-only)
    quant_meta = get_quantization_metadata(filepath)
    if quant_meta:
        algo = quant_meta.get("algorithm", "").lower()
        # Map algorithm names to our format strings
        if algo in ("int8_tensorwise", "int8_tw", "w8a8"):
            return "int8_tensorwise"
        elif algo in ("int8_blockwise", "int8_bw", "int8"):
            return "int8_blockwise"
        elif algo in ("fp8_e4m3", "float8_e4m3fn", "fp8"):
            return "float8_e4m3fn"
        elif algo in ("fp8_blockwise", "float8_e4m3fn_blockwise"):
            return "float8_e4m3fn_blockwise"
        elif algo in ("fp8_rowwise", "float8_e4m3fn_rowwise"):
            return "float8_e4m3fn_rowwise"
        elif algo in ("mxfp8",):
            return "mxfp8"
        elif algo in ("nvfp4",):
            return "nvfp4"
        elif quant_meta:
            # Fallback for mixed precision where `algorithm` might not be at the root or correctly mapped
            # Check if formats like nvfp4 or mxfp8 are explicitly named in the individual layer formats
            layers_dict = quant_meta.get("layers", quant_meta)
            for layer_key, layer_info in layers_dict.items():
                if isinstance(layer_info, dict):
                    fmt = layer_info.get("format", None)
                    if fmt == "int8_tensorwise":
                        return "int8_tensorwise"
                    elif fmt == "int8_blockwise" or fmt == "int8":
                        return "int8_blockwise"
                    elif fmt == "float8_e4m3fn_blockwise":
                        return "float8_e4m3fn_blockwise"
                    elif fmt == "float8_e4m3fn_rowwise":
                        return "float8_e4m3fn_rowwise"
                    elif fmt == "mxfp8":
                        return "mxfp8"
                    elif fmt == "nvfp4":
                        return "nvfp4"
                    elif fmt in ["float8_e4m3fn", "float8_e5m2"]:
                        return "float8_e4m3fn"

    # Try reading per-layer comfy_quant metadata
    try:
        with MemoryEfficientSafeOpen(filepath, device="cpu") as f:
            keys = f.keys()

            # Look for comfy_quant tensors
            comfy_quant_keys = [k for k in keys if k.endswith("comfy_quant")]
            if comfy_quant_keys:
                try:
                    layer_meta = f.get_tensor_as_dict(comfy_quant_keys[0])
                    fmt = layer_meta.get("format", "")
                    if "tensorwise" in fmt or "tw" in fmt:
                        return "int8_tensorwise"
                    elif "blockwise" in fmt or "bw" in fmt:
                        if "int8" in fmt:
                            return "int8_blockwise"
                        elif "fp8" in fmt or "float8" in fmt:
                            return "float8_e4m3fn_blockwise"
                except (ValueError, json.JSONDecodeError):
                    pass

            # Fallback: check weight dtype and look for scale patterns
            weight_keys = [k for k in keys if k.endswith(".weight")]
            scale_keys = [k for k in keys if "weight_scale" in k or "scale_weight" in k]

            if weight_keys:
                # Get first weight metadata
                first_weight_meta = f.header.get(weight_keys[0], {})
                dtype = first_weight_meta.get("dtype", "")

                if dtype == "I8":
                    # Check if scale is scalar (tensorwise) or has shape (blockwise)
                    if scale_keys:
                        first_scale_key = scale_keys[0]
                        scale_meta = f.header.get(first_scale_key, {})
                        scale_shape = scale_meta.get("shape", [])

                        # Scalar or 1-element = tensorwise, 2D = blockwise
                        if not scale_shape or (len(scale_shape) == 1 and scale_shape[0] == 1):
                            return "int8_tensorwise"
                        else:
                            return "int8_blockwise"
                    return "int8_tensorwise"  # Default for int8

                elif dtype in ("F8_E4M3", "F8_E5M2"):
                    return "float8_e4m3fn"

    except Exception:
        pass

    return "unknown"

