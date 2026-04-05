import json
import logging
from typing import Optional, Dict, Any
from unifiedefficientloader import UnifiedSafetensorsLoader, tensor_to_dict

logger = logging.getLogger(__name__)

def _is_scale_tensor(key: str) -> bool:
    """Helper to detect tensors containing scales."""
    return key.endswith(".weight_scale") or key.endswith(".weight_scale_2") or key.endswith(".scale_weight")


def extract_quantization_metadata(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Extract quantization metadata from a safetensors file.
    
    Returns a dict with a 'layers' key where each layer maps to its config
    e.g., {"layers": {"prefix": {"format": "float8_e4m3fn"}, ...}}
    The 'format' values are QUANT_ALGOS keys.
    Returns a dict with an 'inferred_format' if no explicit metadata is found
    but scale tensors are present.
    Returns None if no quantization is found.
    """
    try:
        with UnifiedSafetensorsLoader(filepath, low_memory=True) as loader:
            # 1. Check __metadata__ for _quantization_metadata
            metadata = loader.metadata() or {}
            quant_meta_str = metadata.get("_quantization_metadata")
            
            if quant_meta_str:
                try:
                    return json.loads(quant_meta_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to decode _quantization_metadata in {filepath}: {e}")
            
            # 2. Scan for .comfy_quant U8 keys
            all_keys = loader.keys()
            comfy_quant_keys = [k for k in all_keys if k.endswith(".comfy_quant")]
            
            if comfy_quant_keys:
                layers = {}
                for key in comfy_quant_keys:
                    layer_prefix = key[:-len(".comfy_quant")]
                    try:
                        tensor = loader.get_tensor(key)
                        layer_conf = tensor_to_dict(tensor)
                        layers[layer_prefix] = layer_conf
                    except Exception as e:
                        logger.warning(f"Failed to load or parse comfy_quant tensor {key}: {e}")
                        
                if layers:
                    return {"layers": layers}
            
            # 3. Check for scale tensors indicating quantization without metadata
            # We can infer the format from the data type of the weight tensor
            for key in all_keys:
                if _is_scale_tensor(key):
                    # Found a scale, let's find the matching weight to determine the format
                    if key.endswith(".weight_scale"):
                        weight_key = key[:-len(".weight_scale")] + ".weight"
                    elif key.endswith(".weight_scale_2"):
                        weight_key = key[:-len(".weight_scale_2")] + ".weight"
                    else: # .scale_weight (old fp8_scaled format)
                        weight_key = key[:-len(".scale_weight")] + ".weight"
                    
                    if weight_key in all_keys:
                        # Use get_shape or check header directly for dtype
                        # UnifiedSafetensorsLoader._header contains dict with "dtype"
                        if hasattr(loader, '_header') and weight_key in loader._header:
                            dtype_str = loader._header[weight_key].get("dtype", "")
                            if dtype_str == "I8":
                                return {"inferred_format": "int8"}
                            elif dtype_str in ["F8_E4M3", "F8_E5M2"]:
                                return {"inferred_format": "float8_e4m3fn"}
                            elif dtype_str == "U8":
                                return {"inferred_format": "nvfp4"}
                    
                    # Fallback if weight not found or dtype not recognized
                    return {"inferred_format": "int8"}
                    
            return None
            
    except Exception as e:
        logger.error(f"Error extracting quantization metadata from {filepath}: {e}")
        return None


def detect_quant_format(filepath: str) -> str:
    """
    Detect the primary quantization format of a safetensors file.
    
    Returns a string representing the format (e.g., 'float8_e4m3fn', 'int8_tensorwise', 'mixed').
    Returns 'unknown' if no quantization is detected.
    """
    meta = extract_quantization_metadata(filepath)
    if not meta:
        return "unknown"
        
    if "inferred_format" in meta:
        return meta["inferred_format"]
        
    layers = meta.get("layers", {})
    if not layers:
        return "unknown"
        
    formats = set()
    for layer_conf in layers.values():
        fmt = layer_conf.get("format")
        if fmt:
            formats.add(fmt)
            
    if not formats:
        return "unknown"
        
    if len(formats) == 1:
        return formats.pop()
        
    # Return mixed if multiple formats exist, unified_ops handles this.
    return "mixed"
