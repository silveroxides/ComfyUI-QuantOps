import json
import logging
import torch
from typing import Optional, Dict, Any, Tuple

try:
    from unifiedefficientloader import UnifiedSafetensorsLoader, tensor_to_dict
    _UNIFIED_LOADER_AVAILABLE = True
except ImportError:
    _UNIFIED_LOADER_AVAILABLE = False
    tensor_to_dict = None

logger = logging.getLogger(__name__)


def detect_layer_quantization(state_dict, prefix=""):
    """Check if state_dict contains .comfy_quant metadata tensors under the given prefix.

    Mirrors comfy.utils.detect_layer_quantization but lives here so QuantOps
    never needs to import the helper at runtime (the user asked us to avoid
    calling ComfyUI helpers directly).

    Returns ``{"mixed_ops": True}`` when at least one key is found, else ``None``.
    """
    for k in state_dict:
        if k.startswith(prefix) and k.endswith(".comfy_quant"):
            return {"mixed_ops": True}
    return None


def convert_old_quants(state_dict, model_prefix="", metadata=None):
    """Process state_dict + file metadata so every quantised layer gets a
    ``.comfy_quant`` uint8 tensor describing its format.

    This is our own re-implementation of ``comfy.utils.convert_old_quants``.
    ComfyUI skips its version when ``custom_operations`` is set in
    ``model_options`` (which is always the case for QuantOps), so we must
    run it ourselves *before* handing the state_dict to ComfyUI.

    The function handles three scenarios:

    1. ``_quantization_metadata`` present in file metadata → parse JSON,
       inject ``.comfy_quant`` tensors.
    2. Legacy ``scaled_fp8`` sentinel key → rename ``scale_weight`` →
       ``weight_scale``, build per-layer config, inject ``.comfy_quant``.
    3. Neither present → do nothing (model may already contain
       ``.comfy_quant`` keys from ``--comfy_quant`` export, or is
       unquantised).

    Returns
    -------
    state_dict : dict
        Possibly modified in-place.
    metadata : dict
        The (unchanged) metadata.
    quant_metadata : dict | None
        ``{"layers": {prefix: {config}, ...}}`` when quantisation was
        detected, else ``None``.
    """
    if metadata is None:
        metadata = {}

    quant_metadata = None

    if "_quantization_metadata" not in metadata:
        # --- Legacy scaled-FP8 format ---
        scaled_fp8_key = "{}scaled_fp8".format(model_prefix)

        if scaled_fp8_key in state_dict:
            scaled_fp8_weight = state_dict[scaled_fp8_key]
            scaled_fp8_dtype = scaled_fp8_weight.dtype
            if scaled_fp8_dtype == torch.float32:
                scaled_fp8_dtype = torch.float8_e4m3fn

            full_precision_matrix_mult = scaled_fp8_weight.nelement() == 2

            out_sd = {}
            layers = {}
            for k in list(state_dict.keys()):
                if k == scaled_fp8_key:
                    continue
                if not k.startswith(model_prefix):
                    out_sd[k] = state_dict[k]
                    continue

                k_out = k
                w = state_dict.pop(k)
                layer = None

                if k_out.endswith(".scale_weight"):
                    layer = k_out[: -len(".scale_weight")]
                    k_out = "{}.weight_scale".format(layer)

                if layer is not None:
                    layer_conf = {"format": "float8_e4m3fn"}
                    if full_precision_matrix_mult:
                        layer_conf["full_precision_matrix_mult"] = full_precision_matrix_mult
                    layers[layer] = layer_conf

                if k_out.endswith(".scale_input"):
                    layer = k_out[: -len(".scale_input")]
                    k_out = "{}.input_scale".format(layer)
                    if w.item() == 1.0:
                        continue

                out_sd[k_out] = w

            state_dict = out_sd
            quant_metadata = {"layers": layers}
    else:
        quant_metadata = json.loads(metadata["_quantization_metadata"])

    # Inject .comfy_quant tensors so that _load_from_state_dict can read
    # per-layer config regardless of how the model was exported.
    if quant_metadata is not None:
        layers = quant_metadata.get("layers", {})
        for k, v in layers.items():
            comfy_quant_key = "{}.comfy_quant".format(k)
            state_dict[comfy_quant_key] = torch.tensor(
                list(json.dumps(v).encode("utf-8")), dtype=torch.uint8
            )

    return state_dict, metadata, quant_metadata


def _is_scale_tensor(key: str) -> bool:
    """Helper to detect tensors containing scales."""
    return (
        key.endswith(".weight_scale")
        or key.endswith(".weight_scale_2")
        or key.endswith(".scale_weight")
    )


def extract_quantization_metadata(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Extract quantization metadata from a safetensors file WITHOUT loading
    the full state dict.  Used only for lightweight pre-detection (e.g.
    the 'auto' format option).

    Returns a dict with a 'layers' key where each layer maps to its config
    e.g., {"layers": {"prefix": {"format": "float8_e4m3fn"}, ...}}
    The 'format' values are QUANT_ALGOS keys.
    Returns a dict with an 'inferred_format' if no explicit metadata is found
    but scale tensors are present.
    Returns None if no quantization is found.
    """
    if not _UNIFIED_LOADER_AVAILABLE:
        logger.warning("UnifiedSafetensorsLoader not available, cannot extract quantization metadata")
        return None

    try:
        with UnifiedSafetensorsLoader(filepath, low_memory=True) as loader:
            # 1. Check __metadata__ for _quantization_metadata
            file_metadata = loader.metadata() or {}
            quant_meta_str = file_metadata.get("_quantization_metadata")

            if quant_meta_str:
                try:
                    return json.loads(quant_meta_str)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to decode _quantization_metadata in {filepath}: {e}"
                    )

            # 2. Scan for .comfy_quant U8 keys
            all_keys = loader.keys()
            comfy_quant_keys = [k for k in all_keys if k.endswith(".comfy_quant")]

            if comfy_quant_keys:
                layers = {}
                for key in comfy_quant_keys:
                    layer_prefix = key[: -len(".comfy_quant")]
                    try:
                        tensor = loader.get_tensor(key)
                        layer_conf = tensor_to_dict(tensor)
                        layers[layer_prefix] = layer_conf
                    except Exception as e:
                        logger.warning(
                            f"Failed to load or parse comfy_quant tensor {key}: {e}"
                        )

                if layers:
                    return {"layers": layers}

            # 3. Check for scale tensors indicating quantization without metadata
            for key in all_keys:
                if _is_scale_tensor(key):
                    # Found a scale, determine format from the weight dtype
                    if key.endswith(".weight_scale"):
                        weight_key = key[: -len(".weight_scale")] + ".weight"
                    elif key.endswith(".weight_scale_2"):
                        weight_key = key[: -len(".weight_scale_2")] + ".weight"
                    else:  # .scale_weight (old fp8_scaled format)
                        weight_key = key[: -len(".scale_weight")] + ".weight"

                    if weight_key in all_keys:
                        if hasattr(loader, "_header") and weight_key in loader._header:
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
        logger.error(
            f"Error extracting quantization metadata from {filepath}: {e}"
        )
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
