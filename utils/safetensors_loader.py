import ctypes
import json
import logging
import os
import threading
import warnings
import torch
from typing import Optional, Dict, Any, Tuple

try:
    from unifiedefficientloader import UnifiedSafetensorsLoader, tensor_to_dict

    _UNIFIED_LOADER_AVAILABLE = True
except ImportError:
    _UNIFIED_LOADER_AVAILABLE = False
    tensor_to_dict = None

logger = logging.getLogger(__name__)

# Safetensors dtype string -> torch dtype (mirrors comfy/utils.py _TYPES)
_SAFETENSORS_TYPES = {
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
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "C64": torch.complex64,
    "U64": torch.uint64,
    "U32": torch.uint32,
    "U16": torch.uint16,
}


def mmap_load_safetensors(filepath):
    """Load safetensors via mmap, stamping tensors with ComfyUI's dynamic VRAM
    protocol attributes so ModelPatcherDynamic can page weights lazily.

    Hybrid approach:
    - ``UnifiedSafetensorsLoader(low_memory=True)`` parses the safetensors
      header (handles dtype mapping, offset extraction, metadata).
    - ``comfy_aimdo.model_mmap.ModelMMAP`` creates the OS memory mapping,
      identical to what ``comfy/utils.py:load_safetensors()`` uses.
    - Each tensor storage is stamped with the three ComfyUI protocol attrs:
        ``_comfy_tensor_file_slice``   -- for fast file-based GPU staging
        ``_comfy_tensor_mmap_refs``    -- keeps mmap + memoryview alive
        ``_comfy_tensor_mmap_touched`` -- tracks OS page-in for residency
    - Falls back to ``async_load_safetensors()`` (load_all) on any failure.

    Used when ``disable_dynamic=False`` so ComfyUI's dynamic VRAM system
    can page weights in/out on demand instead of holding the full model in RAM.

    Parameters
    ----------
    filepath : str
        Path to a ``.safetensors`` file.

    Returns
    -------
    state_dict : dict
        Tensor name -> mmap-backed ``torch.Tensor`` (CPU, read-only view).
    metadata : dict
        File-level metadata from the safetensors header.
    """
    if not _UNIFIED_LOADER_AVAILABLE:
        raise ImportError(
            "unifiedefficientloader is required for mmap_load_safetensors. "
            "Install with: pip install unifiedefficientloader"
        )

    # --- 1. Parse header via UEL (handles safetensors format details) ---
    try:
        loader = UnifiedSafetensorsLoader(filepath, low_memory=True)
        header = loader._header
        header_size = loader._header_size
        metadata = loader.metadata() or {}
        all_keys = loader.keys()
        loader.close()
    except Exception as e:
        logger.warning(
            f"mmap_load_safetensors: UEL header parse failed ({e}), "
            f"falling back to async_load_safetensors"
        )
        return async_load_safetensors(filepath)

    # --- 2. Import comfy_aimdo mmap and ComfyUI TensorFileSlice ---
    try:
        import comfy_aimdo.model_mmap
        from comfy.memory_management import TensorFileSlice
    except ImportError as e:
        logger.warning(
            f"mmap_load_safetensors: comfy_aimdo or TensorFileSlice unavailable "
            f"({e}), falling back to async_load_safetensors"
        )
        return async_load_safetensors(filepath)

    # --- 3. Create mmap mapping and unbuffered file handle ---
    try:
        model_mmap = comfy_aimdo.model_mmap.ModelMMAP(filepath)
    except Exception as e:
        logger.warning(
            f"mmap_load_safetensors: ModelMMAP creation failed ({e}), "
            f"falling back to async_load_safetensors"
        )
        return async_load_safetensors(filepath)

    # Unbuffered file handle for TensorFileSlice fast-read path
    f = open(filepath, "rb", buffering=0)
    file_size = os.path.getsize(filepath)

    # Full-file memoryview over the mmap'd region (read-only)
    mv = memoryview((ctypes.c_uint8 * file_size).from_address(model_mmap.get()))

    # Safetensors data region starts after 8-byte LE length prefix + header JSON
    data_base_offset = 8 + header_size
    mv_data = mv[data_base_offset:]

    # --- 4. Build state dict with mmap-backed stamped tensors ---
    sd = {}
    file_lock = threading.Lock()

    for name in all_keys:
        info = header[name]
        start, end = info["data_offsets"]

        if start == end:
            # Zero-size tensor: allocate empty (no data in file)
            sd[name] = torch.empty(
                info["shape"], dtype=_SAFETENSORS_TYPES[info["dtype"]]
            )
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="The given buffer is not writable"
                )
                # Zero-copy view into mmap'd memory (read-only by design)
                tensor = torch.frombuffer(
                    mv_data[start:end],
                    dtype=_SAFETENSORS_TYPES[info["dtype"]],
                ).view(info["shape"])

                storage = tensor.untyped_storage()

                # --- ComfyUI dynamic VRAM protocol ---
                # Enables read_tensor_file_slice_into() fast path (disk -> staging buf)
                setattr(
                    storage,
                    "_comfy_tensor_file_slice",
                    TensorFileSlice(
                        f, file_lock, data_base_offset + start, end - start
                    ),
                )
                # Keeps model_mmap and memoryview alive while any tensor is alive
                setattr(storage, "_comfy_tensor_mmap_refs", (model_mmap, mv))
                # Tracks OS page-faults for module_mmap_residency() accounting
                setattr(storage, "_comfy_tensor_mmap_touched", False)

                sd[name] = tensor

    logger.info(
        f"mmap_load_safetensors: {filepath}: {len(sd)} tensors, "
        f"{len(metadata)} metadata keys (ComfyUI dynamic VRAM protocol active)"
    )
    return sd, metadata


def async_load_safetensors(filepath):
    """Load all tensors and metadata from a safetensors file using async parallel I/O.

    Uses UnifiedSafetensorsLoader.load_all() which streams tensors from disk
    via a multi-threaded pool for parallel reads.

    Requires ``unifiedefficientloader`` to be installed.

    Returns
    -------
    state_dict : dict
        All tensors keyed by name, on CPU.
    metadata : dict
        File-level metadata from the safetensors header.

    Raises
    ------
    ImportError
        If ``unifiedefficientloader`` is not installed.
    """
    if not _UNIFIED_LOADER_AVAILABLE:
        raise ImportError(
            "unifiedefficientloader is required for async_load_safetensors. "
            "Install with: pip install unifiedefficientloader"
        )

    with UnifiedSafetensorsLoader(filepath, low_memory=True) as loader:
        sd = loader.load_all()
        metadata = loader.metadata() or {}

    logger.info(
        f"Async-loaded {filepath}: {len(sd)} tensors, " f"{len(metadata)} metadata keys"
    )
    return sd, metadata


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


def _infer_layer_format(weight, scale, scale_2):
    """Infer quantization format from a weight tensor's dtype and accompanying
    scale tensor shapes.

    Parameters
    ----------
    weight : torch.Tensor
        The ``.weight`` tensor for a layer.
    scale : torch.Tensor | None
        The ``.weight_scale`` (or ``.scale_weight``) tensor, if present.
    scale_2 : torch.Tensor | None
        The ``.weight_scale_2`` tensor, if present (used by NVFP4).

    Returns
    -------
    dict | None
        A layer config dict (e.g. ``{"format": "int8_tensorwise"}``) when a
        quantised pattern is recognised, otherwise ``None``.
    """
    # NVFP4: uint8 packed weights with a secondary scale
    if weight.dtype == torch.uint8 and scale_2 is not None:
        return {"format": "nvfp4"}

    # INT8: int8 weight with a scale tensor
    if weight.dtype == torch.int8 and scale is not None:
        if scale.ndim == 0 or (scale.ndim == 1 and scale.numel() == 1):
            return {"format": "int8_tensorwise"}
        # Per-channel (per-row) scale: [N] or [N, 1] where N == weight.shape[0]
        if (
            (scale.ndim == 1 and scale.numel() == weight.shape[0])
            or (scale.ndim == 2 and scale.shape[0] == weight.shape[0] and scale.shape[1] == 1)
        ):
            return {"format": "int8_tensorwise"}
        return {"format": "int8"}

    # FP8: float8 weight with a scale tensor
    if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and scale is not None:
        if scale.ndim == 0 or (scale.ndim == 1 and scale.numel() == 1):
            return {"format": "float8_e4m3fn"}
        if scale.ndim == 1 and scale.numel() == weight.shape[0]:
            return {"format": "float8_e4m3fn_rowwise"}
        if scale.ndim == 2:
            return {"format": "float8_e4m3fn_blockwise"}
        return {"format": "float8_e4m3fn"}

    return None


def convert_old_quants(state_dict, model_prefix="", metadata=None):
    """Process state_dict + file metadata so every quantised layer gets a
    ``.comfy_quant`` uint8 tensor describing its format, and the returned
    ``metadata`` dict contains ``_quantization_metadata`` as a JSON string.

    This is our own re-implementation of ``comfy.utils.convert_old_quants``.
    ComfyUI skips its version when ``custom_operations`` is set in
    ``model_options`` (which is always the case for QuantOps), so we must
    run it ourselves *before* handing the state_dict to ComfyUI.

    The function handles five scenarios in priority order:

    1. ``_quantization_metadata`` present in file metadata → parse JSON,
       inject ``.comfy_quant`` tensors.
    2. Legacy ``scaled_fp8`` sentinel key → rename ``scale_weight`` →
       ``weight_scale``, build per-layer config, inject ``.comfy_quant``.
    3. Existing ``.comfy_quant`` tensors in the state dict (from
       ``--comfy_quant`` export) → parse them to reconstruct the
       ``quant_metadata`` dict.
    4. Quantised weight dtype + scale patterns (int8, fp8, nvfp4) without
       any explicit metadata → infer format, inject ``.comfy_quant``.
    5. None of the above → model is unquantised, do nothing.

    In all cases where quantisation is detected, ``metadata`` is updated
    with ``metadata["_quantization_metadata"] = json.dumps(quant_metadata)``
    so downstream ComfyUI APIs that receive the metadata dict can see it.

    Returns
    -------
    state_dict : dict
        Possibly modified in-place.
    metadata : dict
        Updated with ``_quantization_metadata`` when quantisation detected.
    quant_metadata : dict | None
        ``{"layers": {prefix: {config}, ...}}`` when quantisation was
        detected, else ``None``.
    """
    if metadata is None:
        metadata = {}

    quant_metadata = None

    if "_quantization_metadata" not in metadata:
        # --- Scenario 2: Legacy scaled-FP8 format ---
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
                        layer_conf["full_precision_matrix_mult"] = (
                            full_precision_matrix_mult
                        )
                    layers[layer] = layer_conf

                if k_out.endswith(".scale_input"):
                    layer = k_out[: -len(".scale_input")]
                    k_out = "{}.input_scale".format(layer)
                    if w.item() == 1.0:
                        continue

                out_sd[k_out] = w

            state_dict = out_sd
            quant_metadata = {"layers": layers}

        # --- Scenario 3: Reconstruct from existing .comfy_quant tensors ---
        if quant_metadata is None:
            existing_cq_keys = [k for k in state_dict if k.endswith(".comfy_quant")]
            if existing_cq_keys:
                layers = {}
                for cq_key in existing_cq_keys:
                    layer_name = cq_key[: -len(".comfy_quant")]
                    cq_tensor = state_dict[cq_key]
                    try:
                        cq_str = cq_tensor.numpy().tobytes().decode("utf-8").strip()
                        if cq_str.startswith("{{") and cq_str.endswith("}}"):
                            cq_str = cq_str[1:-1]
                        layer_conf = json.loads(cq_str)
                    except Exception:
                        if tensor_to_dict is not None:
                            try:
                                layer_conf = tensor_to_dict(cq_tensor)
                            except Exception:
                                continue
                        else:
                            continue
                    layers[layer_name] = layer_conf
                if layers:
                    quant_metadata = {"layers": layers}

        # --- Scenario 4: Infer from weight dtype + scale patterns ---
        if quant_metadata is None:
            layers = {}
            seen = set()
            for k in list(state_dict.keys()):
                if not k.endswith(".weight"):
                    continue
                layer_name = k[: -len(".weight")]
                if layer_name in seen:
                    continue
                seen.add(layer_name)

                weight = state_dict[k]
                scale = state_dict.get(layer_name + ".weight_scale")
                if scale is None:
                    scale = state_dict.get(layer_name + ".scale_weight")
                scale_2 = state_dict.get(layer_name + ".weight_scale_2")

                layer_conf = _infer_layer_format(weight, scale, scale_2)
                if layer_conf is not None:
                    layers[layer_name] = layer_conf

            if layers:
                quant_metadata = {"layers": layers}
    else:
        # --- Scenario 1: _quantization_metadata already in file metadata ---
        qm_str = metadata["_quantization_metadata"].strip()
        if qm_str.startswith("{{") and qm_str.endswith("}}"):
            qm_str = qm_str[1:-1]
        quant_metadata = json.loads(qm_str)

    # Inject .comfy_quant tensors so that _load_from_state_dict can read
    # per-layer config regardless of how the model was exported.
    # Skip keys that already exist (Scenario 3 preserves them).
    if quant_metadata is not None:
        layers = quant_metadata.get("layers", {})
        for layer_name, layer_conf in layers.items():
            comfy_quant_key = "{}.comfy_quant".format(layer_name)
            if comfy_quant_key not in state_dict:
                state_dict[comfy_quant_key] = torch.tensor(
                    list(json.dumps(layer_conf).encode("utf-8")),
                    dtype=torch.uint8,
                )

        # Ensure metadata carries _quantization_metadata for downstream
        # ComfyUI APIs (load_state_dict_guess_config, load_diffusion_model_state_dict, etc.)
        if "_quantization_metadata" not in metadata:
            metadata["_quantization_metadata"] = json.dumps(quant_metadata)

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
        logger.warning(
            "UnifiedSafetensorsLoader not available, cannot extract quantization metadata"
        )
        return None

    try:
        with UnifiedSafetensorsLoader(filepath, low_memory=True) as loader:
            # 1. Check __metadata__ for _quantization_metadata
            file_metadata = loader.metadata() or {}
            quant_meta_str = file_metadata.get("_quantization_metadata")

            if quant_meta_str:
                try:
                    quant_meta_str = quant_meta_str.strip()
                    if quant_meta_str.startswith("{{") and quant_meta_str.endswith(
                        "}}"
                    ):
                        quant_meta_str = quant_meta_str[1:-1]
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
