"""
Automatic stock-loader integration for ComfyUI-QuantOps.

ComfyUI core can detect QuantOps metadata after this custom node registers
extra QUANT_ALGOS entries, but the stock mixed-precision loader only knows how
to instantiate ComfyUI-native formats. This patch makes stock loaders use
QuantOps custom_operations when a model contains QuantOps-only formats.
"""

import dataclasses
import json
import logging
from typing import Iterable, Optional, Tuple

import torch


NATIVE_COMFY_FORMATS = {
    "float8_e4m3fn",
    "float8_e5m2",
    "mxfp8",
    "nvfp4",
}


def _metadata_formats(quant_metadata: Optional[dict]) -> set[str]:
    if not quant_metadata:
        return set()
    layers = quant_metadata.get("layers", {})
    return {conf.get("format") for conf in layers.values() if conf.get("format")}


def _needs_quantops(quant_metadata: Optional[dict]) -> bool:
    formats = _metadata_formats(quant_metadata)
    return bool(formats - NATIVE_COMFY_FORMATS)


def _prepare_quantops_options(
    state_dict: dict,
    metadata: Optional[dict],
    model_options: Optional[dict],
    model_prefix: str = "",
) -> Tuple[dict, dict, dict, Optional[dict], bool]:
    """Inject .comfy_quant tensors and QuantOps custom_operations if needed."""
    from .utils.safetensors_loader import convert_old_quants

    metadata = dict(metadata or {})
    state_dict, metadata, quant_metadata = convert_old_quants(
        state_dict,
        model_prefix=model_prefix,
        metadata=metadata,
    )

    if not _needs_quantops(quant_metadata):
        return state_dict, metadata, dict(model_options or {}), quant_metadata, False

    from .unified_ops import make_quant_ops

    patched_options = dict(model_options or {})
    base_ops = patched_options.get("custom_operations", None)
    patched_options["custom_operations"] = make_quant_ops(base_ops)
    patched_options.setdefault("quantization_metadata", {"mixed_ops": True})

    formats = sorted(_metadata_formats(quant_metadata) - NATIVE_COMFY_FORMATS)
    logging.info(
        "ComfyUI-QuantOps: auto-injected custom operations for stock loader formats: %s",
        ", ".join(formats),
    )
    return state_dict, metadata, patched_options, quant_metadata, True


def _dtype_from_metadata(value, default=torch.bfloat16):
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        return {
            "torch.bfloat16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "torch.float16": torch.float16,
            "float16": torch.float16,
            "torch.float32": torch.float32,
            "float32": torch.float32,
        }.get(value, default)
    return default


def _parse_comfy_quant_tensor(tensor) -> Optional[dict]:
    try:
        text = tensor.detach().cpu().numpy().tobytes().decode("utf-8").strip()
        if text.startswith("{{") and text.endswith("}}"):
            text = text[1:-1]
        return json.loads(text)
    except Exception:
        return None


def _quant_metadata_from_state_dict(state_dict: dict, quant_metadata: Optional[dict]) -> dict:
    if quant_metadata is not None:
        return quant_metadata

    layers = {}
    for key, value in state_dict.items():
        if not key.endswith(".comfy_quant"):
            continue
        layer_name = key[: -len(".comfy_quant")]
        layer_conf = _parse_comfy_quant_tensor(value)
        if layer_conf:
            layers[layer_name] = layer_conf
    return {"layers": layers} if layers else {"layers": {}}


def _get_first_tensor(state_dict: dict, keys: Iterable[str]):
    for key in keys:
        value = state_dict.get(key)
        if value is not None:
            return value
    return None


def _manual_dequantize_int8(weight: torch.Tensor, scale: Optional[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    if scale is None:
        return weight.to(dtype)

    scale = scale.to(torch.float32)
    if scale.ndim == 0 or (scale.ndim == 1 and scale.numel() == 1):
        return (weight.to(torch.float32) * scale.reshape(())).to(dtype)
    if scale.ndim == 1 and scale.numel() == weight.shape[0]:
        return (weight.to(torch.float32) * scale.reshape(-1, 1)).to(dtype)
    if scale.ndim == 2 and scale.shape[0] == weight.shape[0] and scale.shape[1] == 1:
        return (weight.to(torch.float32) * scale.to(torch.float32)).to(dtype)

    raise RuntimeError(f"Unsupported INT8 fallback scale shape: {tuple(scale.shape)}")


def _dequantize_layer(state_dict: dict, layer_name: str, layer_conf: dict) -> bool:
    from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor, get_layout_class

    weight_key = f"{layer_name}.weight"
    weight = state_dict.get(weight_key)
    if weight is None:
        return False

    quant_format = layer_conf.get("format")
    if not quant_format:
        return False

    scale = _get_first_tensor(
        state_dict,
        (f"{layer_name}.weight_scale", f"{layer_name}.scale_weight"),
    )
    scale_2 = state_dict.get(f"{layer_name}.weight_scale_2")
    target_dtype = _dtype_from_metadata(layer_conf.get("orig_dtype"), torch.bfloat16)
    orig_shape = tuple(layer_conf.get("orig_shape", tuple(weight.shape)))

    try:
        qconfig = QUANT_ALGOS[quant_format]
        layout_name = qconfig["comfy_tensor_layout"]
        layout_cls = get_layout_class(layout_name)

        params_kwargs = {
            "scale": scale.to(torch.float32) if scale is not None else torch.tensor(1.0),
            "orig_dtype": target_dtype,
            "orig_shape": orig_shape,
            "block_size": int(layer_conf.get("group_size") or qconfig.get("group_size") or 128),
            "is_weight": True,
        }
        if scale_2 is not None:
            params_kwargs["block_scale"] = scale
            params_kwargs["scale"] = scale_2.to(torch.float32)

        field_names = {field.name for field in dataclasses.fields(layout_cls.Params)}
        params_kwargs = {
            key: value for key, value in params_kwargs.items() if key in field_names
        }
        params = layout_cls.Params(**params_kwargs)
        quantized = QuantizedTensor(
            weight.to(qconfig["storage_t"]),
            layout_name,
            params,
        )
        state_dict[weight_key] = quantized.dequantize().to(target_dtype).cpu()
    except Exception as exc:
        if weight.dtype == torch.int8:
            logging.warning(
                "ComfyUI-QuantOps: generic fallback failed for %s (%s); using INT8 dequant fallback",
                layer_name,
                exc,
            )
            state_dict[weight_key] = _manual_dequantize_int8(weight, scale, target_dtype).cpu()
        else:
            logging.warning(
                "ComfyUI-QuantOps: could not dequantize %s fallback layer %s: %s",
                quant_format,
                layer_name,
                exc,
            )
            return False

    for suffix in (
        ".weight_scale",
        ".scale_weight",
        ".weight_scale_2",
        ".weight_scalar",
        ".input_scale",
        ".scale_input",
        ".comfy_quant",
    ):
        state_dict.pop(f"{layer_name}{suffix}", None)
    return True


def _dequantize_for_native_fallback(
    state_dict: dict,
    metadata: Optional[dict],
    quant_metadata: Optional[dict],
) -> Tuple[dict, dict, int]:
    fallback_sd = dict(state_dict)
    fallback_metadata = dict(metadata or {})
    quant_metadata = _quant_metadata_from_state_dict(fallback_sd, quant_metadata)

    converted = 0
    for layer_name, layer_conf in quant_metadata.get("layers", {}).items():
        if layer_conf.get("format") in NATIVE_COMFY_FORMATS:
            continue
        if _dequantize_layer(fallback_sd, layer_name, layer_conf):
            converted += 1

    fallback_metadata.pop("_quantization_metadata", None)
    return fallback_sd, fallback_metadata, converted


def _is_quantops_load_failure(exc: BaseException) -> bool:
    text = str(exc)
    return (
        "Unsupported quantization format" in text
        or "TensorWiseINT8Layout" in text
        or "BlockWiseINT8Layout" in text
        or "RowWiseFP8Layout" in text
        or "BlockWiseFP8Layout" in text
        or "HybridMXFP8Layout" in text
    )


def install_auto_patch() -> None:
    import comfy.model_detection
    import comfy.sd

    if getattr(comfy.sd, "_quantops_auto_patch_installed", False):
        return

    original_load_diffusion_model_state_dict = comfy.sd.load_diffusion_model_state_dict
    original_load_state_dict_guess_config = comfy.sd.load_state_dict_guess_config

    def load_diffusion_model_state_dict_quantops(
        sd,
        model_options={},
        metadata=None,
        disable_dynamic=False,
    ):
        prepared_sd, prepared_metadata, patched_options, quant_metadata, did_patch = _prepare_quantops_options(
            dict(sd),
            metadata,
            model_options,
        )
        if not did_patch:
            return original_load_diffusion_model_state_dict(
                sd,
                model_options=model_options,
                metadata=metadata,
                disable_dynamic=disable_dynamic,
            )

        try:
            return original_load_diffusion_model_state_dict(
                dict(prepared_sd),
                model_options=patched_options,
                metadata=prepared_metadata,
                disable_dynamic=disable_dynamic,
            )
        except Exception as exc:
            if not _is_quantops_load_failure(exc):
                raise
            logging.warning(
                "ComfyUI-QuantOps: auto custom load failed (%s). Falling back to dequantized BF16 load.",
                exc,
            )
            fallback_sd, fallback_metadata, converted = _dequantize_for_native_fallback(
                prepared_sd,
                prepared_metadata,
                quant_metadata,
            )
            if converted == 0:
                raise
            fallback_options = dict(model_options or {})
            fallback_options.pop("custom_operations", None)
            return original_load_diffusion_model_state_dict(
                fallback_sd,
                model_options=fallback_options,
                metadata=fallback_metadata,
                disable_dynamic=disable_dynamic,
            )

    def load_state_dict_guess_config_quantops(
        sd,
        output_vae=True,
        output_clip=True,
        output_clipvision=False,
        embedding_directory=None,
        output_model=True,
        model_options={},
        te_model_options={},
        metadata=None,
        disable_dynamic=False,
    ):
        diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
        prepared_sd, prepared_metadata, patched_options, quant_metadata, did_patch = _prepare_quantops_options(
            dict(sd),
            metadata,
            model_options,
            model_prefix=diffusion_model_prefix,
        )
        if not did_patch:
            return original_load_state_dict_guess_config(
                sd,
                output_vae=output_vae,
                output_clip=output_clip,
                output_clipvision=output_clipvision,
                embedding_directory=embedding_directory,
                output_model=output_model,
                model_options=model_options,
                te_model_options=te_model_options,
                metadata=metadata,
                disable_dynamic=disable_dynamic,
            )

        try:
            return original_load_state_dict_guess_config(
                dict(prepared_sd),
                output_vae=output_vae,
                output_clip=output_clip,
                output_clipvision=output_clipvision,
                embedding_directory=embedding_directory,
                output_model=output_model,
                model_options=patched_options,
                te_model_options=te_model_options,
                metadata=prepared_metadata,
                disable_dynamic=disable_dynamic,
            )
        except Exception as exc:
            if not _is_quantops_load_failure(exc):
                raise
            logging.warning(
                "ComfyUI-QuantOps: auto checkpoint load failed (%s). Falling back to dequantized BF16 UNet load.",
                exc,
            )
            fallback_sd, fallback_metadata, converted = _dequantize_for_native_fallback(
                prepared_sd,
                prepared_metadata,
                quant_metadata,
            )
            if converted == 0:
                raise
            fallback_options = dict(model_options or {})
            fallback_options.pop("custom_operations", None)
            return original_load_state_dict_guess_config(
                fallback_sd,
                output_vae=output_vae,
                output_clip=output_clip,
                output_clipvision=output_clipvision,
                embedding_directory=embedding_directory,
                output_model=output_model,
                model_options=fallback_options,
                te_model_options=te_model_options,
                metadata=fallback_metadata,
                disable_dynamic=disable_dynamic,
            )

    comfy.sd.load_diffusion_model_state_dict = load_diffusion_model_state_dict_quantops
    comfy.sd.load_state_dict_guess_config = load_state_dict_guess_config_quantops
    comfy.sd._quantops_auto_patch_installed = True
    comfy.sd._quantops_original_load_diffusion_model_state_dict = original_load_diffusion_model_state_dict
    comfy.sd._quantops_original_load_state_dict_guess_config = original_load_state_dict_guess_config

    logging.info("ComfyUI-QuantOps: installed stock-loader auto patch")
