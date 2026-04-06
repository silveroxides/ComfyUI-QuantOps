"""
Loader nodes for quantized models.

These nodes provide custom model loading with:
- Kernel backend selection (pytorch/triton)
- Legacy format support (scale_weight -> weight_scale conversion)
- INT8, FP8, and BNB 4-bit variants
"""

import json
import logging
import torch
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_base
import comfy.model_patcher
import comfy.supported_models_base
import comfy.latent_formats
import comfy.conds

# Try to import UnifiedSafetensorsLoader for aimdo-free loading
try:
    from unifiedefficientloader import UnifiedSafetensorsLoader
    _UNIFIED_LOADER_AVAILABLE = True
except ImportError:
    _UNIFIED_LOADER_AVAILABLE = False


def _load_safetensors(filepath, low_memory=True):
    """Load a safetensors file, bypassing comfy_aimdo/dynamic VRAM when possible.

    Uses UnifiedSafetensorsLoader if available, otherwise falls back to
    comfy.utils.load_torch_file.

    Returns:
        Tuple of (state_dict, metadata)
    """
    if _UNIFIED_LOADER_AVAILABLE:
        if low_memory:
            logging.info(f"Loading {filepath} with UnifiedSafetensorsLoader in low memory mode (fast, efficient, minimal VRAM impact)")
            with UnifiedSafetensorsLoader(filepath, low_memory=low_memory) as loader:
                sd = {key: loader.get_tensor(key) for key in loader.keys()}
                metadata = loader.metadata() or {}
                print(f"Loaded state dict with keys: {list(sd.keys())[:10]}... and metadata keys: {list(metadata.keys())}")
            return sd, metadata
        else:
            logging.info(f"Loading {filepath} with comfy.utils.load_torch_file (aimdo/dynamic VRAM will be active)")
            sd, metadata = comfy.utils.load_torch_file(filepath, safe_load=True, return_metadata=True)
            print(f"Loaded state dict with keys: {list(sd.keys())[:10]}... and metadata keys: {list(metadata.keys())}")
            return sd, metadata
    else:
        logging.warning(
            "unifiedefficientloader not installed, falling back to comfy.utils.load_torch_file "
            "(aimdo/dynamic VRAM will be active). Install with: pip install unifiedefficientloader"
        )
        sd, metadata = comfy.utils.load_torch_file(filepath, safe_load=True, return_metadata=True)
        print(f"Loaded state dict with keys: {list(sd.keys())[:10]}... and metadata keys: {list(metadata.keys())}")
        return sd, metadata


def _prepare_state_dict(sd, metadata, model_prefix=""):
    """Run our own convert_old_quants on the state dict so that every
    quantised layer gets a ``.comfy_quant`` metadata tensor.

    ComfyUI skips its ``convert_old_quants`` when ``custom_operations`` is
    set in model_options (which is always the case for QuantOps loaders).
    We must therefore do it ourselves before handing the state dict over.

    Returns (sd, metadata, quant_metadata).
    """
    from ..utils.safetensors_loader import convert_old_quants
    return convert_old_quants(sd, model_prefix=model_prefix, metadata=metadata)


def _detect_te_quantization(state_dict):
    """Detect text-encoder quantisation from state dict .comfy_quant keys.

    Returns a dict with the model-specific quantization metadata keys that
    ComfyUI's ``te()`` factory functions expect, e.g.::

        {"llama_quantization_metadata": {"mixed_ops": True},
         "dtype_llama": torch.bfloat16}

    or::

        {"t5_quantization_metadata": {"mixed_ops": True},
         "dtype_t5": torch.bfloat16}

    The keys mirror what ``comfy.text_encoders.hunyuan_video.llama_detect``
    and ``comfy.text_encoders.sd3_clip.t5_xxl_detect`` produce.
    """
    from ..utils.safetensors_loader import detect_layer_quantization

    out = {}

    # --- llama / qwen style ---
    llama_norm_keys = ["model.norm.weight", "model.layers.0.input_layernorm.weight"]
    for norm_key in llama_norm_keys:
        if norm_key in state_dict:
            out["dtype_llama"] = state_dict[norm_key].dtype
            break

    llama_weight_names = [
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.linear_attn.in_proj_a.weight",
    ]
    is_llama = any(k in state_dict for k in llama_weight_names)
    if is_llama:
        quant = detect_layer_quantization(state_dict, prefix="")
        if quant is not None:
            out["llama_quantization_metadata"] = quant

    # --- T5-XXL style ---
    t5_key = "encoder.final_layer_norm.weight"
    t5_key_old = "encoder.block.23.layer.1.DenseReluDense.wi_1.weight"
    t5_key_old2 = "encoder.block.23.layer.1.DenseReluDense.wi.weight"
    is_t5 = t5_key in state_dict or t5_key_old in state_dict or t5_key_old2 in state_dict
    if is_t5:
        if t5_key in state_dict:
            out["dtype_t5"] = state_dict[t5_key].dtype
        quant = detect_layer_quantization(state_dict, prefix="")
        if quant is not None:
            out["t5_quantization_metadata"] = quant

    return out


def _configure_int8_backend(kernel_backend):
    """Set up INT8 kernel backend (triton or pytorch)."""
    try:
        import comfy_kitchen as ck
        if kernel_backend == "triton":
            ck.set_backend_priority(["triton", "cuda", "eager"])
        else:
            ck.set_backend_priority(["cuda", "eager", "triton"])
        logging.debug(f"Configured backend priority for '{kernel_backend}'")
    except ImportError:
        try:
            from ..quant_layouts.int8_layout import BlockWiseINT8Layout
            BlockWiseINT8Layout.set_backend(kernel_backend)
            logging.debug(f"Configured INT8 backend to '{kernel_backend}'")
        except Exception as e:
            if kernel_backend == "triton":
                logging.warning(f"Failed to configure Triton backend: {e}")
    except Exception as e:
        logging.warning(f"Failed to configure comfy_kitchen backend: {e}")


def _build_model_options(quant_format, sd, metadata, kernel_backend="pytorch",
                         base_options=None, te_quant_info=None):
    """Build model_options for ComfyUI model loading.

    Parameters
    ----------
    quant_format : str
        User-selected format ("auto", "int8", etc.).
    sd : dict
        Already-loaded state dict (*after* ``_prepare_state_dict``).
    metadata : dict
        File metadata from safetensors.
    kernel_backend : str
        "pytorch" or "triton".
    base_options : dict | None
        Extra options to include (e.g. initial_device).
    te_quant_info : dict | None
        Output from ``_detect_te_quantization``.  When present the relevant
        ``*_quantization_metadata`` keys are forwarded into model_options so
        that ComfyUI's ``te()`` factories can pick them up.
    """
    model_options = dict(base_options) if base_options else {}

    # Detect formats from already-processed state dict
    if quant_format == "auto":
        from ..utils.safetensors_loader import detect_layer_quantization
        quant = detect_layer_quantization(sd, prefix="")
        has_int8 = any(
            k.endswith(".weight") and sd[k].dtype == torch.int8
            for k in sd if k.endswith(".weight")
        )
        if has_int8:
            _configure_int8_backend(kernel_backend)
    elif quant_format in ("int8", "int8_tensorwise"):
        _configure_int8_backend(kernel_backend)

    # Forward text-encoder quantization metadata into model_options
    if te_quant_info:
        for key in ("llama_quantization_metadata", "t5_quantization_metadata",
                     "t5xxl_quantization_metadata", "quantization_metadata"):
            if key in te_quant_info:
                model_options[key] = te_quant_info[key]

    # Attach UnifiedQuantOps for all quantized formats
    try:
        from ..unified_ops import UnifiedQuantOps
        model_options["custom_operations"] = UnifiedQuantOps
    except ImportError as e:
        logging.warning(f"UnifiedQuantOps not available: {e}")

    return model_options


class QuantizedModelLoader:
    """
    Load models with custom quantization layouts and kernel backend selection.

    Supports models quantized by convert_to_quant (with or without --comfy_quant flag).
    Automatically handles legacy scale_weight -> weight_scale conversion.

    FP8 Modes:
    - float8_e4m3fn: Standard tensor-scaled FP8 (uses ComfyUI built-in handling)
    - float8_e4m3fn_blockwise: Block-wise scaled FP8 (uses HybridFP8Ops)
    - float8_e4m3fn_rowwise: Row-wise scaled FP8 (uses HybridFP8Ops)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "quant_format": (["auto", "int8", "int8_tensorwise", "float8_e4m3fn", "float8_e4m3fn_blockwise", "float8_e4m3fn_rowwise", "mxfp8", "hybrid_mxfp8", "nvfp4"],),
                "kernel_backend": (["pytorch", "triton"],),
                "disable_dynamic": ("BOOLEAN", {"default": True}),
                "low_memory": ("BOOLEAN", {"default": True, "tooltip": "Use fast and efficient low impact loading of model. Set to False to use comfy's default loading."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders/quantized"
    DESCRIPTION = "Load checkpoints with custom quantization support. int8_tensorwise uses torch._int_mm for fast inference."


    def load_checkpoint(
        self, ckpt_name, quant_format, kernel_backend, disable_dynamic, low_memory
    ):
        """Load a checkpoint with the specified quantization format and kernel backend."""

        # Get full checkpoint path
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        # 1. Load safetensors FIRST so we have sd + metadata for detection
        sd, metadata = _load_safetensors(ckpt_path, low_memory=low_memory)

        # 2. Inject .comfy_quant tensors from _quantization_metadata / legacy formats
        sd, metadata, _qm = _prepare_state_dict(sd, metadata)

        # 3. Build model options using the already-loaded state dict
        model_options = _build_model_options(quant_format, sd, metadata, kernel_backend)

        # Build model from state dict
        try:
            out = comfy.sd.load_state_dict_guess_config(
                sd,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                model_options=model_options,
                disable_dynamic=disable_dynamic,
                metadata=metadata,
            )
        except Exception as e:
            logging.warning(f"QuantizedModelLoader: state_dict load failed, falling back to path-based loading: {e}")
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                model_options=model_options,
                disable_dynamic=disable_dynamic,
            )

        model = out[0]
        clip = out[1]
        vae = out[2]

        # Set cached patcher init for dynamic reloading (mirrors load_checkpoint_guess_config)
        embedding_directory = folder_paths.get_folder_paths("embeddings")
        if model is not None:
            model.cached_patcher_init = (
                comfy.sd.load_checkpoint_guess_config_model_only,
                (ckpt_path, embedding_directory, model_options, {}),
            )
        if clip is not None:
            clip.patcher.cached_patcher_init = (
                comfy.sd.load_checkpoint_guess_config_clip_only,
                (ckpt_path, embedding_directory, model_options, {}),
            )

        return (model, clip, vae)


class QuantizedUNETLoader:
    """
    Load UNET/diffusion models with custom quantization layouts.

    Handles legacy scale_weight format automatically.

    FP8 Modes:
    - float8_e4m3fn: Standard tensor-scaled FP8 (uses ComfyUI built-in handling)
    - float8_e4m3fn_blockwise: Block-wise scaled FP8 (uses HybridFP8Ops)
    - float8_e4m3fn_rowwise: Row-wise scaled FP8 (uses HybridFP8Ops)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "quant_format": (["auto", "int8", "int8_tensorwise", "float8_e4m3fn", "float8_e4m3fn_blockwise", "float8_e4m3fn_rowwise", "mxfp8", "hybrid_mxfp8", "nvfp4"],),
                "kernel_backend": (["pytorch", "triton"],),
                "disable_dynamic": ("BOOLEAN", {"default": True}),
                "low_memory": ("BOOLEAN", {"default": True, "tooltip": "Use fast and efficient low impact loading of model. Set to False to use comfy's default loading."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders/quantized"
    DESCRIPTION = "Load diffusion models with custom quantization support. int8_tensorwise uses torch._int_mm for fast inference."

    def load_unet(self, unet_name, quant_format, kernel_backend, disable_dynamic, low_memory):
        """Load a UNET model with the specified settings."""

        # Get model path
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)

        # 1. Load safetensors FIRST
        sd, metadata = _load_safetensors(unet_path, low_memory=low_memory)

        # 2. Inject .comfy_quant tensors from _quantization_metadata / legacy formats
        sd, metadata, _qm = _prepare_state_dict(sd, metadata)

        # 3. Build model options using the already-loaded state dict
        model_options = _build_model_options(quant_format, sd, metadata, kernel_backend)

        # Build model from state dict
        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata, disable_dynamic=disable_dynamic)

        return (model,)


class QuantizedCLIPLoader:
    """
    Load CLIP/text encoders with custom quantization layouts.

    Supports text encoders quantized by convert_to_quant with various formats.

    FP8 Modes:
    - float8_e4m3fn: Standard tensor-scaled FP8 (uses ComfyUI built-in handling)
    - float8_e4m3fn_blockwise: Block-wise scaled FP8 (uses HybridFP8Ops)
    - float8_e4m3fn_rowwise: Row-wise scaled FP8 (uses HybridFP8Ops)
    """

    # CLIPType options matching built-in CLIPLoader from nodes.py
    CLIP_TYPES = [
        "stable_diffusion",
        "stable_cascade",
        "sd3",
        "stable_audio",
        "mochi",
        "ltxv",
        "pixart",
        "cosmos",
        "lumina2",
        "wan",
        "hidream",
        "chroma",
        "ace",
        "omnigen2",
        "qwen_image",
        "hunyuan_image",
        "flux",
        "hunyuan_video",
        "flux2",
        "ovis",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"),),
                "type": (cls.CLIP_TYPES,),
                "quant_format": (["auto", "int8", "int8_tensorwise", "float8_e4m3fn", "float8_e4m3fn_blockwise", "float8_e4m3fn_rowwise", "mxfp8", "hybrid_mxfp8", "nvfp4"],),
                "kernel_backend": (["pytorch", "triton"],),
                "disable_dynamic": ("BOOLEAN", {"default": True}),
                "low_memory": ("BOOLEAN", {"default": True, "tooltip": "Use fast and efficient low impact loading of model. Set to False to use comfy's default loading."}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "loaders/quantized"
    DESCRIPTION = "Load quantized text encoders (CLIP, T5, etc.). int8_tensorwise uses torch._int_mm for fast inference."

    def load_clip(self, clip_name, type, quant_format, kernel_backend, disable_dynamic, low_memory):
        """Load a CLIP/text encoder with quantization support."""
        import comfy.model_management

        # Get clip path
        clip_path = folder_paths.get_full_path("text_encoders", clip_name)

        # Convert type string to CLIPType enum
        clip_type = getattr(
            comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION
        )

        base_options = {
            "initial_device": comfy.model_management.text_encoder_offload_device()
        }

        # 1. Load safetensors FIRST
        sd, metadata = _load_safetensors(clip_path, low_memory=low_memory)

        # 2. Inject .comfy_quant tensors from _quantization_metadata / legacy formats
        sd, metadata, _qm = _prepare_state_dict(sd, metadata)

        # 3. Detect text-encoder-specific quantization metadata
        te_quant_info = _detect_te_quantization(sd)

        # 4. Build model options with te quant info forwarded
        model_options = _build_model_options(
            quant_format, sd, metadata, kernel_backend,
            base_options=base_options, te_quant_info=te_quant_info,
        )

        # Load text encoder using ComfyUI's API
        clip = comfy.sd.load_text_encoder_state_dicts(
            state_dicts=[sd],
            clip_type=clip_type,
            model_options=model_options,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            disable_dynamic=disable_dynamic,
        )

        return (clip,)


class QuantizedDualCLIPLoader:
    """
    Load two text encoders with custom quantization layouts (e.g. CLIP-L + T5).

    Either or both encoders may be quantized. The Hybrid ops classes handle
    mixed quantized/unquantized weights transparently.

    text_encoder2 lists files from both text_encoders and checkpoints folders.
    When type is 'ltxv', text_encoder2 resolves from checkpoints; otherwise from text_encoders.
    """

    CLIP_TYPES = [
        "sdxl",
        "sd3",
        "flux",
        "hunyuan_video",
        "hidream",
        "hunyuan_image",
        "hunyuan_video_15",
        "kandinsky5",
        "kandinsky5_image",
        "ltxv",
        "newbie",
        "ace",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        te_list = folder_paths.get_filename_list("text_encoders")
        te_and_ckpt_list = list(te_list) + list(folder_paths.get_filename_list("checkpoints"))
        return {
            "required": {
                "text_encoder1": (te_list,),
                "text_encoder2": (te_and_ckpt_list,),
                "type": (cls.CLIP_TYPES,),
                "quant_format": (["auto", "int8", "int8_tensorwise", "float8_e4m3fn", "float8_e4m3fn_blockwise", "float8_e4m3fn_rowwise", "mxfp8", "hybrid_mxfp8", "nvfp4"],),
                "kernel_backend": (["pytorch", "triton"],),
                "disable_dynamic": ("BOOLEAN", {"default": True}),
                "low_memory": ("BOOLEAN", {"default": True, "tooltip": "Use fast and efficient low impact loading of model. Set to False to use comfy's default loading."}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "loaders/quantized"
    DESCRIPTION = (
        "Load two quantized text encoders (e.g. CLIP-L + T5). "
        "int8_tensorwise uses torch._int_mm for fast inference.\n\n"
        "[Recipes]\n"
        "sdxl: clip-l, clip-g\n"
        "sd3: clip-l, clip-g / clip-l, t5 / clip-g, t5\n"
        "flux: clip-l, t5\n"
        "hidream: at least one of t5 or llama, recommended t5 and llama\n"
        "hunyuan_image: qwen2.5vl 7b and byt5 small\n"
        "newbie: gemma-3-4b-it, jina clip v2"
    )

    def load_clip(self, text_encoder1, text_encoder2, type, quant_format, kernel_backend, disable_dynamic, low_memory):
        """Load two text encoders with quantization support."""
        import comfy.model_management

        # Resolve paths
        clip_path1 = folder_paths.get_full_path("text_encoders", text_encoder1)

        # For ltxv, text_encoder2 resolves from checkpoints; otherwise from text_encoders
        if type == "ltxv":
            clip_path2 = folder_paths.get_full_path("checkpoints", text_encoder2)
        else:
            clip_path2 = folder_paths.get_full_path("text_encoders", text_encoder2)

        # Convert type string to CLIPType enum
        clip_type = getattr(
            comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION
        )

        base_options = {
            "initial_device": comfy.model_management.text_encoder_offload_device()
        }

        # 1. Load both state dicts FIRST
        sd1, metadata1 = _load_safetensors(clip_path1, low_memory=low_memory)
        sd2, metadata2 = _load_safetensors(clip_path2, low_memory=low_memory)

        # 2. Inject .comfy_quant tensors for both
        sd1, metadata1, _qm1 = _prepare_state_dict(sd1, metadata1)
        sd2, metadata2, _qm2 = _prepare_state_dict(sd2, metadata2)

        # 3. Detect text-encoder-specific quantization from both state dicts
        te_quant_info = {}
        for sd_i in (sd1, sd2):
            te_quant_info.update(_detect_te_quantization(sd_i))

        # 4. Build model options with te quant info
        model_options = _build_model_options(
            quant_format, sd1, metadata1, kernel_backend,
            base_options=base_options, te_quant_info=te_quant_info,
        )

        # Load dual text encoders using ComfyUI's API
        clip = comfy.sd.load_text_encoder_state_dicts(
            state_dicts=[sd1, sd2],
            clip_type=clip_type,
            model_options=model_options,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            disable_dynamic=disable_dynamic,
        )

        return (clip,)


class BNB4bitFluxConfig(comfy.supported_models_base.BASE):
    """Minimal model config for BNB 4-bit Flux models."""
    unet_config = {}
    unet_extra_config = {}
    latent_format = comfy.latent_formats.Flux
    memory_usage_factor = 2.8
    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    def __init__(self, is_flux2=False):
        self.unet_config = {}
        self.latent_format = comfy.latent_formats.Flux2() if is_flux2 else comfy.latent_formats.Flux()
        self.unet_config["disable_unet_model_creation"] = True
        if is_flux2:
            self.memory_usage_factor = 2.8 * 4 * 2.36  # Flux2 uses more memory


class BNB4bitFluxModel(comfy.model_base.BaseModel):
    """Base model class for BNB 4-bit Flux loading."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        guidance = kwargs.get("guidance", 3.5)
        if guidance is not None:
            out['guidance'] = comfy.conds.CONDRegular(torch.FloatTensor([guidance]))
        return out


class BNB4bitUNETLoader:
    """
    Load UNET/diffusion models quantized to BNB 4-bit (NF4/FP4) format.

    Supports models quantized by convert_to_quant with --bnb-4bit flag.
    Dequantizes weights during forward pass using pure PyTorch.

    Auto-detects Flux vs Flux2 from state dict keys (not shapes).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
            },
            "optional": {
                "model_type_override": (["auto", "flux2", "flux", "chroma", "chroma_radiance", "chroma_radiance_x0"],),
            },
        }


    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders/quantized"
    DESCRIPTION = "Load BNB 4-bit (NF4/FP4) quantized Flux/Chroma/Radiance models. Uses pure PyTorch dequantization."

    def _detect_model_type(self, state_dict_keys):
        """
        Detect model type from state dict keys (not shapes).

        Detection logic:
        - Flux2: has double_stream_modulation_img.lin.weight
        - Chroma: has distilled_guidance_layer but NOT nerf and NOT __x0__
        - Chroma Radiance: has distilled_guidance_layer AND nerf but NOT __x0__
        - Chroma Radiance X0: has distilled_guidance_layer AND nerf AND __x0__
        - Flux: default (has double_blocks but none of the above)
        """
        # Helper to check key presence (handles BNB suffix keys)
        def has_key_pattern(pattern):
            return any(pattern in k for k in state_dict_keys)

        # Check for Flux2 unique key
        if has_key_pattern("double_stream_modulation_img.lin.weight"):
            return "flux2"

        # Check for Chroma/Radiance variants
        has_distilled = has_key_pattern("distilled_guidance_layer.")
        has_nerf = has_key_pattern("nerf_blocks.")
        has_x0 = has_key_pattern("__x0__")

        if has_distilled:
            if has_nerf:
                if has_x0:
                    return "chroma_radiance_x0"
                else:
                    return "chroma_radiance"
            else:
                return "chroma"

        # Default to Flux1
        return "flux"

    def load_unet(self, unet_name, model_type_override="auto"):
        """Load a BNB 4-bit quantized UNET model."""
        import comfy.model_management as model_management
        import comfy.ldm.flux.model as flux_model

        try:
            from ..bnb4bit_ops import HybridBNB4bitOps
        except ImportError as e:
            logging.error(f"Failed to import HybridBNB4bitOps: {e}")
            raise

        # Get model path and load state dict
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        sd = comfy.utils.load_torch_file(unet_path)

        # Strip prefix if present
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("model.diffusion_model."):
                new_sd[k[22:]] = v
            else:
                new_sd[k] = v
        sd = new_sd

        # Detect or use override
        if model_type_override == "auto":
            model_type = self._detect_model_type(sd.keys())
            logging.info(f"BNB4bitUNETLoader: Auto-detected model type: {model_type}")
        else:
            model_type = model_type_override
            logging.info(f"BNB4bitUNETLoader: Using override model type: {model_type}")

        logging.info(f"BNB4bitUNETLoader: Loading {unet_name} as {model_type}")

        is_flux2 = model_type == "flux2"
        is_chroma = model_type in ("chroma", "chroma_radiance", "chroma_radiance_x0")

        load_device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()
        unet_dtype = torch.bfloat16

        # Import shape extraction helper
        from ..bnb4bit_ops import get_original_shape

        # Extract dimensions from quant_state metadata (BNB stores original shapes)
        img_in_shape = get_original_shape(sd, "img_in.weight")
        txt_in_shape = get_original_shape(sd, "txt_in.weight")
        guidance_in_shape = get_original_shape(sd, "guidance_in.in_layer.weight")

        # Derive model dimensions from shapes
        if img_in_shape:
            hidden_size = img_in_shape[0]
            # in_channels depends on patch_size
        else:
            hidden_size = 6144 if is_flux2 else 3072

        if txt_in_shape:
            context_in_dim = txt_in_shape[1]
        else:
            context_in_dim = 15360 if is_flux2 else 4096

        if guidance_in_shape:
            vec_in_dim = guidance_in_shape[1]
        else:
            vec_in_dim = 256 if is_flux2 else 768

        # Count blocks from state dict keys
        def count_blocks(keys, prefix):
            max_idx = -1
            for k in keys:
                if prefix in k:
                    try:
                        idx = int(k.split(prefix)[1].split('.')[0])
                        max_idx = max(max_idx, idx)
                    except (ValueError, IndexError):
                        pass
            return max_idx + 1 if max_idx >= 0 else 0

        depth = count_blocks(sd.keys(), "double_blocks.")
        depth_single_blocks = count_blocks(sd.keys(), "single_blocks.")

        logging.info(f"BNB4bitUNETLoader: Extracted from quant_state:")
        logging.info(f"  hidden_size={hidden_size}, context_in_dim={context_in_dim}, vec_in_dim={vec_in_dim}")
        logging.info(f"  depth={depth}, depth_single_blocks={depth_single_blocks}")

        # Build FluxParams based on detected model type + extracted dimensions
        if model_type == "flux2":
            patch_size = 1
            in_channels = img_in_shape[1] // (patch_size * patch_size) if img_in_shape else 128
            params = flux_model.FluxParams(
                in_channels=in_channels,
                out_channels=128,
                vec_in_dim=vec_in_dim,
                context_in_dim=context_in_dim,
                hidden_size=hidden_size,
                mlp_ratio=3.0,
                num_heads=48,
                depth=depth if depth > 0 else 8,
                depth_single_blocks=depth_single_blocks if depth_single_blocks > 0 else 48,
                axes_dim=[32, 32, 32, 32],
                theta=2000,
                patch_size=patch_size,
                qkv_bias=False,
                guidance_embed=True,
                txt_ids_dims=[3],
                global_modulation=True,
                mlp_silu_act=True,
                ops_bias=False,
            )
        elif model_type == "chroma":
            patch_size = 2
            in_channels = img_in_shape[1] // (patch_size * patch_size) if img_in_shape else 64
            params = flux_model.FluxParams(
                in_channels=in_channels,
                out_channels=64,
                vec_in_dim=vec_in_dim,
                context_in_dim=context_in_dim,
                hidden_size=hidden_size,
                mlp_ratio=4.0,
                num_heads=24,
                depth=depth if depth > 0 else 19,
                depth_single_blocks=depth_single_blocks if depth_single_blocks > 0 else 38,
                axes_dim=[16, 56, 56],
                theta=10000,
                patch_size=patch_size,
                qkv_bias=True,
                guidance_embed=False,
                txt_ids_dims=[],
            )
        elif model_type in ("chroma_radiance", "chroma_radiance_x0"):
            patch_size = 16
            params = flux_model.FluxParams(
                in_channels=3,
                out_channels=3,
                vec_in_dim=vec_in_dim,
                context_in_dim=context_in_dim,
                hidden_size=hidden_size,
                mlp_ratio=4.0,
                num_heads=24,
                depth=depth if depth > 0 else 19,
                depth_single_blocks=depth_single_blocks if depth_single_blocks > 0 else 38,
                axes_dim=[16, 56, 56],
                theta=10000,
                patch_size=patch_size,
                qkv_bias=True,
                guidance_embed=False,
                txt_ids_dims=[],
            )
        else:  # flux (default)
            patch_size = 2
            in_channels = img_in_shape[1] // (patch_size * patch_size) if img_in_shape else 16
            params = flux_model.FluxParams(
                in_channels=in_channels,
                out_channels=16,
                vec_in_dim=vec_in_dim,
                context_in_dim=context_in_dim,
                hidden_size=hidden_size,
                mlp_ratio=4.0,
                num_heads=24,
                depth=depth if depth > 0 else 19,
                depth_single_blocks=depth_single_blocks if depth_single_blocks > 0 else 38,
                axes_dim=[16, 56, 56],
                theta=10000,
                patch_size=patch_size,
                qkv_bias=True,
                guidance_embed=True,
                txt_ids_dims=[],
            )

        # Create model config and base model
        model_conf = BNB4bitFluxConfig(is_flux2=is_flux2)
        model_conf.set_inference_dtype(unet_dtype, unet_dtype)  # Set compute dtype
        model = BNB4bitFluxModel(
            model_conf,
            model_type=comfy.model_base.ModelType.FLUX,
            device=load_device
        )

        logging.info(f"BNB4bitUNETLoader: Creating Flux model with HybridBNB4bitOps")

        # Create diffusion model with our custom ops
        model.diffusion_model = flux_model.Flux(
            device=offload_device,
            dtype=unet_dtype,
            operations=HybridBNB4bitOps,
            **{k: getattr(params, k) for k in params.__dataclass_fields__}
        )
        model.diffusion_model.eval()
        model.diffusion_model.dtype = unet_dtype

        # Load weights from packed state dict using our custom ops
        m, u = model.diffusion_model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            logging.warning(f"BNB4bitUNETLoader: missing keys: {len(m)}")
            logging.debug(f"Missing: {m[:10]}...")
        if len(u) > 0:
            logging.warning(f"BNB4bitUNETLoader: unexpected keys: {len(u)}")
            logging.debug(f"Unexpected: {u[:10]}...")

        logging.info(f"BNB4bitUNETLoader: Successfully loaded {unet_name}")

        patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
        return (patcher,)


class QuantizedModelLoaderSimple:
    """Simple loader for quantized models (no format or backend selection)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "disable_dynamic": ("BOOLEAN", {"default": True}),
                "low_memory": ("BOOLEAN", {"default": True, "tooltip": "Use fast and efficient low impact loading of model. Set to False to use comfy's default loading."}),
            },
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders/quantized"
    DESCRIPTION = "Simple loader for custom quantized models. Automatically detects formats."

    def load_checkpoint(self, ckpt_name, disable_dynamic, low_memory):
        return QuantizedModelLoader().load_checkpoint(ckpt_name, "auto", "pytorch", disable_dynamic, low_memory)


class QuantizedUNETLoaderSimple:
    """Simple loader for quantized UNET models (no format or backend selection)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "disable_dynamic": ("BOOLEAN", {"default": True}),
                "low_memory": ("BOOLEAN", {"default": True, "tooltip": "Use fast and efficient low impact loading of model. Set to False to use comfy's default loading."}),
            },
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders/quantized"
    DESCRIPTION = "Simple loader for custom quantized diffusion models. Automatically detects formats."

    def load_unet(self, unet_name, disable_dynamic, low_memory):
        return QuantizedUNETLoader().load_unet(unet_name, "auto", "pytorch", disable_dynamic, low_memory)


class QuantizedCLIPLoaderSimple:
    """Simple loader for quantized CLIP models (no format or backend selection)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"),),
                "type": (QuantizedCLIPLoader.CLIP_TYPES,),
                "disable_dynamic": ("BOOLEAN", {"default": True}),
                "low_memory": ("BOOLEAN", {"default": True, "tooltip": "Use fast and efficient low impact loading of model. Set to False to use comfy's default loading."}),
            },
        }
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "loaders/quantized"
    DESCRIPTION = "Simple loader for custom quantized text encoders. Automatically detects formats."

    def load_clip(self, clip_name, type, disable_dynamic, low_memory):
        return QuantizedCLIPLoader().load_clip(clip_name, type, "auto", "pytorch", disable_dynamic, low_memory)


class QuantizedDualCLIPLoaderSimple:
    """Simple loader for dual quantized CLIP models (no format or backend selection)."""
    @classmethod
    def INPUT_TYPES(cls):
        te_list = folder_paths.get_filename_list("text_encoders")
        te_and_ckpt_list = list(te_list) + list(folder_paths.get_filename_list("checkpoints"))
        return {
            "required": {
                "text_encoder1": (te_list,),
                "text_encoder2": (te_and_ckpt_list,),
                "type": (QuantizedDualCLIPLoader.CLIP_TYPES,),
                "disable_dynamic": ("BOOLEAN", {"default": True}),
                "low_memory": ("BOOLEAN", {"default": True, "tooltip": "Use fast and efficient low impact loading of model. Set to False to use comfy's default loading."}),

            },
        }
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "loaders/quantized"
    DESCRIPTION = "Simple loader for dual custom quantized text encoders. Automatically detects formats."

    def load_clip(self, text_encoder1, text_encoder2, type, disable_dynamic, low_memory):
        return QuantizedDualCLIPLoader().load_clip(text_encoder1, text_encoder2, type, "auto", "pytorch", disable_dynamic, low_memory)


class EfficientVAELoader:
    """
    Load VAE models using direct safetensors loading, bypassing aimdo/dynamic VRAM.

    Uses UnifiedSafetensorsLoader when available to avoid ComfyUI's aimdo mmap layer,
    then constructs the VAE via comfy.sd.VAE(sd=..., metadata=...).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "low_memory": ("BOOLEAN", {"default": True, "tooltip": "Use fast and efficient low impact loading of model. Set to False to use comfy's default loading."}),
            },
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "loaders/quantized"
    DESCRIPTION = "Load VAE models with direct safetensors loading (bypasses aimdo/dynamic VRAM)."

    def load_vae(self, vae_name, low_memory):
        """Load a VAE model, bypassing aimdo/dynamic VRAM."""
        vae_path = folder_paths.get_full_path("vae", vae_name)

        # Load safetensors directly, bypassing aimdo/dynamic VRAM
        sd, metadata = _load_safetensors(vae_path, low_memory=low_memory)

        # Construct VAE from state dict (comfy.sd.VAE auto-detects architecture)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()

        return (vae,)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "QuantizedModelLoader": QuantizedModelLoader,
    "QuantizedUNETLoader": QuantizedUNETLoader,
    "QuantizedCLIPLoader": QuantizedCLIPLoader,
    "QuantizedDualCLIPLoader": QuantizedDualCLIPLoader,
    "QuantizedModelLoaderSimple": QuantizedModelLoaderSimple,
    "QuantizedUNETLoaderSimple": QuantizedUNETLoaderSimple,
    "QuantizedCLIPLoaderSimple": QuantizedCLIPLoaderSimple,
    "QuantizedDualCLIPLoaderSimple": QuantizedDualCLIPLoaderSimple,
    "BNB4bitUNETLoader": BNB4bitUNETLoader,
    "EfficientVAELoader": EfficientVAELoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QuantizedModelLoader": "Load Checkpoint (Quantized)",
    "QuantizedUNETLoader": "Load Diffusion Model (Quantized)",
    "QuantizedCLIPLoader": "Load CLIP (Quantized)",
    "QuantizedDualCLIPLoader": "Load DualCLIP (Quantized)",
    "QuantizedModelLoaderSimple": "Load Checkpoint (Quantized, Simple)",
    "QuantizedUNETLoaderSimple": "Load Diffusion Model (Quantized, Simple)",
    "QuantizedCLIPLoaderSimple": "Load CLIP (Quantized, Simple)",
    "QuantizedDualCLIPLoaderSimple": "Load DualCLIP (Quantized, Simple)",
    "BNB4bitUNETLoader": "Load Diffusion Model (BNB 4-bit)",
    "EfficientVAELoader": "Load VAE (No Dynamic VRAM)",
}


