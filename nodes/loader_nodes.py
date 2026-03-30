"""
Loader nodes for quantized models.

These nodes provide custom model loading with:
- Kernel backend selection (pytorch/triton)
- Legacy format support (scale_weight -> weight_scale conversion)
- INT8, FP8, and BNB 4-bit variants
"""

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

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# Shared constants for quantized loader nodes
QUANT_FORMAT_OPTIONS = [
    "auto", "int8", "int8_tensorwise",
    "float8_e4m3fn", "float8_e4m3fn_blockwise", "float8_e4m3fn_rowwise",
    "mxfp8", "hybrid_mxfp8", "nvfp4",
]
KERNEL_BACKEND_OPTIONS = ["pytorch", "triton"]

CLIP_TYPE_OPTIONS = [
    "stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi",
    "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma",
    "ace", "omnigen2", "qwen_image", "hunyuan_image", "flux",
    "hunyuan_video", "flux2", "ovis",
]
DUAL_CLIP_TYPE_OPTIONS = [
    "sdxl", "sd3", "flux", "hunyuan_video", "hidream",
    "hunyuan_image", "hunyuan_video_15", "kandinsky5",
    "kandinsky5_image", "ltxv", "newbie", "ace",
]

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
        with UnifiedSafetensorsLoader(filepath, low_memory=low_memory) as loader:
            sd = {key: loader.get_tensor(key) for key in loader.keys()}
            metadata = loader.metadata() or {}
        return sd, metadata
    else:
        logging.warning(
            "unifiedefficientloader not installed, falling back to comfy.utils.load_torch_file "
            "(aimdo/dynamic VRAM will be active). Install with: pip install unifiedefficientloader"
        )
        return comfy.utils.load_torch_file(filepath, safe_load=True, return_metadata=True)


def _configure_int8_backend(quant_format: str, kernel_backend: str) -> None:
    """Configure INT8 kernel backend if applicable (only affects INT8 blockwise)."""
    if quant_format == "int8":
        try:
            from ..quant_layouts.int8_layout import BlockWiseINT8Layout
            BlockWiseINT8Layout.set_backend(kernel_backend)
            logging.debug(f"Configured INT8 backend to '{kernel_backend}'")
        except Exception as e:
            if kernel_backend == "triton":
                logging.warning(f"Failed to configure Triton backend: {e}")


def _build_model_options(quant_format: str, model_path: str, base_options: dict = None) -> dict:
    """Build model_options dict with auto-detection and UnifiedQuantOps setup.

    For all quant formats (including 'auto'), sets up UnifiedQuantOps as custom_operations.
    When format is 'auto', also performs header-only format detection for logging.

    Args:
        quant_format: The quantization format string.
        model_path: Full path to the model file (used for auto-detection).
        base_options: Optional dict of pre-existing model options to merge with.

    Returns:
        dict with model_options including custom_operations if applicable.
    """
    model_options = dict(base_options) if base_options else {}

    if quant_format == "auto":
        try:
            from ..utils.safetensors_loader import detect_quant_format
            detected_format = detect_quant_format(model_path)
            logging.info(f"Auto-detected quant format: {detected_format}")
        except Exception as e:
            logging.warning(f"Quant format detection failed: {e}")

    # Use UnifiedQuantOps for all formats (handles mixed quantization)
    try:
        from ..unified_ops import UnifiedQuantOps
        model_options["custom_operations"] = UnifiedQuantOps
        if quant_format != "auto":
            logging.info(f"Using UnifiedQuantOps for {quant_format} models")
    except ImportError as e:
        logging.warning(f"UnifiedQuantOps not available: {e}")

    return model_options


def _detect_bnb_model_type(state_dict_keys):
    """Detect BNB model type from state dict keys.

    Detection logic:
    - Flux2: has double_stream_modulation_img.lin.weight
    - Chroma: has distilled_guidance_layer but NOT nerf and NOT __x0__
    - Chroma Radiance: has distilled_guidance_layer AND nerf but NOT __x0__
    - Chroma Radiance X0: has distilled_guidance_layer AND nerf AND __x0__
    - Flux: default
    """
    def has_key_pattern(pattern):
        return any(pattern in k for k in state_dict_keys)

    if has_key_pattern("double_stream_modulation_img.lin.weight"):
        return "flux2"

    has_distilled = has_key_pattern("distilled_guidance_layer.")
    has_nerf = has_key_pattern("nerf_blocks.")
    has_x0 = has_key_pattern("__x0__")

    if has_distilled:
        if has_nerf:
            return "chroma_radiance_x0" if has_x0 else "chroma_radiance"
        return "chroma"

    return "flux"


def _load_unet(unet_name, quant_format, kernel_backend, disable_dynamic, low_memory):
    """Core UNET/diffusion model loading logic.

    Handles INT8 backend config, format detection, safetensors loading, and model construction.
    Used by both QuantizedUNETLoader and QuantizedUNETLoaderSimple.

    Returns:
        The loaded model object.
    """
    _configure_int8_backend(quant_format, kernel_backend)

    unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
    model_options = _build_model_options(quant_format, unet_path)

    sd, metadata = _load_safetensors(unet_path, low_memory=low_memory)

    model = comfy.sd.load_diffusion_model_state_dict(
        sd, model_options=model_options, metadata=metadata, disable_dynamic=disable_dynamic
    )
    return model


def _load_checkpoint(ckpt_name, quant_format, kernel_backend, disable_dynamic, low_memory):
    """Core checkpoint loading logic.

    Handles INT8 backend config, format detection, safetensors loading, and model construction.
    Used by both QuantizedModelLoader and QuantizedModelLoaderSimple.

    Returns:
        Tuple of (model, clip, vae).
    """
    _configure_int8_backend(quant_format, kernel_backend)

    ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
    model_options = _build_model_options(quant_format, ckpt_path)

    sd, metadata = _load_safetensors(ckpt_path, low_memory=low_memory)

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
        logging.warning(f"State dict load failed, falling back to path-based loading: {e}")
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

    # Set cached patcher init for dynamic reloading
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


def _load_clip(clip_name, clip_type_str, quant_format, kernel_backend, disable_dynamic, low_memory):
    """Core CLIP/text encoder loading logic.

    Used by both QuantizedCLIPLoader and QuantizedCLIPLoaderSimple.

    Args:
        clip_name: Filename of the text encoder.
        clip_type_str: String name of clip type (e.g. 'flux', 'sd3').
        quant_format: Quantization format string.
        kernel_backend: Backend string ('pytorch' or 'triton').
        disable_dynamic: Whether to disable dynamic VRAM.
        low_memory: Whether to use low-memory loading.

    Returns:
        The loaded CLIP object.
    """
    import comfy.model_management

    _configure_int8_backend(quant_format, kernel_backend)

    clip_path = folder_paths.get_full_path("text_encoders", clip_name)

    # Convert type string to CLIPType enum
    clip_type = getattr(comfy.sd.CLIPType, clip_type_str.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)

    # Build model options with initial_device for text encoders
    base_options = {"initial_device": comfy.model_management.text_encoder_offload_device()}
    model_options = _build_model_options(quant_format, clip_path, base_options=base_options)

    sd, metadata = _load_safetensors(clip_path, low_memory=low_memory)

    clip = comfy.sd.load_text_encoder_state_dicts(
        state_dicts=[sd],
        clip_type=clip_type,
        model_options=model_options,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        disable_dynamic=disable_dynamic,
    )
    return clip


def _load_dual_clip(text_encoder1, text_encoder2, clip_type_str, quant_format, kernel_backend, disable_dynamic, low_memory):
    """Core dual CLIP/text encoder loading logic.

    Used by both QuantizedDualCLIPLoader and QuantizedDualCLIPLoaderSimple.

    Args:
        text_encoder1: Filename of first text encoder.
        text_encoder2: Filename of second text encoder.
        clip_type_str: String name of clip type.
        quant_format: Quantization format string.
        kernel_backend: Backend string.
        disable_dynamic: Whether to disable dynamic VRAM.
        low_memory: Whether to use low-memory loading.

    Returns:
        The loaded CLIP object.
    """
    import comfy.model_management

    _configure_int8_backend(quant_format, kernel_backend)

    clip_path1 = folder_paths.get_full_path("text_encoders", text_encoder1)

    # For ltxv, text_encoder2 resolves from checkpoints; otherwise from text_encoders
    if clip_type_str == "ltxv":
        clip_path2 = folder_paths.get_full_path("checkpoints", text_encoder2)
    else:
        clip_path2 = folder_paths.get_full_path("text_encoders", text_encoder2)

    clip_type = getattr(comfy.sd.CLIPType, clip_type_str.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)

    base_options = {"initial_device": comfy.model_management.text_encoder_offload_device()}
    model_options = _build_model_options(quant_format, clip_path1, base_options=base_options)

    sd1, metadata1 = _load_safetensors(clip_path1, low_memory=low_memory)
    sd2, metadata2 = _load_safetensors(clip_path2, low_memory=low_memory)

    clip = comfy.sd.load_text_encoder_state_dicts(
        state_dicts=[sd1, sd2],
        clip_type=clip_type,
        model_options=model_options,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        disable_dynamic=disable_dynamic,
    )
    return clip


class QuantizedModelLoader(io.ComfyNode):
    """Load models with custom quantization layouts and kernel backend selection."""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QuantizedModelLoader",
            display_name="Load Checkpoint (Quantized)",
            category="loaders/quantized",
            description="Load checkpoints with custom quantization support. int8_tensorwise uses torch._int_mm for fast inference.",
            inputs=[
                io.Combo.Input("ckpt_name", options=folder_paths.get_filename_list("checkpoints")),
                io.Combo.Input("quant_format", options=QUANT_FORMAT_OPTIONS, default="auto"),
                io.Combo.Input("kernel_backend", options=KERNEL_BACKEND_OPTIONS, default="pytorch"),
                io.Boolean.Input("disable_dynamic", default=True),
                io.Boolean.Input("low_memory", default=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Clip.Output(display_name="clip"),
                io.Vae.Output(display_name="vae"),
            ],
        )

    @classmethod
    def execute(cls, ckpt_name, quant_format, kernel_backend, disable_dynamic, low_memory) -> io.NodeOutput:
        model, clip, vae = _load_checkpoint(ckpt_name, quant_format, kernel_backend, disable_dynamic, low_memory)
        return io.NodeOutput(model, clip, vae)


class QuantizedUNETLoader(io.ComfyNode):
    """Load UNET/diffusion models with custom quantization layouts."""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QuantizedUNETLoader",
            display_name="Load Diffusion Model (Quantized)",
            category="loaders/quantized",
            description="Load diffusion models with custom quantization support. int8_tensorwise uses torch._int_mm for fast inference.",
            inputs=[
                io.Combo.Input("unet_name", options=folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("quant_format", options=QUANT_FORMAT_OPTIONS, default="auto"),
                io.Combo.Input("kernel_backend", options=KERNEL_BACKEND_OPTIONS, default="pytorch"),
                io.Boolean.Input("disable_dynamic", default=True),
                io.Boolean.Input("low_memory", default=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, unet_name, quant_format, kernel_backend, disable_dynamic, low_memory) -> io.NodeOutput:
        model = _load_unet(unet_name, quant_format, kernel_backend, disable_dynamic, low_memory)
        return io.NodeOutput(model)


class QuantizedCLIPLoader(io.ComfyNode):
    """Load CLIP/text encoders with custom quantization layouts."""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QuantizedCLIPLoader",
            display_name="Load CLIP (Quantized)",
            category="loaders/quantized",
            description="Load quantized text encoders (CLIP, T5, etc.). int8_tensorwise uses torch._int_mm for fast inference.",
            inputs=[
                io.Combo.Input("clip_name", options=folder_paths.get_filename_list("text_encoders")),
                io.Combo.Input("type", options=CLIP_TYPE_OPTIONS),
                io.Combo.Input("quant_format", options=QUANT_FORMAT_OPTIONS, default="auto"),
                io.Combo.Input("kernel_backend", options=KERNEL_BACKEND_OPTIONS, default="pytorch"),
                io.Boolean.Input("disable_dynamic", default=True),
                io.Boolean.Input("low_memory", default=True),
            ],
            outputs=[
                io.Clip.Output(display_name="clip"),
            ],
        )

    @classmethod
    def execute(cls, clip_name, type, quant_format, kernel_backend, disable_dynamic, low_memory) -> io.NodeOutput:
        clip = _load_clip(clip_name, type, quant_format, kernel_backend, disable_dynamic, low_memory)
        return io.NodeOutput(clip)


class QuantizedDualCLIPLoader(io.ComfyNode):
    """Load two text encoders with custom quantization layouts (e.g. CLIP-L + T5)."""
    @classmethod
    def define_schema(cls):
        te_list = folder_paths.get_filename_list("text_encoders")
        te_and_ckpt_list = list(te_list) + list(folder_paths.get_filename_list("checkpoints"))
        return io.Schema(
            node_id="QuantizedDualCLIPLoader",
            display_name="Load DualCLIP (Quantized)",
            category="loaders/quantized",
            description=(
                "Load two quantized text encoders (e.g. CLIP-L + T5). "
                "int8_tensorwise uses torch._int_mm for fast inference.\n\n"
                "[Recipes]\n"
                "sdxl: clip-l, clip-g\n"
                "sd3: clip-l, clip-g / clip-l, t5 / clip-g, t5\n"
                "flux: clip-l, t5\n"
                "hidream: at least one of t5 or llama, recommended t5 and llama\n"
                "hunyuan_image: qwen2.5vl 7b and byt5 small\n"
                "newbie: gemma-3-4b-it, jina clip v2"
            ),
            inputs=[
                io.Combo.Input("text_encoder1", options=te_list),
                io.Combo.Input("text_encoder2", options=te_and_ckpt_list),
                io.Combo.Input("type", options=DUAL_CLIP_TYPE_OPTIONS),
                io.Combo.Input("quant_format", options=QUANT_FORMAT_OPTIONS, default="auto"),
                io.Combo.Input("kernel_backend", options=KERNEL_BACKEND_OPTIONS, default="pytorch"),
                io.Boolean.Input("disable_dynamic", default=True),
                io.Boolean.Input("low_memory", default=True),
            ],
            outputs=[
                io.Clip.Output(display_name="clip"),
            ],
        )

    @classmethod
    def execute(cls, text_encoder1, text_encoder2, type, quant_format, kernel_backend, disable_dynamic, low_memory) -> io.NodeOutput:
        clip = _load_dual_clip(text_encoder1, text_encoder2, type, quant_format, kernel_backend, disable_dynamic, low_memory)
        return io.NodeOutput(clip)


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


class BNB4bitUNETLoader(io.ComfyNode):
    """Load UNET/diffusion models quantized to BNB 4-bit (NF4/FP4) format."""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BNB4bitUNETLoader",
            display_name="Load Diffusion Model (BNB 4-bit)",
            category="loaders/quantized",
            description="Load BNB 4-bit (NF4/FP4) quantized Flux/Chroma/Radiance models. Uses pure PyTorch dequantization.",
            inputs=[
                io.Combo.Input("unet_name", options=folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("model_type_override", options=["auto", "flux2", "flux", "chroma", "chroma_radiance", "chroma_radiance_x0"], default="auto", optional=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, unet_name, model_type_override="auto") -> io.NodeOutput:
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
            model_type = _detect_bnb_model_type(sd.keys())
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
        return io.NodeOutput(patcher)


class QuantizedModelLoaderSimple(io.ComfyNode):
    """Simple loader for quantized models (no format or backend selection)."""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QuantizedModelLoaderSimple",
            display_name="Load Checkpoint (Quantized, Simple)",
            category="loaders/quantized",
            description="Simple loader for custom quantized models. Automatically detects formats.",
            inputs=[
                io.Combo.Input("ckpt_name", options=folder_paths.get_filename_list("checkpoints")),
                io.Boolean.Input("disable_dynamic", default=True),
                io.Boolean.Input("low_memory", default=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Clip.Output(display_name="clip"),
                io.Vae.Output(display_name="vae"),
            ],
        )

    @classmethod
    def execute(cls, ckpt_name, disable_dynamic, low_memory) -> io.NodeOutput:
        model, clip, vae = _load_checkpoint(ckpt_name, "auto", "pytorch", disable_dynamic, low_memory)
        return io.NodeOutput(model, clip, vae)


class QuantizedUNETLoaderSimple(io.ComfyNode):
    """Simple loader for quantized UNET models (no format or backend selection)."""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QuantizedUNETLoaderSimple",
            display_name="Load Diffusion Model (Quantized, Simple)",
            category="loaders/quantized",
            description="Simple loader for custom quantized diffusion models. Automatically detects formats.",
            inputs=[
                io.Combo.Input("unet_name", options=folder_paths.get_filename_list("diffusion_models")),
                io.Boolean.Input("disable_dynamic", default=True),
                io.Boolean.Input("low_memory", default=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, unet_name, disable_dynamic, low_memory) -> io.NodeOutput:
        model = _load_unet(unet_name, "auto", "pytorch", disable_dynamic, low_memory)
        return io.NodeOutput(model)


class QuantizedCLIPLoaderSimple(io.ComfyNode):
    """Simple loader for quantized CLIP models (no format or backend selection)."""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QuantizedCLIPLoaderSimple",
            display_name="Load CLIP (Quantized, Simple)",
            category="loaders/quantized",
            description="Simple loader for custom quantized text encoders. Automatically detects formats.",
            inputs=[
                io.Combo.Input("clip_name", options=folder_paths.get_filename_list("text_encoders")),
                io.Combo.Input("type", options=CLIP_TYPE_OPTIONS),
                io.Boolean.Input("disable_dynamic", default=True),
                io.Boolean.Input("low_memory", default=True),
            ],
            outputs=[
                io.Clip.Output(display_name="clip"),
            ],
        )

    @classmethod
    def execute(cls, clip_name, type, disable_dynamic, low_memory) -> io.NodeOutput:
        clip = _load_clip(clip_name, type, "auto", "pytorch", disable_dynamic, low_memory)
        return io.NodeOutput(clip)


class QuantizedDualCLIPLoaderSimple(io.ComfyNode):
    """Simple loader for dual quantized CLIP models (no format or backend selection)."""
    @classmethod
    def define_schema(cls):
        te_list = folder_paths.get_filename_list("text_encoders")
        te_and_ckpt_list = list(te_list) + list(folder_paths.get_filename_list("checkpoints"))
        return io.Schema(
            node_id="QuantizedDualCLIPLoaderSimple",
            display_name="Load DualCLIP (Quantized, Simple)",
            category="loaders/quantized",
            description="Simple loader for dual custom quantized text encoders. Automatically detects formats.",
            inputs=[
                io.Combo.Input("text_encoder1", options=te_list),
                io.Combo.Input("text_encoder2", options=te_and_ckpt_list),
                io.Combo.Input("type", options=DUAL_CLIP_TYPE_OPTIONS),
                io.Boolean.Input("disable_dynamic", default=True),
                io.Boolean.Input("low_memory", default=True),
            ],
            outputs=[
                io.Clip.Output(display_name="clip"),
            ],
        )

    @classmethod
    def execute(cls, text_encoder1, text_encoder2, type, disable_dynamic, low_memory) -> io.NodeOutput:
        clip = _load_dual_clip(text_encoder1, text_encoder2, type, "auto", "pytorch", disable_dynamic, low_memory)
        return io.NodeOutput(clip)


class EfficientVAELoader(io.ComfyNode):
    """Load VAE models using direct safetensors loading, bypassing aimdo/dynamic VRAM."""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EfficientVAELoader",
            display_name="Load VAE (No Dynamic VRAM)",
            category="loaders/quantized",
            description="Load VAE models with direct safetensors loading (bypasses aimdo/dynamic VRAM).",
            inputs=[
                io.Combo.Input("vae_name", options=folder_paths.get_filename_list("vae")),
                io.Boolean.Input("low_memory", default=True),
            ],
            outputs=[
                io.Vae.Output(display_name="vae"),
            ],
        )

    @classmethod
    def execute(cls, vae_name, low_memory) -> io.NodeOutput:
        """Load a VAE model, bypassing aimdo/dynamic VRAM."""
        vae_path = folder_paths.get_full_path("vae", vae_name)

        # Load safetensors directly, bypassing aimdo/dynamic VRAM
        sd, metadata = _load_safetensors(vae_path, low_memory=low_memory)

        # Construct VAE from state dict (comfy.sd.VAE auto-detects architecture)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()

        return io.NodeOutput(vae)


# V3 ComfyUI extension registration
class QuantOpsExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            QuantizedModelLoader,
            QuantizedUNETLoader,
            QuantizedCLIPLoader,
            QuantizedDualCLIPLoader,
            BNB4bitUNETLoader,
            QuantizedModelLoaderSimple,
            QuantizedUNETLoaderSimple,
            QuantizedCLIPLoaderSimple,
            QuantizedDualCLIPLoaderSimple,
            EfficientVAELoader,
        ]


async def comfy_entrypoint() -> QuantOpsExtension:
    return QuantOpsExtension()

