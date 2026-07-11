"""
ComfyUI-QuantOps: Extended Quantization Layouts for ComfyUI

This custom node extends ComfyUI's quantization system with additional layouts:
- INT8 blockwise (with optional Triton acceleration)
- Row-wise and Block-wise FP8 variants

All layouts are lazy-loaded to avoid import errors when optional dependencies
(like Triton) are not installed.
"""

import logging

import torch
from comfy.quant_ops import QUANT_ALGOS, register_layout_class


_NATIVE_COMFY_FORMATS = frozenset(QUANT_ALGOS)

# =============================================================================
# Module-level state for comfy-kitchen backend integration
# =============================================================================

_CK_AVAILABLE = False
_CK_TRITON_AVAILABLE = False


def is_ck_triton_available() -> bool:
    """Check if comfy-kitchen triton backend is available and enabled."""
    return _CK_TRITON_AVAILABLE


# =============================================================================
# Backend Setup
# =============================================================================


def _setup_comfy_kitchen_backends():
    """
    Configure comfy-kitchen backends for QuantOps.
    """
    global _CK_AVAILABLE, _CK_TRITON_AVAILABLE

    try:
        import comfy_kitchen as ck
        _CK_AVAILABLE = True
    except ImportError:
        logging.debug("ComfyUI-QuantOps: comfy-kitchen not available")
        _CK_AVAILABLE = False
        _CK_TRITON_AVAILABLE = False
        return

    try:
        backends = ck.list_backends()
        triton_info = backends.get("triton", {})

        if triton_info.get("available") and not triton_info.get("disabled"):
            _CK_TRITON_AVAILABLE = True
            logging.info("ComfyUI-QuantOps: comfy-kitchen triton backend available")
        elif triton_info.get("available"):
            logging.info("ComfyUI-QuantOps: comfy-kitchen triton backend disabled")
            _CK_TRITON_AVAILABLE = False
        else:
            unavail_reason = triton_info.get("unavailable_reason", "unknown")
            logging.info(f"ComfyUI-QuantOps: comfy-kitchen triton unavailable: {unavail_reason}")
            _CK_TRITON_AVAILABLE = False

    except Exception as e:
        logging.warning(f"ComfyUI-QuantOps: Failed to inspect ck triton backend: {e}")
        _CK_TRITON_AVAILABLE = False


# =============================================================================
# Layout Registration
# =============================================================================


def _register_layouts():
    """Register our custom layouts into ComfyUI's layout registry and QUANT_ALGOS dict."""
    try:
        registered = []
        if "int8_blockwise" not in QUANT_ALGOS:
            from .quant_layouts.int8_layout import BlockWiseINT8Layout

            register_layout_class("BlockWiseINT8Layout", BlockWiseINT8Layout)
            registered.append("BlockWiseINT8Layout")
        if "float8_e4m3fn_rowwise" not in QUANT_ALGOS:
            from .quant_layouts.fp8_variants import RowWiseFP8Layout

            register_layout_class("RowWiseFP8Layout", RowWiseFP8Layout)
            registered.append("RowWiseFP8Layout")
        if "float8_e4m3fn_blockwise" not in QUANT_ALGOS:
            from .quant_layouts.fp8_variants import BlockWiseFP8Layout

            register_layout_class("BlockWiseFP8Layout", BlockWiseFP8Layout)
            registered.append("BlockWiseFP8Layout")

        # Tensorwise INT8 is native in current ComfyUI. Only register it on
        # older installs that do not already expose the format.
        if "int8_tensorwise" not in QUANT_ALGOS:
            try:
                from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
                register_layout_class("TensorWiseINT8Layout", TensorWiseINT8Layout)
                QUANT_ALGOS["int8_tensorwise"] = {
                    "storage_t": torch.int8,
                    "parameters": {"weight_scale", "input_scale"},
                    "comfy_tensor_layout": "TensorWiseINT8Layout",
                }
                registered.append("TensorWiseINT8Layout")
                logging.info("ComfyUI-QuantOps: Registered TensorWiseINT8Layout")
            except ImportError:
                logging.debug("ComfyUI-QuantOps: TensorWiseINT8Layout not available")

        # Register QUANT_ALGOS
        QUANT_ALGOS.setdefault(
            "int8_blockwise",
            {
                "storage_t": torch.int8,
                "parameters": {"weight_scale", "input_scale"},
                "comfy_tensor_layout": "BlockWiseINT8Layout",
                "group_size": 128,
                "asymmetric_layout": True,
            },
        )
        QUANT_ALGOS.setdefault(
            "float8_e4m3fn_rowwise",
            {
                "storage_t": torch.float8_e4m3fn,
                "parameters": {"weight_scale", "input_scale"},
                "comfy_tensor_layout": "RowWiseFP8Layout",
            },
        )
        QUANT_ALGOS.setdefault(
            "float8_e4m3fn_blockwise",
            {
                "storage_t": torch.float8_e4m3fn,
                "parameters": {"weight_scale", "input_scale"},
                "comfy_tensor_layout": "BlockWiseFP8Layout",
                "group_size": 64,
            },
        )

        # MXFP8 from comfy_kitchen
        try:
            from comfy_kitchen.tensor import TensorCoreMXFP8Layout
            if "mxfp8" not in QUANT_ALGOS:
                register_layout_class("TensorCoreMXFP8Layout", TensorCoreMXFP8Layout)
                registered.append("TensorCoreMXFP8Layout")
                logging.info("ComfyUI-QuantOps: Registered TensorCoreMXFP8Layout")
        except ImportError:
            logging.debug("ComfyUI-QuantOps: TensorCoreMXFP8Layout not available")

        QUANT_ALGOS.setdefault(
            "mxfp8",
            {
                "storage_t": torch.float8_e4m3fn,
                "parameters": {"weight_scale"},
                "comfy_tensor_layout": "TensorCoreMXFP8Layout",
                "group_size": 32,
            },
        )

        # Hybrid MXFP8 from comfy_kitchen
        if "hybrid_mxfp8" not in QUANT_ALGOS:
            try:
                from comfy_kitchen.tensor import HybridMXFP8Layout
                register_layout_class("HybridMXFP8Layout", HybridMXFP8Layout)
                registered.append("HybridMXFP8Layout")
                logging.info("ComfyUI-QuantOps: Registered HybridMXFP8Layout")
            except ImportError:
                logging.debug("ComfyUI-QuantOps: HybridMXFP8Layout not available")

        QUANT_ALGOS.setdefault(
            "hybrid_mxfp8",
            {
                "storage_t": torch.float8_e4m3fn,
                "parameters": {"weight_scale", "weight_scalar"},
                "comfy_tensor_layout": "HybridMXFP8Layout",
                "group_size": 32,
            },
        )

        # NVFP4: Don't register layout (ComfyUI core does this), just add QUANT_ALGOS entry if missing
        QUANT_ALGOS.setdefault(
            "nvfp4",
            {
                "storage_t": torch.uint8,
                "parameters": {"weight_scale", "weight_scale_2"},
                "comfy_tensor_layout": "TensorCoreNVFP4Layout",
                "group_size": 16,
            },
        )

        # Verify registration
        logging.info(f"ComfyUI-QuantOps: Registered layouts: {registered}")

    except Exception as e:
        logging.error(f"ComfyUI-QuantOps: Failed to register layouts: {e}")


# =============================================================================
# Module Initialization
# =============================================================================

# Setup backends first (enables ck triton, registers our kernels)
_setup_comfy_kitchen_backends()
logging.info(
    "ComfyUI-QuantOps diagnostic: native formats before extension: %s",
    ", ".join(sorted(_NATIVE_COMFY_FORMATS)),
)

# Register layouts
_register_layouts()
logging.info(
    "ComfyUI-QuantOps diagnostic: formats added to registry: %s",
    ", ".join(sorted(set(QUANT_ALGOS) - _NATIVE_COMFY_FORMATS)) or "none",
)

# Patch stock ComfyUI loaders so QuantOps-only metadata works from normal loaders.
try:
    from .auto_patch import install_auto_patch

    install_auto_patch(_NATIVE_COMFY_FORMATS)
except Exception as e:
    logging.warning(f"ComfyUI-QuantOps: failed to install stock-loader auto patch: {e}")

# Extend an installed GGUF provider without replacing its loader implementation.
try:
    from .gguf_integration import install_gguf_integration

    install_gguf_integration()
except Exception as e:
    logging.warning(f"ComfyUI-QuantOps: failed to install GGUF integration: {e}")

# Import nodes for ComfyUI discovery
from .nodes.loader_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "is_ck_triton_available",
]
