"""
ComfyUI-QuantOps: Extended Quantization Layouts for ComfyUI

This custom node extends ComfyUI's quantization system with additional layouts:
- INT8 blockwise (with optional Triton acceleration)
- INT8 tensorwise (uses torch._int_mm with dynamic activation quant)
- Row-wise and Block-wise FP8 variants

All layouts are lazy-loaded to avoid import errors when optional dependencies
(like Triton) are not installed.
"""

import logging

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
    
    1. Re-enable triton backend (ComfyUI disables it by default)
    2. Register QuantOps kernels as a custom backend
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
    
    # Step 1: Re-enable triton backend (ComfyUI disables it)
    try:
        ck.enable_backend("triton")
        
        backends = ck.list_backends()
        triton_info = backends.get("triton", {})
        
        if triton_info.get("available") and not triton_info.get("disabled"):
            _CK_TRITON_AVAILABLE = True
            logging.info("ComfyUI-QuantOps: Enabled comfy-kitchen triton backend")
        else:
            unavail_reason = triton_info.get("unavailable_reason", "unknown")
            logging.info(f"ComfyUI-QuantOps: comfy-kitchen triton unavailable: {unavail_reason}")
            _CK_TRITON_AVAILABLE = False
            
    except Exception as e:
        logging.warning(f"ComfyUI-QuantOps: Failed to enable ck triton backend: {e}")
        _CK_TRITON_AVAILABLE = False
    
    # Step 2: Register QuantOps kernels as a custom backend
    _register_quantops_backend()


def _register_quantops_backend():
    """
    Register QuantOps Triton kernels with comfy-kitchen registry.
    
    This allows ck dispatch to use our INT8/FP8 kernels.
    """
    try:
        import torch
        from comfy_kitchen.registry import registry
        from comfy_kitchen.constraints import (
            FunctionConstraints,
            ParamConstraint,
            ExactDims,
            DivisibleBy,
        )
        
        # Import our kernel modules
        from .kernels import int8_kernels
        from .kernels import fp8_kernels
        
        cuda_devices = frozenset({"cuda"})
        standard_floats = frozenset({torch.float32, torch.float16, torch.bfloat16})
        
        # Build constraints for INT8 kernels
        int8_constraints = {
            "act_quant": FunctionConstraints(
                params={
                    "x": ParamConstraint(
                        dtypes=standard_floats,
                        shape_rules=(DivisibleBy(-1, 128),),  # Last dim divisible by block_size
                    ),
                },
                default_devices=cuda_devices,
            ),
            "act_dequant": FunctionConstraints(
                params={
                    "x": ParamConstraint(dtypes=frozenset({torch.int8})),
                    "s": ParamConstraint(dtypes=frozenset({torch.float32})),
                },
                default_devices=cuda_devices,
            ),
            "weight_quant": FunctionConstraints(
                params={
                    "x": ParamConstraint(
                        dtypes=standard_floats,
                        shape_rules=(ExactDims(2),),
                    ),
                },
                default_devices=cuda_devices,
            ),
            "weight_dequant": FunctionConstraints(
                params={
                    "x": ParamConstraint(dtypes=frozenset({torch.int8})),
                    "s": ParamConstraint(dtypes=frozenset({torch.float32})),
                },
                default_devices=cuda_devices,
            ),
        }
        
        # Build constraints for FP8 kernels
        fp8_constraints = {
            "fp8_act_quant": FunctionConstraints(
                params={
                    "x": ParamConstraint(dtypes=standard_floats),
                },
                default_devices=cuda_devices,
            ),
            "fp8_gemm_blockwise": FunctionConstraints(
                params={
                    "a": ParamConstraint(dtypes=frozenset({torch.float8_e4m3fn})),
                    "b": ParamConstraint(dtypes=frozenset({torch.float8_e4m3fn})),
                    "a_s": ParamConstraint(dtypes=frozenset({torch.float32})),
                    "b_s": ParamConstraint(dtypes=frozenset({torch.float32})),
                },
                default_devices=cuda_devices,
            ),
            "fp8_gemm_rowwise": FunctionConstraints(
                params={
                    "a": ParamConstraint(dtypes=frozenset({torch.float8_e4m3fn})),
                    "b": ParamConstraint(dtypes=frozenset({torch.float8_e4m3fn})),
                    "a_s": ParamConstraint(dtypes=frozenset({torch.float32})),
                    "b_s": ParamConstraint(dtypes=frozenset({torch.float32})),
                },
                default_devices=cuda_devices,
            ),
        }
        
        # Register INT8 backend
        try:
            registry.register(
                name="quantops_int8",
                module=int8_kernels,
                capabilities=int8_constraints,
            )
            logging.info("ComfyUI-QuantOps: Registered quantops_int8 backend")
        except Exception as e:
            logging.debug(f"ComfyUI-QuantOps: Could not register INT8 backend: {e}")
        
        # Register FP8 backend
        try:
            registry.register(
                name="quantops_fp8",
                module=fp8_kernels,
                capabilities=fp8_constraints,
            )
            logging.info("ComfyUI-QuantOps: Registered quantops_fp8 backend")
        except Exception as e:
            logging.debug(f"ComfyUI-QuantOps: Could not register FP8 backend: {e}")
            
    except ImportError as e:
        logging.debug(f"ComfyUI-QuantOps: Could not register backends (missing deps): {e}")
    except Exception as e:
        logging.warning(f"ComfyUI-QuantOps: Backend registration failed: {e}")


# =============================================================================
# Layout Registration
# =============================================================================


def _register_layouts():
    """Register our custom layouts into ComfyUI's layout registry and QUANT_ALGOS dict."""
    try:
        from comfy.quant_ops import QUANT_ALGOS, register_layout_class
        import torch

        # Import our layouts (this also registers their operation handlers)
        from .quant_layouts.int8_layout import BlockWiseINT8Layout
        from .quant_layouts.fp8_variants import RowWiseFP8Layout, BlockWiseFP8Layout

        # Register layouts using the new comfy_kitchen API
        register_layout_class("BlockWiseINT8Layout", BlockWiseINT8Layout)
        register_layout_class("RowWiseFP8Layout", RowWiseFP8Layout)
        register_layout_class("BlockWiseFP8Layout", BlockWiseFP8Layout)

        # Tensorwise INT8 from comfy_kitchen
        try:
            from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
            register_layout_class("TensorWiseINT8Layout", TensorWiseINT8Layout)
            logging.info("ComfyUI-QuantOps: Registered TensorWiseINT8Layout")
        except ImportError:
            logging.debug("ComfyUI-QuantOps: TensorWiseINT8Layout not available")

        # Register QUANT_ALGOS
        QUANT_ALGOS.setdefault(
            "int8_tensorwise",
            {
                "storage_t": torch.int8,
                "parameters": {"weight_scale", "input_scale"}, # Keep input_scale if checkpoints have it
                "comfy_tensor_layout": "TensorWiseINT8Layout", # Must match the class name above
            }
        )
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
            register_layout_class("TensorCoreMXFP8Layout", TensorCoreMXFP8Layout)
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
        try:
            from comfy_kitchen.tensor import HybridMXFP8Layout
            register_layout_class("HybridMXFP8Layout", HybridMXFP8Layout)
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

        # --- QuantHandler registration (PR test) ---
        try:
            from comfy.quant_ops import QuantHandler, register_quant_handler

            def build_int8_tensorwise_params(
                layout_cls, state_dict, prefix, device,
                manually_loaded_keys, compute_dtype,
                out_features, in_features, load_scale_fn,
            ):
                scale = load_scale_fn(state_dict, prefix, "weight_scale", device, manually_loaded_keys)
                return layout_cls.Params(
                    scale=scale,
                    orig_dtype=compute_dtype,
                    orig_shape=(out_features, in_features),
                    is_weight=True,
                )

            def build_int8_blockwise_params(
                layout_cls, state_dict, prefix, device,
                manually_loaded_keys, compute_dtype,
                out_features, in_features, load_scale_fn,
            ):
                scale = load_scale_fn(state_dict, prefix, "weight_scale", device, manually_loaded_keys)
                return layout_cls.Params(
                    scale=scale,
                    orig_dtype=compute_dtype,
                    orig_shape=(out_features, in_features),
                    block_size=128,
                    is_weight=True,
                )

            # TensorWiseINT8Layout (imported above in its own try block)
            try:
                register_quant_handler("int8_tensorwise", QuantHandler(
                    layout_class=TensorWiseINT8Layout,
                    storage_t=torch.int8,
                    parameters={"weight_scale", "input_scale"},
                    comfy_tensor_layout="TensorWiseINT8Layout",
                    build_params=build_int8_tensorwise_params,
                ))
                logging.info("ComfyUI-QuantOps: Registered int8_tensorwise via QuantHandler")
            except NameError:
                logging.debug("ComfyUI-QuantOps: TensorWiseINT8Layout not available for QuantHandler registration")

            # BlockWiseINT8Layout (imported at top of _register_layouts)
            register_quant_handler("int8_blockwise", QuantHandler(
                layout_class=BlockWiseINT8Layout,
                storage_t=torch.int8,
                parameters={"weight_scale", "input_scale"},
                comfy_tensor_layout="BlockWiseINT8Layout",
                group_size=128,
                build_params=build_int8_blockwise_params,
            ))
            logging.info("ComfyUI-QuantOps: Registered int8_blockwise via QuantHandler")

        except ImportError:
            logging.info("ComfyUI-QuantOps: QuantHandler API not available, using legacy registration only")

        # Verify registration
        registered = ["BlockWiseINT8Layout", "TensorWiseINT8Layout", "RowWiseFP8Layout", "BlockWiseFP8Layout", "TensorCoreMXFP8Layout"]
        logging.info(f"ComfyUI-QuantOps: Registered layouts: {registered}")

    except Exception as e:
        logging.error(f"ComfyUI-QuantOps: Failed to register layouts: {e}")


# =============================================================================
# Module Initialization
# =============================================================================

# Setup backends first (enables ck triton, registers our kernels)
_setup_comfy_kitchen_backends()

# Register layouts
_register_layouts()

# Import nodes for ComfyUI discovery
from .nodes.loader_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "is_ck_triton_available",
]
