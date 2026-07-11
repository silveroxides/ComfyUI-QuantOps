"""Optional integration with an installed ComfyUI GGUF loader."""

import functools
import logging
import os
import sys


GGUF_IMAGE_ARCHITECTURES = {"ideogram", "krea2"}


def _provider_name(loader_module) -> str:
    module_file = getattr(loader_module, "__file__", "")
    if module_file:
        return os.path.basename(os.path.dirname(module_file))
    return loader_module.__name__


def install_gguf_integration() -> bool:
    """Enable supported image architectures and diagnostics on the GGUF provider."""
    import nodes

    loader_class = nodes.NODE_CLASS_MAPPINGS.get("UnetLoaderGGUF")
    if loader_class is None:
        logging.info(
            "ComfyUI-QuantOps GGUF diagnostic: provider=unavailable status=disabled "
            "reason=UnetLoaderGGUF-not-registered"
        )
        return False

    method_globals = loader_class.load_unet.__globals__
    gguf_loader = method_globals.get("gguf_sd_loader")
    loader_module = sys.modules.get(getattr(gguf_loader, "__module__", ""))
    architectures = getattr(loader_module, "IMG_ARCH_LIST", None)
    if gguf_loader is None or loader_module is None or not isinstance(architectures, set):
        logging.warning(
            "ComfyUI-QuantOps GGUF diagnostic: provider=%s status=disabled "
            "reason=incompatible-provider-api",
            loader_class.__module__,
        )
        return False

    provider = _provider_name(loader_module)
    already_supported = architectures & GGUF_IMAGE_ARCHITECTURES
    added = GGUF_IMAGE_ARCHITECTURES - architectures
    architectures.update(GGUF_IMAGE_ARCHITECTURES)

    if getattr(gguf_loader, "_quantops_diagnostic_wrapper", False):
        logging.info(
            "ComfyUI-QuantOps GGUF diagnostic: provider=%s status=active "
            "architectures_added=none architectures_supported=%s",
            provider,
            ",".join(sorted(GGUF_IMAGE_ARCHITECTURES)),
        )
        return True

    @functools.wraps(gguf_loader)
    def logged_gguf_loader(path, handle_prefix="model.diffusion_model.", is_text_model=False):
        path_text = os.fspath(path) if path is not None else "<missing>"
        model_kind = "text" if is_text_model else "diffusion"
        logging.info(
            "ComfyUI-QuantOps GGUF diagnostic: decision=gguf-provider provider=%s "
            "model_kind=%s file=%s path=%s",
            provider,
            model_kind,
            os.path.basename(path_text),
            path_text,
        )
        try:
            state_dict, extra = gguf_loader(path, handle_prefix, is_text_model)
        except Exception as error:
            logging.error(
                "ComfyUI-QuantOps GGUF diagnostic: decision=gguf-provider provider=%s "
                "model_kind=%s file=%s result=error error_type=%s error=%s",
                provider,
                model_kind,
                os.path.basename(path_text),
                type(error).__name__,
                error,
            )
            raise

        metadata = extra.get("metadata", {})
        logging.info(
            "ComfyUI-QuantOps GGUF diagnostic: decision=gguf-provider provider=%s "
            "model_kind=%s file=%s result=accepted architecture=%s state_keys=%d "
            "metadata_fields=%d",
            provider,
            model_kind,
            os.path.basename(path_text),
            extra.get("arch_str", "unknown"),
            len(state_dict),
            len(metadata),
        )
        return state_dict, extra

    logged_gguf_loader._quantops_diagnostic_wrapper = True
    loader_module.gguf_sd_loader = logged_gguf_loader
    method_globals["gguf_sd_loader"] = logged_gguf_loader

    logging.info(
        "ComfyUI-QuantOps GGUF diagnostic: provider=%s status=active "
        "architectures_added=%s architectures_already_supported=%s",
        provider,
        ",".join(sorted(added)) or "none",
        ",".join(sorted(already_supported)) or "none",
    )
    return True
