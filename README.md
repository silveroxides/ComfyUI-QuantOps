# ComfyUI-QuantOps IS NO LONGER NEEDED SO IT HAS BEEN DEPRECATED

## Support for int8 ConvRot quants is built into ComfyUI now


Extended quantization layouts for ComfyUI, enabling loading and inference with models quantized by [convert_to_quant](https://github.com/silveroxides/convert_to_quant).

This is experimental and due to lack of proper support and merging of PR in ComfyUI, do not expect this to work without putting in the effort.
I don't have the time or the energy to keep this up and will close entire project if i keep getting bunch of low effort issues posted expecting me go serve a fix up on a silver platter.


### tl;dr Go complain at ComfyOrg. Not here.

### The following is the last update I make regarding this.

In order to use int8_tensorwise(RTX 30xx-series or newer GPU) you will need the following:

- torch 2.10+cu130 or higher
- installed the latest of my custom comfy-kitchen fork wheels with the int8-tensorwise support
- enable the use of triton backend by using --enable-triton-backend launch argument in ComfyUI

Step 1: Install Triton
Activate your virtual environment used by ComfyUI and install triton.
For Windows you need to use this but linux can install latest triton as usual.
```
# for torch 2.10 and 2.11
pip install -U "triton-windows<3.7"
# for torch 2.12
pip install -U "triton-windows<3.8"
```

Step 3: Install my comfy-kitchen
Download the latest uploaded version matching you python of my pre-compiled .whl file from my [HuggingFace repository](https://huggingface.co/silveroxides/comfy-kitchen-int8-wheels/tree/main) (Latest as of 13 June 2026)

Install it directly pointing to the file path:
```
pip install --no-deps --force-reinstall --no-cache-dir "path/to/comfy-kitchen.whl"
```

Step 4: Install/Update ComfyUI-QuantOps
You just need to ensure it's fully up to date to read the new model formats.
Run these commands:

```
cd custom_nodes/ComfyUI-QuantOps
git pull
```

When launching Comfyui add launch argument:
```
--enable-triton-backend
```

You can get most of the models here: https://huggingface.co/silveroxides

## License

MIT License

## Acknowledgements

- [lyogavin](https://github.com/lyogavin) for [PR #10864](https://github.com/comfyanonymous/ComfyUI/pull/10864) to ComfyUI.
- [Clybius](https://github.com/Clybius) for inspiring me to take on quantization and his [Learned-Rounding](https://github.com/Clybius/Learned-Rounding) repository.
