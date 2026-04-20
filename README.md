# ComfyUI-QuantOps

Extended quantization layouts for ComfyUI, enabling loading and inference with models quantized by [convert_to_quant](https://github.com/silveroxides/convert_to_quant).

This is experimental and due to lack of proper support and merging of PR in ComfyUI, do not expect this to work without putting in the effort.
I don't have the time or the energy to keep this up and will close entire project if i keep getting bunch of low effort issues posted expecting me go serve a fix up on a silver platter.


### tl;dr Go complain at ComfyOrg. Not here.

### The following is the last update I make regarding this.

In order to use int8_tensorwise(RTX 30xx-series or newer GPU) or mxfp8(RTX 50xx-series/Blackwell GPU) you will need the following:

- torch 2.10+cu130
- installed the latest of my custom comfy-kitchen fork
- For int8 you also need to merge my PR https://github.com/Comfy-Org/ComfyUI/pull/12730

Step 1: Install Triton
Activate your virtual environment used by ComfyUI and install triton.
For Windows you need to use this but linux can install latest triton as usual.
```
pip install -U "triton-windows<3.7
```

Step 3: Install my comfy-kitchen
Download the latest uploaded version matching you python of my pre-compiled .whl file from my [HuggingFace repository](https://huggingface.co/silveroxides/Chroma1-HD-fp8-scaled/tree/main/experimental)

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
Step 5: Apply the "Triton PR" to Core ComfyUI

you can pull that specific code into a testing branch using these commands:

```
git fetch origin pull/12730/head:triton-testing
git checkout triton-testing
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
