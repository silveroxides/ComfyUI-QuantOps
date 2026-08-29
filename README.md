# ComfyUI-QuantOps IS NO LONGER NEEDED SO IT HAS BEEN DEPRECATED

## Support for int8 ConvRot quants is built into ComfyUI now


Additional quantized tensor layouts and model loaders for ComfyUI. The metadata formats are produced by [convert_to_quant](https://github.com/silveroxides/convert_to_quant).

## Automatic routing

QuantOps records ComfyUI's native quantization formats before registering its own layouts. Normal checkpoint and diffusion-model loaders then route each model from its embedded metadata:

1. Models using only ComfyUI-native formats stay on the ComfyUI path.
2. Models containing a QuantOps-only format receive QuantOps custom operations.
3. Unquantized and unknown formats are left to ComfyUI instead of being claimed by QuantOps.

This is automatic in normal ComfyUI workflows and in SwarmUI-generated workflows. Swarm presets and dtype overrides continue through the original loader arguments.

The log reports every decision as `decision=native-bypass`, `decision=quantops`, `decision=no-quantization`, or `decision=core-pass-through`.

## QuantOps-only metadata formats

These canonical metadata formats are not native to the current ComfyUI quantization registry and have been verified through the automatic loader path:

| Metadata format | Storage and scaling | Default execution |
| --- | --- | --- |
| `int8_blockwise` | INT8 weights with a 2D block-scale grid, normally 128 x 128 blocks | PyTorch fallback by default; optional Triton backend in the Quantized loaders |
| `float8_e4m3fn_rowwise` | FP8 E4M3 weights with one scale per output row | QuantOps Triton kernel on compatible CUDA systems; dequantized fallback otherwise |
| `float8_e4m3fn_blockwise` | FP8 E4M3 weights with a 2D block-scale grid, normally 64 x 64 blocks | QuantOps Triton kernel on compatible CUDA systems; dequantized fallback otherwise |

The legacy metadata name `int8` is accepted as a blockwise INT8 compatibility alias when the scale shape identifies a block layout. New files should use `int8_blockwise`.

### Conditional format

`hybrid_mxfp8` describes 32-element MXFP8 blocks with an additional weight scalar. It is usable only when the installed `comfy-kitchen` exports `HybridMXFP8Layout`. The installed `comfy-kitchen` 0.2.17 in the tested environment does not export that class, so this format is not available there.

## Dedicated BNB 4-bit formats

The `BNB4bitUNETLoader` node supports bitsandbytes-compatible `NF4` and `FP4` safetensors without requiring bitsandbytes at runtime. It uses pure PyTorch dequantization and currently targets Flux, Flux 2, Chroma, Chroma Radiance, and Chroma Radiance X0 model structures.

NF4 and FP4 use their dedicated loader. They are not automatic `QUANT_ALGOS` metadata formats.

## Formats already native to ComfyUI

QuantOps does not replace ComfyUI operations for these formats:

- `convrot_w4a4`
- `float8_e4m3fn`
- `float8_e5m2`
- `int8_tensorwise`, including per-channel files whose names may say rowwise
- `mxfp8`
- `nvfp4`

For these models the expected log is `decision=native-bypass` followed by `path=core result=completed`. Embedded metadata is authoritative; filenames are not used to select a quantization layout.

## GGUF integration

GGUF is delegated to an installed provider that registers `UnetLoaderGGUF`, such as [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF). QuantOps does not duplicate or replace its GGUF tensor implementation.

When the provider is present, QuantOps adds automatic image-architecture recognition for:

- `krea2`
- `ideogram`

SwarmUI already selects `UnetLoaderGGUF` for `.gguf` diffusion models. QuantOps extends the provider in memory and logs the detected architecture, GGUF tensor types, state-key count, and acceptance or error result. If no provider is installed, GGUF integration is disabled without affecting other formats.

## Loader coverage

- Stock ComfyUI diffusion-model loaders: automatic native or QuantOps routing.
- Stock ComfyUI checkpoint loaders: automatic routing for the diffusion-model portion.
- `QuantizedModelLoader` and `QuantizedUNETLoader`: explicit format and backend controls.
- `QuantizedCLIPLoader` and `QuantizedDualCLIPLoader`: quantized text-encoder loading.
- `BNB4bitUNETLoader`: dedicated NF4 and FP4 loading.

## Runtime diagnostics

Useful successful log sequences include:

```text
ComfyUI-QuantOps diagnostic: loader=diffusion decision=native-bypass ...
ComfyUI-QuantOps diagnostic: loader=diffusion path=core result=completed

ComfyUI-QuantOps diagnostic: loader=diffusion decision=quantops ...
ComfyUI-QuantOps FP8 blockwise: path=triton-dynamic ...
ComfyUI-QuantOps diagnostic: loader=diffusion path=quantops result=completed

ComfyUI-QuantOps GGUF diagnostic: decision=gguf-provider ...
ComfyUI-QuantOps GGUF diagnostic: result=accepted architecture=krea2 ...
```

Repeated kernel-path messages are rate-limited. If an FP8 Triton kernel fails, QuantOps disables that kernel path for the session, emits one concise warning, and uses the dequantized fallback.

## Triton

Use a Triton build compatible with the PyTorch and CUDA versions in the ComfyUI virtual environment. On Windows this normally means `triton-windows`. The ComfyUI launch option below enables its `comfy-kitchen` Triton backend:

```text
--enable-triton-backend
```

QuantOps FP8 kernels are selected when a compatible Triton installation and CUDA device are available. The explicit Quantized loaders expose the blockwise INT8 backend choice.

## Updating

```text
cd custom_nodes/ComfyUI-QuantOps
git pull
```

Quantized model releases are available from [silveroxides on Hugging Face](https://huggingface.co/silveroxides).

## License

MIT License

## Acknowledgements

- [lyogavin](https://github.com/lyogavin) for [ComfyUI PR #10864](https://github.com/comfyanonymous/ComfyUI/pull/10864).
- [Clybius](https://github.com/Clybius) for the [Learned-Rounding](https://github.com/Clybius/Learned-Rounding) project.
