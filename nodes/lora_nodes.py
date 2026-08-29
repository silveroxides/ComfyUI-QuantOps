"""
LoRA Loader Nodes for Quantized Models

This module provides LoRA loading functionality that works correctly with
all quantization formats (INT8, FP8, MXFP8, NVFP4, BNB 4-bit).

The standard LoRA nodes in ComfyUI may not work with quantized models
because they don't properly handle the QuantizedTensor wrapper and dequantization.
"""

import logging
import torch
import torch.nn.functional as F
import comfy.model_patcher
import comfy.ops
import comfy.utils
import comfy.sd
import comfy.lora
import folder_paths
from comfy.quant_ops import QuantizedTensor


class QuantizedLoRALoader:
    """
    Load LoRA weights and apply them to a quantized model.
    
    This node properly handles quantized weights by using ComfyUI's bypass mode,
    which injects LoRA computation during forward pass without modifying base weights.
    This approach works correctly with all quantization formats.
    
    This ensures LoRA works with all quantization formats including:
    - INT8 (block-wise and tensor-wise)
    - FP8 (row-wise and block-wise)
    - MXFP8, Hybrid MXFP8
    - NVFP4
    - BNB 4-bit (NF4/FP4)
    """

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            },
            "optional": {
                "strength_text_encoder": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "loaders/lora"
    DESCRIPTION = (
        "Load LoRA and apply to quantized models. "
        "This node uses bypass mode which works with all quantization formats."
    )

    def load_lora(self, model, lora_name, strength_model, strength_clip, strength_text_encoder=1.0):
        """
        Load a LoRA and apply it to a quantized model.
        
        This implementation uses ComfyUI's bypass mode (load_bypass_lora_for_models),
        which injects LoRA computation during forward pass without modifying base weights.
        
        Args:
            model: The quantized model to patch
            lora_name: Name of LoRA file
            strength_model: Strength multiplier for UNET/diffusion model
            strength_clip: Strength multiplier for CLIP model
            strength_text_encoder: Strength multiplier for text encoder (if applicable)
        """
        if strength_model == 0 and strength_clip == 0 and strength_text_encoder == 0:
            return (model,)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        # Use bypass mode which works with quantized models
        clip = getattr(model, 'clip', None)
        model_lora, clip_lora = comfy.sd.load_bypass_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )

        # Apply text encoder strength if different from clip strength
        if strength_text_encoder != strength_clip and clip_lora is not None:
            # Re-apply with text encoder strength
            model_lora, clip_lora = comfy.sd.load_bypass_lora_for_models(
                model_lora, clip_lora, lora, 0, strength_text_encoder
            )

        logging.info(f"QuantizedLoRALoader: Applied LoRA {lora_name} with strength_model={strength_model}, strength_clip={strength_clip}")
        
        return (model_lora,)


class QuantizedLoraStack:
    """
    Stack multiple LoRAs and apply them to a quantized model.
    
    Similar to rgthree-comfy's Power Lora Loader, this node allows you to:
    - Add multiple LoRAs with individual on/off toggles
    - Set individual strengths for each LoRA
    - Enable/disable LoRAs without removing them from the node
    
    This approach works correctly with all quantization formats.
    """

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "lora_1": (["None"] + lora_list, {"default": "None"}),
                "lora_1_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "lora_1_on": ("BOOLEAN", {"default": True}),
                "lora_2": (["None"] + lora_list, {"default": "None"}),
                "lora_2_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "lora_2_on": ("BOOLEAN", {"default": True}),
                "lora_3": (["None"] + lora_list, {"default": "None"}),
                "lora_3_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "lora_3_on": ("BOOLEAN", {"default": True}),
                "lora_4": (["None"] + lora_list, {"default": "None"}),
                "lora_4_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "lora_4_on": ("BOOLEAN", {"default": True}),
                "lora_5": (["None"] + lora_list, {"default": "None"}),
                "lora_5_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "lora_5_on": ("BOOLEAN", {"default": True}),
                "lora_6": (["None"] + lora_list, {"default": "None"}),
                "lora_6_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "lora_6_on": ("BOOLEAN", {"default": True}),
                "lora_7": (["None"] + lora_list, {"default": "None"}),
                "lora_7_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "lora_7_on": ("BOOLEAN", {"default": True}),
                "lora_8": (["None"] + lora_list, {"default": "None"}),
                "lora_8_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "lora_8_on": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "stack_loras"
    CATEGORY = "loaders/lora"
    DESCRIPTION = (
        "Stack multiple LoRAs and apply to quantized models. "
        "Each LoRA has an on/off toggle and individual strength control."
    )

    def stack_loras(self, model, 
                   lora_1=None, lora_1_strength=1.0, lora_1_on=True,
                   lora_2=None, lora_2_strength=1.0, lora_2_on=True,
                   lora_3=None, lora_3_strength=1.0, lora_3_on=True,
                   lora_4=None, lora_4_strength=1.0, lora_4_on=True,
                   lora_5=None, lora_5_strength=1.0, lora_5_on=True,
                   lora_6=None, lora_6_strength=1.0, lora_6_on=True,
                   lora_7=None, lora_7_strength=1.0, lora_7_on=True,
                   lora_8=None, lora_8_strength=1.0, lora_8_on=True):
        """
        Stack multiple LoRAs and apply to a quantized model.
        
        Args:
            model: The quantized model to patch
            lora_1-8: LoRA filenames (or "None" to skip)
            lora_1-8_strength: Strength multipliers for each LoRA
            lora_1-8_on: Enable/disable toggle for each LoRA
        """
        # Collect enabled LoRAs
        lora_configs = []
        
        lora_vars = [
            (lora_1, lora_1_strength, lora_1_on),
            (lora_2, lora_2_strength, lora_2_on),
            (lora_3, lora_3_strength, lora_3_on),
            (lora_4, lora_4_strength, lora_4_on),
            (lora_5, lora_5_strength, lora_5_on),
            (lora_6, lora_6_strength, lora_6_on),
            (lora_7, lora_7_strength, lora_7_on),
            (lora_8, lora_8_strength, lora_8_on),
        ]
        
        for lora_name, strength, is_on in lora_vars:
            if lora_name is not None and lora_name != "None" and is_on and strength != 0:
                lora_configs.append((lora_name, strength))
        
        if not lora_configs:
            return (model,)
        
        # Apply each LoRA sequentially using bypass mode
        # IMPORTANT: load_bypass_lora_for_models returns a CLONE of the model
        # We must pass the returned model to the next LoRA call
        result_model = model
        result_clip = getattr(model, 'clip', None)
        
        for lora_name, strength in lora_configs:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            
            # Apply using bypass mode - pass the updated model/clip from previous iteration
            result_model, result_clip = comfy.sd.load_bypass_lora_for_models(
                result_model, result_clip, lora, strength, strength
            )
        
        logging.info(f"QuantizedLoraStack: Applied {len(lora_configs)} LoRAs to quantized model")
        
        return (result_model,)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "QuantizedLoRALoader": QuantizedLoRALoader,
    "QuantizedLoraStack": QuantizedLoraStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QuantizedLoRALoader": "Load LoRA (Quantized)",
    "QuantizedLoraStack": "Stack LoRAs (Quantized)",
}
