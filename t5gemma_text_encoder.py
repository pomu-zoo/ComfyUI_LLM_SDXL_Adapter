import torch
import logging

logger = logging.getLogger("LLM-SDXL-Adapter")


class T5GEMMATextEncoder:
    """
    ComfyUI node that encodes text using a loaded Language Model
    Supports various LLM architectures with chat templates
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LLM_MODEL",),
                "tokenizer": ("LLM_TOKENIZER",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "masterpiece, best quality, 1girl, anime style"
                }),
            }
        }
    
    RETURN_TYPES = ("LLM_HIDDEN_STATES", "STRING")
    RETURN_NAMES = ("hidden_states", "info")
    FUNCTION = "encode_text"
    CATEGORY = "llm_sdxl"
    
    def encode_text(self, model, tokenizer, text, system_prompt="You are expert in understanding of user prompts for image generations. Create an image according to the prompt from user.", skip_first=27):
        """
        Encode text using Language Model and return hidden states
        """
        try:
            # Get model device
            device = next(model.parameters()).device
            
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding = "max_length",
                max_length=512, 
                truncation = True,
            ).to(device)
            
            # Generate hidden states
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Extract hidden states, skipping first tokens
            hidden_states = outputs['last_hidden_state'].to(torch.float)
            
            # Prepare info
            info = f"Text: {text[:50]}...\nEncoded: {hidden_states.shape[1]}\nShape: {hidden_states.shape}"
            
            logger.info(f"Encoded text with shape: {hidden_states.shape}")
            
            return (hidden_states, info)
            
        except Exception as e:
            logger.error(f"Failed to encode text: {str(e)}")
            raise Exception(f"Text encoding failed: {str(e)}")


# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "T5GEMMATextEncoder": T5GEMMATextEncoder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "T5GEMMATextEncoder": "T5GemmaText Encoder"
} 