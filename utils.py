import os
import logging

logger = logging.getLogger(__name__)


def get_llm_checkpoints():
    """
    Scan models/LLM directory for available checkpoints
    Returns list of checkpoint directories/files
    """
    llm_path = os.path.join("models", "LLM")
    checkpoints = []
    
    if os.path.exists(llm_path):
        for item in os.listdir(llm_path):
            item_path = os.path.join(llm_path, item)
            if os.path.isdir(item_path):
                # Check if it's a valid model directory (contains config.json or similar)
                if any(f in os.listdir(item_path) for f in ['config.json', 'model.safetensors', 'pytorch_model.bin']):
                    checkpoints.append(item_path)
            elif item.endswith(('.safetensors', '.bin', '.pt')):
                checkpoints.append(item_path)
    
    if not checkpoints:
        checkpoints = ["./gemma-3-1b-it"]  # Default fallback
    
    return checkpoints


def get_llm_adapters():
    """
    Scan models/llm_adapters directory for available adapters
    Returns list of adapter files
    """
    adapters_path = os.path.join("models", "llm_adapters")
    adapters = []
    
    if os.path.exists(adapters_path):
        for item in os.listdir(adapters_path):
            if item.endswith('.safetensors'):
                adapters.append(os.path.join(adapters_path, item))
    
    if not adapters:
        adapters = ["checkpoint_epoch_10.safetensors"]  # Default fallback
    
    return adapters
