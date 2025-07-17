import os
import logging
import folder_paths

logger = logging.getLogger(__name__)

def get_llm_dict():
    """
    Get the dictionary of LLM checkpoints.
    Keys are the names of the LLM checkpoints, values are the paths to the LLM checkpoints.
    """
    llm_dict = {}
    llm_path = os.path.join(folder_paths.models_dir, "LLM")

    if os.path.exists(llm_path):
        for item in os.listdir(llm_path):
            item_path = os.path.join(llm_path, item)
            if os.path.isdir(item_path):
                # Check if it's a valid model directory (contains config.json or similar)
                if any(f in os.listdir(item_path) for f in ['config.json', 'model.safetensors', 'pytorch_model.bin']):
                    llm_dict[item] = item_path
            elif item.endswith(('.safetensors', '.bin', '.pt')):
                llm_dict[item] = item_path

    return llm_dict
    
def get_adapters_dict():
    """
    Get the dictionary of LLM adapters.
    Keys are the names of the LLM adapters, values are the paths to the LLM adapters.
    """
    adapters_dict = {}
    adapters_path = os.path.join(folder_paths.models_dir, "llm_adapters")

    if os.path.exists(adapters_path):
        for item in os.listdir(adapters_path):
            if item.endswith('.safetensors'):
                adapters_dict[item] = os.path.join(adapters_path, item)

    return adapters_dict

def get_llm_checkpoints():
    """
    Get the list of available LLM checkpoints.
    """
    return list(get_llm_dict().keys())

def get_llm_adapters():
    """
    Get the list of available LLM adapters.
    """
    return list(get_adapters_dict().keys())

def get_llm_checkpoint_path(model_name):
    """
    Get the path to a LLM checkpoint.
    """
    llm_dict = get_llm_dict()

    if model_name in llm_dict:
        return llm_dict[model_name]
    else:
        raise ValueError(f"Model {model_name} not found")

def get_llm_adapter_path(adapter_name):
    """
    Get the path to an LLM adapter.
    """
    adapters_dict = get_adapters_dict()

    if adapter_name in adapters_dict:
        return adapters_dict[adapter_name]
    else:
        raise ValueError(f"Adapter {adapter_name} not found")
