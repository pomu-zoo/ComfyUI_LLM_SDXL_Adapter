# ComfyUI LLM SDXL Adapter

![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-orange.svg)

A comprehensive set of ComfyUI nodes for using Large Language Models (LLM) as text encoders for SDXL image generation through a trained adapter.

<img width="1803" height="904" alt="image" src="https://github.com/user-attachments/assets/e8e5f047-37e7-4f8b-9bbd-78d70e2a7d80" />

[Image with workflow](https://files.catbox.moe/om6tc4.png)


## 🎯 Available Adapters

### RouWei-Gemma Adapter 
Trained adapter for using Gemma-3-1b as text encoder for [Rouwei v0.8](https://civitai.com/models/950531) (vpred or epsilon or [base](https://huggingface.co/Minthy/RouWei-0.8/blob/main/rouwei_080_base_fp16.safetensors)).

**Download Links:**
- [CivitAI Model](https://civitai.com/models/1782437)
- [HuggingFace Repository](https://huggingface.co/Minthy/RouWei-Gemma)

## 📦 Installation

### Requirements
- Python 3.8+
- ComfyUI
- Latest transformers library (tested on 4.53.1)

### Install Dependencies
```bash
pip install transformers>=4.53.1 safetensors einops torch
```

### Install Nodes
1. Clone the repository to `ComfyUI/custom_nodes/`:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/NeuroSenko/ComfyUI_LLM_SDXL_Adapter.git
```

2. Restart ComfyUI

### Setup RouWei-Gemma Adapter

1. **Download the adapter:**
   - Download from [CivitAI](https://civitai.com/models/1782437) or [HuggingFace](https://huggingface.co/Minthy/RouWei-Gemma)
   - Place the adapter file in `ComfyUI/models/llm_adapters/`

2. **Download Gemma-3-1b-it model:**
   - Download [gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) ([non-gated mirror](https://huggingface.co/unsloth/gemma-3-1b-it))
   - Place in `ComfyUI/models/llm/gemma-3-1b-it/`
   - **Note:** You need ALL files from the original model for proper functionality (not just .safetensors)

3. **Download Rouwei checkpoint:**
   - Get [Rouwei v0.8](https://civitai.com/models/950531) (vpred, epsilon, or [base](https://huggingface.co/Minthy/RouWei-0.8/blob/main/rouwei_080_base_fp16.safetensors)) if you don't have it
   - Place in your regular ComfyUI checkpoints folder

## 📁 File Structure Example

- **LLM Models:** `ComfyUI/models/llm/gemma-3-1b-it/` (all model files)
- **Adapters:** `ComfyUI/models/llm_adapters/rouweiGemma_g31b27k.safetensors`
- **Checkpoints:** `ComfyUI/models/checkpoints/rouwei_v0.8_vpred.safetensors`

## 🔍 Debugging

To enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.INFO)
```
