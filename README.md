# ComfyUI LLM SDXL Adapter

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-orange.svg)

A comprehensive set of ComfyUI nodes for using Large Language Models (LLM) as text encoders for SDXL image generation through a trained adapter.

<img width="1795" alt="image" src="https://github.com/user-attachments/assets/a2685d2b-3c19-44ad-8005-037bcf900859" />

[Image with workflow](https://files.catbox.moe/f45gnj.png)

## 🚀 Key Features

- **Multi-LLM Support**: Gemma and other compatible models (extensible architecture)
- **Flexible Architecture**: Modular nodes for customizable workflows
- **Transformer Adapter**: Convert LLM embeddings to SDXL format
- **Chat Templates**: Support for system prompts and conversation formatting
- **Memory Optimization**: Model loading management and caching
- **Configurable Parameters**: Flexible adapter architecture configuration

## 📦 Installation

### Requirements
- Python 3.8+
- ComfyUI
- CUDA (recommended)
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

## 🗂️ Project Structure

```
comfyui_llm_sdxl_adapter/
├── __init__.py                  # Entry point and node registration
├── llm_model_loader.py          # LLM model loader
├── llm_text_encoder.py          # LLM text encoder
├── lll_adapter_loader.py        # Adapter loader
├── llm_to_sdxl_adapter.py       # Adapter architecture
├── apply_llm_to_sdxl_adapter.py # Adapter application
├── utils.py                     # Utility functions
└── README.md                    # Documentation
```

## 🔧 ComfyUI Nodes

### LLM Model Loader
Loads language model and tokenizer.

**Inputs:**
- `model_path`: Path to LLM model
- `device`: Device (auto/cuda:0/cuda:1/cpu)
- `force_reload`: Force reload model

**Outputs:**
- `model`: LLM model instance
- `tokenizer`: Tokenizer
- `info`: Model information

### LLM Text Encoder
Encodes text using loaded LLM.

**Inputs:**
- `model`: LLM model
- `tokenizer`: Tokenizer
- `text`: Text to encode
- `system_prompt`: System prompt (optional)
- `skip_first`: Number of tokens to skip

**Outputs:**
- `hidden_states`: LLM hidden states
- `info`: Encoding information

### LLM Adapter Loader
Loads trained LLM -> SDXL adapter.

**Inputs:**
- `adapter_path`: Path to adapter file (.safetensors)
- `llm_dim`: LLM dimension (default 1152)
- `sdxl_seq_dim`: SDXL sequence dimension (2048)
- `sdxl_pooled_dim`: SDXL pooled output dimension (1280)
- `target_seq_len`: Target sequence length (308)
- `n_wide_blocks`: Number of wide blocks (2)
- `n_narrow_blocks`: Number of narrow blocks (3)

**Outputs:**
- `adapter`: Adapter instance
- `info`: Adapter information

### Apply LLM To SDXL Adapter
Applies adapter to LLM hidden states.

**Inputs:**
- `llm_hidden_states`: LLM hidden states
- `adapter`: Loaded adapter

**Outputs:**
- `conditioning`: SDXL conditioning
- `info`: Application information

## 🎯 Basic Workflow

```
LLM Model Loader → LLM Text Encoder → LLM Adapter Loader → Apply LLM To SDXL Adapter → KSampler
```

### Usage Example:

1. **Load Model**: Use `LLM Model Loader` to load an LLM (e.g., Gemma-3-1b-it)
2. **Encode Text**: Connect model to `LLM Text Encoder` and input your prompt
3. **Load Adapter**: Use `LLM Adapter Loader` to load trained adapter
4. **Apply Adapter**: Connect hidden states and adapter through `Apply LLM To SDXL Adapter`
5. **Generate**: Connect the resulting conditioning to standard SDXL pipeline

## 📁 File Structure

### LLM Models
Place LLM models in the folder:
```
ComfyUI/models/LLM/
└── gemma-3-1b-it/
```

### Adapters
Place trained adapters in the folder:
```
ComfyUI/models/llm_adapters/
└── gemma_to_sdxl_adapter.safetensors
```

## ⚙️ Adapter Architecture

The `LLMToSDXLAdapter` uses a transformer architecture with two stages:

1. **Wide Processing**: Process full sequence (512 tokens)
2. **Compression**: Compress to target length (308 tokens) via cross-attention
3. **Narrow Processing**: Final processing of compressed sequence
4. **Pooling**: Create pooled output for SDXL

### Key Features:
- Positional embeddings for input and output sequences
- Learnable compression queries for controlled compression
- Attention masking for proper padding handling
- Dropout and LayerNorm for training stability

## 🔍 Debugging

To enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Information about loaded nodes is displayed during ComfyUI initialization.
