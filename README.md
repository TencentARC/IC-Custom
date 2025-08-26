<div align="center">
<img alt="ic-custom" src="./assets/IC-Custom-Icon.png">



<p align="center">
  <a href="https://liyaowei-stu.github.io/project/IC_Custom/"><strong>Project Page</strong></a> |
  <a href="https://arxiv.org/abs/2507.01926"><strong>Paper</strong></a> |
  <a href="https://huggingface.co/spaces/TencentARC/IC-Custom"><strong>Demo</strong></a> |
  <a href="https://huggingface.co/TencentARC/IC-Custom"><strong>Model</strong></a>
</p>
</div>


## 📋 Overview

IC-Custom is designed for diverse image customization scenarios, including:

- **Position-aware**: Input a reference image, target background, and specify the customization location (via segmentation or drawing)
  - *Examples*: Product placement, virtual try-on

- **Position-free**: Input a reference image and a target description to generate a new image with the reference image's ID
  - *Examples*: IP customization and creation

---

## 📑 Table of Contents
- [📋 Overview](#-overview)
- [📑 Table of Contents](#-table-of-contents)
- [🚀 Environment Requirements](#-environment-requirements)
- [🔧 Installation](#-installation)
- [📦 Model Checkpoints](#-model-checkpoints)
- [💻 Running Scripts](#-running-scripts)
  - [App (Gradio Interface)](#app-gradio-interface)
  - [Inference](#inference)
- [📅 Update Logs](#-update-logs)
- [📝 Citation](#-citation)
- [💖 Acknowledgements](#-acknowledgements)
- [📄 License](#-license)
- [📬 Contact](#-contact)
- [🌟 Star History](#-star-history)

---

## 🚀 Environment Requirements

IC-Custom has been implemented and tested on:
- CUDA 12.4
- PyTorch 2.6.0
- Python 3.10.16

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/TencentARC/IC-Custom.git
   cd IC-Custom
   ```

2. **Set up Python environment**
   ```bash
   conda create -n ic-custom python=3.10 -y
   conda activate ic-custom
   pip install -e .
   pip install -r requirements.txt
   ```

3. **Custom CUDA versions (optional)**
   If you require a different CUDA version, you can ignore the torch-related packages listed in `requirements.txt`. Instead, please install PyTorch and xformers that are compatible with your CUDA version by following the instructions on the [official PyTorch website](https://pytorch.org/get-started/previous-versions/).
   
   Example for CUDA 12.4:
   ```bash
   pip3 install xformers torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
   ```

4. **Flash Attention 2 (optional)**
   If your device supports Flash Attention 2, you can optionally install flash-attn. We use flash-attn==2.7.3. Find compatible versions at [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases).

## 📦 Model Checkpoints

You'll need to obtain the model checkpoints before running the inference or app.

- Option 1 — Automatic download: checkpoints are fetched when running scripts (requires valid `HF_TOKEN`).
- Option 2 — Manual download: use the helper script and then set paths in configs.

```bash
sh scripts/inference/download_models.sh $HF_TOKEN
```

Expected directory structure (example):
```
|-- models
    |-- clip-vit-large-patch14
    |-- ic-custom
    |   |-- dit_boundary_embeddings_0x1561.safetensors
    |   |-- dit_lora_0x1561.safetensors
    |   |-- dit_task_register_embeddings_0x1561.safetensors
    |   |-- dit_txt_img_in_0x1561.safetensors
    |   ...
    |-- siglip-so400m-patch14-384
    |-- t5-v1_1-xxl
    |-- ae.safetensors
    |-- flux1-fill-dev.safetensors
    |-- flux1-redux-dev.safetensors
```

After manual download, edit the YAMLs in `configs/**` (e.g., `configs/inference/inference.yaml`) to point to your local model paths.

- A compact per-model list with links is available in `MODEL_CARD.md`.

## 💻 Running Scripts

### App (Gradio Interface)

When running the app, **all required models are automatically downloaded**. Optional models are fetched only when explicitly enabled.

```bash
sh src/app/run_app.sh $HF_TOKEN $HF_CACHE_DIR
```

For required vs optional models and configuration, see [src/app/APP.md](src/app/APP.md).

### Inference

Run the inference script with your Hugging Face token:

```bash
sh scripts/inference/inference.sh $HF_TOKEN $HF_CACHE_DIR
```

Parameters:
- `$HF_TOKEN`: Your Hugging Face access token (required for automatic model download, optional if model paths are specified in `configs/**/*.yaml`)
- `$HF_CACHE_DIR` (optional): Custom cache directory for downloaded models (default: "~/.cache/huggingface/hub")

## 📅 Update Logs

- [X] **2025/07/03** - Released paper, webpage.
- [X] **2025/08/26** - Released Checkpoint v0x1561, HF Demo, and Inference Code.
- [ ] **TBD** - Test and Training Code.

## 📝 Citation

```bibtex
@article{li2025ic,
  title={IC-Custom: Diverse Image Customization via In-Context Learning},
  author={Li, Yaowei and Li, Xiaoyu and Zhang, Zhaoyang and Bian, Yuxuan and Liu, Gan and Li, Xinyuan and Xu, Jiale and Hu, Wenbo and Liu, Yating and Li, Lingen and others},
  journal={arXiv preprint arXiv:2507.01926},
  year={2025}
}
```

## 💖 Acknowledgements

We gratefully acknowledge the use of code from:
- [flux](https://github.com/black-forest-labs/flux)
- [x-flux](https://github.com/XLabs-AI/x-flux/releases)

We also thank [Hugging Face](https://huggingface.co/) for providing professional model hosting and Spaces for deployment.


## 📄 License

We are pleased to support the open source community. For complete license details, see [LICENSE](LICENSE) and [NOTICE](NOTICE). 

## 📬 Contact

For any questions, feel free to email: `liyaowei01@gmail.com`

## 🌟 Star History

<p align="center">
    <a href="https://star-history.com/#TencentARC/IC-Custom" target="_blank">
        <img width="500" src="https://api.star-history.com/svg?repos=TencentARC/IC-Custom&type=Date" alt="Star History Chart">
    </a>
</p>