# IC-Custom Model Card

This document provides a comprehensive overview of all models used in IC-Custom.

---

## 🎯 IC-Custom Models

Our core customization models, automatically downloaded when running the app:

| Model | Purpose | Source |
|-------|---------|--------|
| **dit_lora** | LoRA adaptation for diffusion transformer | [TencentARC/IC-Custom](https://huggingface.co/TencentARC/IC-Custom/blob/main/dit_lora_0x1561.safetensors) |
| **img_txt_in** | Image-text input layers weights | [TencentARC/IC-Custom](https://huggingface.co/TencentARC/IC-Custom/blob/main/dit_txt_img_in_0x1561.safetensors) |
| **boundary_embeddings** | Boundary condition embeddings | [TencentARC/IC-Custom](https://huggingface.co/TencentARC/IC-Custom/blob/main/dit_boundary_embeddings_0x1561.safetensors) |
| **task_register_embeddings** | Task registration embeddings | [TencentARC/IC-Custom](https://huggingface.co/TencentARC/IC-Custom/blob/main/dit_task_register_embeddings_0x1561.safetensors) |

---

## 🔧 Base Models

Foundation models required for IC-Custom operation:

| Model | Purpose | Source |
|-------|---------|--------|
| **CLIP** | Vision-language understanding | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) |
| **T5** | Text processing | [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) |
| **SigLIP** | Image understanding | [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) |
| **Autoencoder** | Image encoding/decoding | [black-forest-labs/FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/blob/main/ae.safetensors) |
| **DIT** | Diffusion model | [black-forest-labs/FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/blob/main/flux1-fill-dev.safetensors) |
| **Redux** | Image processing | [black-forest-labs/FLUX.1-Redux-dev](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev) |

---

## 🎨 App Interactive Models

Additional models for enhanced app functionality:

### Required for App
| Model | Purpose | Source |
|-------|---------|--------|
| **SAM-vit-h** | Image segmentation | [HCMUE-Research/SAM-vit-h](https://huggingface.co/HCMUE-Research/SAM-vit-h/blob/main/sam_vit_h_4b8939.pth) |

### Optional (Enable via CLI flags)
| Model | Purpose | Source | Enable Flag |
|-------|---------|--------|-------------|
| **BEN2** | Background removal | [PramaLLC/BEN2](https://huggingface.co/PramaLLC/BEN2/blob/main/BEN2_Base.pth) | `--enable_ben2_for_mask_ref True` |
| **Qwen2.5-VL** | Prompt generation | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | `--enable_vlm_for_prompt True` |

---

## 🔗 Model Usage

- **Automatic Download**: All required models are downloaded automatically when running scripts
- **Manual Configuration**: Model paths can be specified in [`app.yaml`](configs/app/app.yaml) or [`inference.yaml`](configs/inference/inference.yaml)
- **Optional Models**: BEN2 and Qwen2.5-VL are disabled by default and only downloaded when explicitly enabled

For detailed configuration instructions, see:
- [README.md](README.md) - General setup and model download
- [APP.md](src/app/APP.md) - App-specific configuration and CLI arguments
