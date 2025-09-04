[English](README.md) | **简体中文**

<div align="center">

<img alt="ic-custom" src="./assets/IC-Custom-Icon.png">

<br>

<p align="center">
  <a href="https://liyaowei-stu.github.io/project/IC_Custom/"><strong>项目主页</strong></a> |
  <a href="https://arxiv.org/abs/2507.01926"><strong>论文</strong></a> |
  <a href="https://huggingface.co/spaces/TencentARC/IC-Custom"><strong>在线演示</strong></a> |
  <a href="https://huggingface.co/TencentARC/IC-Custom"><strong>模型</strong></a> |
  <a href="https://www.youtube.com/watch?v=uaiZA3H5RVY"><strong>介绍视频</strong></a>
</p>
</div>


## 📋 概述

IC-Custom 面向多样化的图像定制场景，包括：

- **位置感知模式 (输入参考图、目标图和遮罩）**：
  - 输入参考图、目标背景，并通过分割或涂抹指定定制位置。
  - 示例：商品摆放、虚拟试穿

- **无位置模式（无输入遮罩）**：
  - 输入参考图与文字描述，生成保持参考 ID 的新图像。
  - 示例：IP 定制与创作

<p align="center">
  <img src="assets/teaser-github.jpeg" alt="IC-Custom Teaser" width="80%">
</p>

---

## ✨ 说明

- 当前为 IC-Custom 的**首次发布**（版本 0x1561）。  
- 模型在**遮罩式定制**方面表现更强，尤其适合**包袋、香水、服饰及其他刚性物体**。  
- 训练数据保证清晰度（≥800px），但规模与多样性仍有限；暂不包含人脸相关数据与完整风格迁移。  
- **无遮罩定制**（IP 定制）与人脸等能力仍在持续优化，欢迎社区反馈。  
- 正在探索**加速**（量化、蒸馏）与更细粒度的控制（如产品定制化视角）。  

**资源**

- **ComfyUI**： [ComfyUI_RH_ICCustom](https://github.com/HM-RunningHub/ComfyUI_RH_ICCustom)  
- **RunningHub**：
  - **视频**： [T8star-Aix ComfyUI 演示（Bilibili）](https://www.bilibili.com/video/BV17gaCz7EWM/?spm_id_from=333.337.search-card.all.click&vd_source=b08a459ef4b115fe7614b270fe47627a)  
  - [IC Custom 角色+场景迁移 V1](https://www.runninghub.cn/post/1963310792110215170/?inviteCode=rh-v1121)  
  - [IC Custom 万物迁移 双图参考 V1](https://www.runninghub.cn/post/1963304787548811266/?inviteCode=rh-v1121)  
  - [IC Custom 换装迁移 V1](https://www.runninghub.cn/post/1963310832354562049/?inviteCode=rh-v1121)  
  - [IC Custom 万物迁移 单图参考 V1](https://www.runninghub.cn/post/1963307022995402753/?inviteCode=rh-v1121)  

致谢：ComfyUI 部署由 [HM-RunningHub](https://github.com/HM-RunningHub) 提供支持；RunningHub 工作流由 [T8star-Aix](https://www.runninghub.cn/user-center/1819214514410942465) 贡献。  

---

## 📑 目录
- [📋 概述](#-概述)
- [✨ 说明](#-说明)
- [📑 目录](#-目录)
- [🚀 环境依赖](#-环境依赖)
- [🔧 安装](#-安装)
- [📦 模型检查点](#-模型检查点)
- [💻 运行脚本](#-运行脚本)
  - [应用（Gradio 界面）](#应用gradio-界面)
  - [推理](#推理)
- [📅 更新日志](#-更新日志)
- [📝 引用](#-引用)
- [💖 致谢](#-致谢)
- [限制](#限制)
- [📄 许可证](#-许可证)
- [📬 联系方式](#-联系方式)
- [🌟 Star 历史](#-star-历史)

---

## 🚀 环境依赖

IC-Custom 在以下环境中实现并测试：
- CUDA 12.4
- PyTorch 2.6.0
- Python 3.10.16

## 🔧 安装

1. 克隆仓库
   ```bash
   git clone https://github.com/TencentARC/IC-Custom.git
   cd IC-Custom
   ```

2. 配置 Python 环境
   ```bash
   conda create -n ic-custom python=3.10 -y
   conda activate ic-custom
   pip install -e .
   pip install -r requirements.txt
   ```

3. 自定义 CUDA 版本（可选）
   若需其他 CUDA 版本，可忽略 `requirements.txt` 中与 torch 相关的包，转而根据 [PyTorch 官方网站](https://pytorch.org/get-started/previous-versions/) 指引安装与你 CUDA 版本兼容的 PyTorch 与 xformers。
   
   CUDA 12.4 示例：
   ```bash
   pip3 install xformers torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
   ```

4. Flash Attention 2（可选）
   若设备支持 Flash Attention 2，可选装 flash-attn。我们使用 flash-attn==2.7.3。兼容版本见 [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases)。

## 📦 模型检查点

在运行应用或推理前，需要获取模型检查点：

- 方式一 — 脚本自动下载（需有效的 `HF_TOKEN`）
- 方式二 — 手动下载：使用辅助脚本后，在配置文件中设置本地路径

```bash
sh scripts/inference/download_models.sh $HF_TOKEN
```

期望的目录结构（示例）：
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

手动下载后，请修改 `configs/**` 下的 YAML（如 [`inference.yaml`](configs/inference/inference.yaml)）以指向本地模型路径。

- 各模型的精简链接清单见 [`MODEL_CARD.md`](MODEL_CARD.md)。

## 💻 运行脚本

### 应用（Gradio 界面）

运行应用时，**必须的模型会自动下载**；可选模型仅在显式启用时下载。

```bash
sh src/app/run_app.sh $HF_TOKEN $HF_CACHE_DIR
```

关于必选/可选模型与配置，见 [APP.md](src/app/APP.md)。

### 推理

使用 Hugging Face Token 运行推理脚本：

```bash
sh scripts/inference/inference.sh $HF_TOKEN $HF_CACHE_DIR
```

参数：
- `$HF_TOKEN`：Hugging Face 访问令牌（用于自动下载模型；若在 `configs/**/*.yaml` 中已指定本地路径，则可选）
- `$HF_CACHE_DIR`（可选）：自定义模型缓存目录（默认："~/.cache/huggingface/hub"）

## 📅 更新日志

- [X] **2025/07/03** - 论文与网页发布
- [X] **2025/08/26** - 发布检查点 v0x1561、应用与推理代码
- [ ] **TBD** - 测试与训练代码
- [ ] **TBD** - 更强模型版本

## 📝 引用

```bibtex
@article{li2025ic,
  title={IC-Custom: Diverse Image Customization via In-Context Learning},
  author={Li, Yaowei and Li, Xiaoyu and Zhang, Zhaoyang and Bian, Yuxuan and Liu, Gan and Li, Xinyuan and Xu, Jiale and Hu, Wenbo and Liu, Yating and Li, Lingen and others},
  journal={arXiv preprint arXiv:2507.01926},
  year={2025}
}
```

## 💖 致谢

我们感谢以下代码库：
- [flux](https://github.com/black-forest-labs/flux)
- [x-flux](https://github.com/XLabs-AI/x-flux/releases)

同时感谢 [Hugging Face](https://huggingface.co/) 提供的模型托管与 Spaces 部署支持。


## 限制
当前限制主要在推理速度与更灵活的指令跟随方面。我们将持续改进。如有建议，欢迎反馈。


## 📄 许可证

我们支持开源社区。完整细节见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

## 📬 联系方式

如有问题，欢迎通过 [email](mailto:liyaowei01@gmail.com) 联系我们。


## 🌟 Star 历史

<p align="center">
    <a href="https://star-history.com/#TencentARC/IC-Custom" target="_blank">
        <img width="500" src="https://api.star-history.com/svg?repos=TencentARC/IC-Custom&type=Date" alt="Star History Chart">
    </a>
</p>


