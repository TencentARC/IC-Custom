#### metainfo ####
head = r"""
<div class="elegant-header">
    <div class="header-content">
        <!-- Main title -->
        <h1 class="main-title">
            <span class="title-icon">🎨</span>
            <span class="title-text">IC-Custom</span>
        </h1>
        
        <!-- Subtitle -->
        <p class="subtitle">Transform your images with AI-powered customization</p>
        
        <!-- Action badges -->
        <div class="header-badges">
            <a href="https://liyaowei-stu.github.io/project/IC_Custom/" class="badge-link">
                <span class="badge-icon">🔗</span>
                <span class="badge-text">Project</span>
            </a>
            <a href="https://arxiv.org/abs/2507.01926" class="badge-link">
                <span class="badge-icon">📄</span>
                <span class="badge-text">Paper</span>
            </a>
            <a href="https://github.com/TencentARC/IC-Custom" class="badge-link">
                <span class="badge-icon">💻</span>
                <span class="badge-text">Code</span>
            </a>
        </div>
    </div>
</div>
"""


getting_started = r"""
<div class="getting-started-container">
    <!-- Header -->
    <div class="guide-header">
        <h3 class="guide-title">🚀 Quick Start Guide</h3>
        <p class="guide-subtitle">Follow these steps to customize your images with IC-Custom</p>
    </div>

    <!-- What is IC-Custom -->
    <div class="info-card">
        <div class="info-content">
            <strong class="brand-name">IC-Custom</strong> offers two customization modes:
            <span class="mode-badge position-aware">Position-aware</span>
            (precise placement in masked areas) and
            <span class="mode-badge position-free">Position-free</span>
            (subject-driven generation).
        </div>
    </div>

    <!-- Common Steps -->
    <div class="step-card common-steps">
        <div class="step-header">
            <span class="step-number">1</span>
            Initial Setup (Both Modes)
        </div>
        <ul class="step-list">
            <li>Choose your <strong>customization mode</strong></li>
            <li>Upload a <strong>reference image</strong> 📸</li>
        </ul>
    </div>

    <!-- Position-aware Mode -->
    <div class="step-card position-aware-steps">
        <div class="step-header">
            <span class="step-number">2A</span>
            🎯 Position-aware Mode Steps
        </div>
        <ul class="step-list">
            <li>Select <strong>input mask mode</strong> (precise mask or user-drawn mask)</li>
            <li>Upload <strong>target image</strong> and create mask (click for SAM or brush directly)</li>
            <li>Add <strong>text prompt</strong> (optional) - use VLM buttons for auto-generation</li>
            <li>Review and refine your <strong>mask</strong> using mask tools if needed</li>
            <li>Click <span class="run-button position-aware">Run</span> ✨</li>
        </ul>
    </div>

    <!-- Position-free Mode -->
    <div class="step-card position-free-steps">
        <div class="step-header">
            <span class="step-number">2B</span>
            🎨 Position-free Mode Steps
        </div>
        <ul class="step-list">
            <li>Write your <strong>text prompt</strong> (required) - describe the target scene</li>
            <li>Use VLM buttons for prompt auto-generation or polishing (if enabled)</li>
            <li>Click <span class="run-button position-free">Run</span> ✨</li>
        </ul>
    </div>

    <!-- Quick Tips -->
    <div class="tips-card">
        <div class="tips-content">
            <strong>💡 Quick Tips:</strong> 
            Use <kbd class="key-hint">Alt + "-"</kbd> or <kbd class="key-hint">⌘ + "-"</kbd> to zoom out for better operation • 
            Adjust settings in <kbd class="key-hint">Advanced Options</kbd> • Use mask operations (<kbd class="key-hint">dilate</kbd>/<kbd class="key-hint">erode</kbd>/<kbd class="key-hint">bbox</kbd>) for better results • 
            Try different <kbd class="key-hint">seeds</kbd> for varied outputs
        </div>
    </div>

    <!-- Final Message -->
    <div class="final-message">
        <div class="final-text">
            🎉 Ready to start? Collapse this guide and begin customizing!
        </div>
    </div>
</div>
"""


citation = r"""
If IC-Custom is helpful, please help to ⭐ the <a href='https://github.com/TencentARC/IC-Custom' target='_blank'>Github Repo</a>. Thanks!
[![GitHub Stars](https://img.shields.io/github/stars/TencentARC/IC-Custom?style=social)](https://github.com/TencentARC/IC-Custom)
---
📝 **Citation**
<br>
If our work is useful for your research, please consider citing:
```bibtex
@article{li2025ic,
  title={IC-Custom: Diverse Image Customization via In-Context Learning},
  author={Li, Yaowei and Li, Xiaoyu and Zhang, Zhaoyang and Bian, Yuxuan and Liu, Gan and Li, Xinyuan and Xu, Jiale and Hu, Wenbo and Liu, Yating and Li, Lingen and others},
  journal={arXiv preprint arXiv:2507.01926},
  year={2025}
}
```
📧 **Contact**
<br>
If you have any questions, please feel free to reach me out at <b>liyaowei@gmail.com</b>.
"""