#### metainfo ####
head = r"""
<div style="text-align: center; margin: 1rem 0;">
    <h1 style="margin-bottom: 0.5rem;">
        <div style="font-size: 2.5rem; font-weight: 800;">
            🎨 <span style="background: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 50%, #34d399 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">IC-Custom</span>
        </div>
        <div style="font-size: 1.2rem; color: #4b5563; font-weight: 600; margin-top: 0.5rem;">
            Diverse Image Customization via In-Context Learning
        </div>
    </h1>
    <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;">
        <a href="https://liyaowei-stu.github.io/project/IC_Custom/" class="badge-link" style="text-decoration: none;">
            <img src='https://img.shields.io/badge/🔗_Project_Page-IC--Custom-34d399?style=for-the-badge&labelColor=1f2937' alt='Project Page'>
        </a>
        <a href="https://arxiv.org/abs/2507.01926" class="badge-link" style="text-decoration: none;">
            <img src='https://img.shields.io/badge/📄_Paper-arXiv-0ea5e9?style=for-the-badge&labelColor=1f2937' alt='Paper'>
        </a>
        <a href="https://github.com/TencentARC/IC-Custom" class="badge-link" style="text-decoration: none;">
            <img src='https://img.shields.io/badge/💻_Code-GitHub-6ee7b7?style=for-the-badge&labelColor=1f2937' alt='Code'>
        </a>
    </div>
</div>

<style>
.badge-link:hover {
    transform: translateY(-2px);
    opacity: 0.9;
}
</style>
"""

description = r"""
<div style="color: #1f2937; font-size: 1rem; line-height: 1.5; margin: 1rem 0; padding: 1rem; background: #f8fafc; border-radius: 8px; text-align: left;">
    <p style="margin-bottom: 0.8rem;">
        <span style="font-weight: 600; color: #0ea5e9;">🎨 What is IC-Custom?</span><br>
        IC-Custom supports various image customization scenarios, including position-aware IP insertion and position-free subject-driven generation.
    </p>
    <p style="margin-bottom: 0;">
        <span style="font-weight: 600; color: #34d399;">💡 Two Customization Modes:</span><br>
        <span style="margin-left: 1rem;">• <b>Position-aware:</b> Place content precisely in masked areas (precise or user-drawn mask)</span><br>
        <span style="margin-left: 1rem;">• <b>Position-free:</b> Global re-generation with subject preservation</span>
    </p>
</div>
"""

getting_started = r"""
<div style="padding: 1rem; background: #f8fafc; border-radius: 8px; margin: 0rem 0;">
    <div style="margin-bottom: 1.5rem;">
        <div style="color: #4b5563; margin-bottom: 0.5rem;">Initial Setup:</div>
        <ol style="margin: 0 0 0 1.2rem;">
            <li style="margin-bottom: 0.5rem;">Select a <b>Custmization Mode</b>: Position-aware Mode (reference insertion) or Position-free Mode (subject-driven generation)</li>
            <li style="margin-bottom: 0.5rem;">Upload a <b>reference image</b></li>
        </ol>
    </div>

    <div style="margin-bottom: 1.5rem;">
        <div style="color: #4b5563; font-weight: 600; margin-bottom: 0.5rem;">Position-aware Mode:</div>
        <ol start="3" style="margin: 0 0 0 1.2rem;">
            <li style="margin-bottom: 0.5rem;">Select an <b>input mask mode</b> to switch between precise mask and user-drawn mask</li>
            <li style="margin-bottom: 0.5rem;">Input <b>target image & mask</b>: precise mask (click for SAM prediction) or user-drawn mask (brush directly)</li>
            <li style="margin-bottom: 0.5rem;">
                Input <b>text prompt</b> (optional): describe the target scene<br>
                <span style="color: #64748b; font-size: 0.97em;">
                    (You can use "VLM Auto-generate" to generate a prompt automatically, <b>only available when enable_vlm_for_prompt=True</b>)
                </span>
            </li>
            <li style="margin-bottom: 0.5rem;">View or modify the <b>target mask</b> to ensure proper coverage</li>
            <li style="margin-bottom: 0.5rem;">Click "Submit" to generate the customized image</li>
        </ol>
    </div>

    <div>
        <div style="color: #4b5563; font-weight: 600; margin-bottom: 0.5rem;">Position-free Mode:</div>
        <ol start="3" style="margin: 0 0 0 1.2rem;">
            <li style="margin-bottom: 0.5rem;">
                Input <b>text prompt</b> (necessary): describe the target scene<br>
                <span style="color: #64748b; font-size: 0.97em;">
                    (If <b>enable_vlm_for_prompt=True</b>, you can use "VLM Auto-polish" to polish your prompt, or "VLM Auto-generate" to generate a prompt automatically)
                </span>
            </li>
            <li style="margin-bottom: 0.5rem;">
                Segment the reference image (optional): 
                <span>
                    use the segment tool in advanced options to segment the reference image foreground 
                    <b>only when <code>--enable_ben2_for_mask_ref=True</code></b>
                </span>
            </li>
            <li style="margin-bottom: 0.5rem;">Click "Submit" to generate the customized image</li>
        </ol>
    </div>

    <div style="margin-top: 1.5rem;">
        <div style="color: #4b5563; font-weight: 600; margin-bottom: 0.5rem;">Tips:</div>
        <ul style="margin: 0 0 0 1.2rem;">
            <li style="margin-bottom: 0.5rem;">You can zoom out the page (Alt + "-" or ⌘ + "-") until it fits comfortably for operation</li>
        </ul>
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