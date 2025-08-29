#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Centralized CSS and JS for the IC-Custom app UI.

Expose helpers:
- get_css(): return a single CSS string for gradio Blocks(css=...)
- get_js(): return an JS for gradio.
"""


def get_css() -> str:
    return r"""
    /* Global Optimization Effects - No Layout Changes */
    
    /* Apple-style segmented control for radio buttons */
    #customization_mode_radio .wrap, #input_mask_mode_radio .wrap, #seg_ref_mode_radio .wrap {
        display: flex;
        flex-wrap: nowrap;
        justify-content: center;
        align-items: center;
        gap: 0;
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid var(--neutral-200);
        border-radius: 10px;
        padding: 3px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
    }

    #customization_mode_radio .wrap label, #input_mask_mode_radio .wrap label, #seg_ref_mode_radio .wrap label {
        display: flex;
        flex: 1;
        justify-content: center;
        align-items: center;
        margin: 0;
        padding: 10px 16px;
        box-sizing: border-box;
        border-radius: 7px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        background: transparent;
        border: none;
        font-weight: 500;
        font-size: 0.9rem;
        color: var(--text-secondary);
        cursor: pointer;
        position: relative;
        white-space: nowrap;
        min-width: 0;
    }

    /* Hide the actual radio input */
    #customization_mode_radio .wrap label input[type="radio"],
    #input_mask_mode_radio .wrap label input[type="radio"],
    #seg_ref_mode_radio .wrap label input[type="radio"] {
        display: none;
    }

    /* Hover states */
    #customization_mode_radio .wrap label:hover, 
    #input_mask_mode_radio .wrap label:hover, 
    #seg_ref_mode_radio .wrap label:hover {
        background: rgba(14, 165, 233, 0.1);
        color: var(--primary-blue);
    }

    /* Selected state with smooth background */
    #customization_mode_radio .wrap label:has(input[type="radio"]:checked),
    #input_mask_mode_radio .wrap label:has(input[type="radio"]:checked),
    #seg_ref_mode_radio .wrap label:has(input[type="radio"]:checked) {
        background: var(--primary-blue);
        color: white;
        font-weight: 600;
        box-shadow: 0 2px 6px rgba(14, 165, 233, 0.25);
        transform: none;
    }

    /* Fallback for browsers that don't support :has() */
    #customization_mode_radio .wrap label input[type="radio"]:checked + *,
    #input_mask_mode_radio .wrap label input[type="radio"]:checked + *,
    #seg_ref_mode_radio .wrap label input[type="radio"]:checked + * {
        color: white;
    }

    #customization_mode_radio .wrap:has(input[type="radio"]:checked) label:has(input[type="radio"]:checked),
    #input_mask_mode_radio .wrap:has(input[type="radio"]:checked) label:has(input[type="radio"]:checked),
    #seg_ref_mode_radio .wrap:has(input[type="radio"]:checked) label:has(input[type="radio"]:checked) {
        background: var(--primary-blue);
    }

    /* Active state */
    #customization_mode_radio .wrap label:active, 
    #input_mask_mode_radio .wrap label:active, 
    #seg_ref_mode_radio .wrap label:active {
        transform: scale(0.98);
    }

    /* Elegant header styling */
    .elegant-header {
        text-align: center;
        margin: 0 0 2rem 0;
        padding: 0;
    }

    .header-content {
        display: inline-block;
        padding: 1.8rem 2.5rem;
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.1) 0%, 
            rgba(255, 255, 255, 0.05) 100%);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 
            0 8px 32px rgba(15, 23, 42, 0.04),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }

    .header-content::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.1), 
            transparent);
        transition: left 0.6s ease;
    }

    .header-content:hover::before {
        left: 100%;
    }

    .header-content:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 12px 40px rgba(15, 23, 42, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        border-color: rgba(14, 165, 233, 0.2);
    }

    /* Main title styling */
    .main-title {
        margin: 0 0 0.8rem 0;
        font-size: 2.4rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }

    .title-icon {
        font-size: 2.2rem;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1));
    }

    .title-text {
        background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 50%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: none;
        position: relative;
    }

    /* Subtitle styling */
    .subtitle {
        margin: 0 0 1.2rem 0;
        font-size: 1rem;
        color: #64748b;
        font-weight: 500;
        letter-spacing: 0.025em;
        opacity: 0.9;
    }

    /* Header badges container */
    .header-badges {
        display: flex;
        justify-content: center;
        gap: 0.8rem;
        flex-wrap: wrap;
    }

    /* Individual badge links */
    .badge-link {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.5rem 1rem;
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: #475569;
        text-decoration: none;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        position: relative;
        overflow: hidden;
    }

    .badge-link::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: var(--primary-gradient);
        transition: left 0.3s ease;
        z-index: -1;
    }

    .badge-link:hover::before {
        left: 0;
    }

    .badge-link:hover {
        transform: translateY(-2px);
        color: white;
        border-color: transparent;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
    }

    .badge-icon {
        font-size: 1rem;
        opacity: 0.8;
    }

    .badge-text {
        font-weight: 600;
    }

    /* Getting Started Guide Styling */
    .getting-started-container {
        padding: 1.5rem;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
    }

    .guide-header {
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .guide-title {
        color: var(--text-primary);
        margin: 0 0 0.5rem 0;
        font-size: 1.2rem;
        font-weight: 700;
    }

    .guide-subtitle {
        color: var(--text-muted);
        margin: 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* Info card */
    .info-card {
        background: rgba(255, 255, 255, 0.4);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1.2rem;
        border-left: 3px solid var(--primary-blue);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        transition: all 0.3s ease;
    }

    .info-card:hover {
        background: rgba(255, 255, 255, 0.5);
        transform: translateX(2px);
    }

    .info-content {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .brand-name {
        color: var(--primary-blue);
        font-weight: 700;
    }

    /* Mode badges */
    .mode-badge {
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0 0.2rem;
        transition: all 0.2s ease;
    }

    .mode-badge.position-aware {
        background: var(--badge-blue-bg);
        color: var(--badge-blue-text);
    }

    .mode-badge.position-free {
        background: var(--badge-green-bg);
        color: var(--badge-green-text);
    }

    /* Step cards */
    .step-card {
        background: rgba(255, 255, 255, 0.4);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .step-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 3px;
        transition: all 0.3s ease;
    }

    .step-card.common-steps::before {
        background: var(--neutral-500);
    }

    .step-card.position-aware-steps::before {
        background: var(--position-aware-blue);
    }

    .step-card.position-free-steps::before {
        background: var(--position-free-purple);
    }

    .step-card:hover {
        background: rgba(255, 255, 255, 0.5);
        transform: translateX(2px);
    }

    .step-header {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        font-size: 0.95rem;
    }

    .step-number {
        color: white;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 0.5rem;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .common-steps .step-number {
        background: var(--neutral-500);
    }

    .position-aware-steps .step-number {
        background: var(--position-aware-blue);
    }

    .position-free-steps .step-number {
        background: var(--position-free-purple);
    }

    .step-list {
        margin: 0;
        padding-left: 1.2rem;
        font-size: 0.85rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }

    .step-list li {
        margin-bottom: 0.4rem;
        position: relative;
    }

    .step-list li:last-child {
        margin-bottom: 0;
    }

    /* Run buttons */
    .run-button {
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
        color: white;
        transition: all 0.2s ease;
    }

    .run-button.position-aware {
        background: var(--position-aware-blue);
    }

    .run-button.position-free {
        background: var(--position-free-purple);
    }

    /* Tips card */
    .tips-card {
        background: rgba(241, 245, 249, 0.6);
        border-radius: 8px;
        padding: 0.8rem;
        border-left: 3px solid var(--neutral-400);
        margin-bottom: 1rem;
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        transition: all 0.3s ease;
    }

    .tips-card:hover {
        background: rgba(241, 245, 249, 0.8);
        transform: translateX(2px);
    }

    .tips-content {
        font-size: 0.8rem;
        color: var(--text-tips);
        line-height: 1.5;
    }

    /* Key hints */
    .key-hint {
        background: var(--kbd-bg);
        color: var(--kbd-text);
        padding: 0.1rem 0.3rem;
        border-radius: 3px;
        font-size: 0.75em;
        border: 1px solid var(--kbd-border);
        font-family: monospace;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .key-hint:hover {
        background: var(--primary-blue);
        color: white;
        border-color: var(--primary-blue);
    }

    /* Final message */
    .final-message {
        padding: 0.8rem;
        background: var(--bg-final);
        border-radius: 8px;
        text-align: center;
        transition: all 0.3s ease;
    }

    .final-message:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.1);
    }

    .final-text {
        color: var(--text-final);
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Legacy header badge styling for backward compatibility */
    .header-badge {
        background: var(--primary-gradient);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        box-shadow: 0 1px 3px rgba(14, 165, 233, 0.2);
        display: inline-block;
    }

    .header-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(14, 165, 233, 0.3);
        text-decoration: none;
    }

    /* Accordion styling matching getting_started */
    .gradio-accordion {
        border: 1px solid rgba(14, 165, 233, 0.2);
        border-radius: 8px;
        overflow: visible !important; /* Allow dropdown to overflow */
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%);
    }
    
    /* Ensure accordion content area allows dropdown overflow */
    .gradio-accordion .wrap {
        overflow: visible !important;
    }

    .gradio-accordion > .label-wrap {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%);
        border-bottom: 1px solid rgba(14, 165, 233, 0.2);
        padding: 1rem 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    /* Minimal dropdown styling - let Gradio handle positioning naturally */
    #aspect_ratio_dropdown {
        border-radius: 8px;
    }
    
    /* COMPLETELY REMOVE all dropdown styling - let Gradio handle everything */
    /* This was causing the dropdown to display as a text block instead of options */
    
    /* DO NOT style .gradio-dropdown globally - causes functionality issues */

    /* Slider styling matching theme */
    .gradio-slider {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%) !important;
        border: 1px solid rgba(14, 165, 233, 0.2) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1) !important;
        padding: 12px !important;
    }

    .gradio-slider:hover {
        border-color: rgba(14, 165, 233, 0.3) !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.15) !important;
    }

    /* Slider input styling */
    .gradio-slider input[type="range"] {
        background: transparent !important;
    }

    .gradio-slider input[type="range"]::-webkit-slider-track {
        background: rgba(14, 165, 233, 0.2) !important;
        border-radius: 4px !important;
    }

    .gradio-slider input[type="range"]::-webkit-slider-thumb {
        background: var(--primary-blue) !important;
        border: 2px solid white !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.3) !important;
    }

    /* Checkbox styling matching theme */
    .gradio-checkbox {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%) !important;
        border: 1px solid rgba(14, 165, 233, 0.2) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1) !important;
        padding: 8px 12px !important;
    }

    /* Specific styling for identified components */
    #aspect_ratio_dropdown,
    #text_prompt,
    #move_to_center_checkbox,
    #use_bg_preservation_checkbox {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%) !important;
        border: 1px solid rgba(14, 165, 233, 0.2) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1) !important;
    }

    /* Removed specific aspect_ratio_dropdown styling to avoid conflicts */

    #aspect_ratio_dropdown:hover,
    #text_prompt:hover,
    #move_to_center_checkbox:hover,
    #use_bg_preservation_checkbox:hover {
        border-color: rgba(14, 165, 233, 0.3) !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.15) !important;
    }

    /* Textbox specific styling */
    #text_prompt textarea {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%) !important;
        border: 1px solid rgba(14, 165, 233, 0.2) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1) !important;
    }

    /* Color variables matching getting_started section exactly */
    :root {
        /* Primary colors from getting_started */
        --primary-blue: #0ea5e9;
        --primary-blue-secondary: #06b6d4;
        --primary-green: #10b981;
        --primary-gradient: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 50%, #10b981 100%);
        
        /* Mode-specific colors from getting_started */
        --position-aware-blue: #3b82f6;
        --position-free-purple: #8b5cf6;
        
        /* Badge colors from getting_started */
        --badge-blue-bg: #dbeafe;
        --badge-blue-text: #1e40af;
        --badge-green-bg: #dcfce7;
        --badge-green-text: #166534;
        
        /* Neutral colors from getting_started */
        --neutral-50: #f8fafc;
        --neutral-100: #f1f5f9;
        --neutral-200: #e2e8f0;
        --neutral-300: #cbd5e1;
        --neutral-400: #94a3b8;
        --neutral-500: #64748b;
        --neutral-600: #475569;
        --neutral-700: #334155;
        --neutral-800: #1e293b;
        
        /* Text colors from getting_started */
        --text-primary: #1e293b;
        --text-secondary: #4b5563;
        --text-muted: #64748b;
        --text-tips: #475569;
        --text-final: #0c4a6e;
        
        /* Background colors from getting_started */
        --bg-primary: white;
        --bg-secondary: #f8fafc;
        --bg-tips: #f1f5f9;
        --bg-final: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        
        /* Keyboard hint styles from getting_started */
        --kbd-bg: #e2e8f0;
        --kbd-text: #475569;
        --kbd-border: #cbd5e1;
    }

    /* Global smooth transitions - exclude dropdowns */
    *:not(.gradio-dropdown):not(.gradio-dropdown *) {
        transition: all 0.2s ease;
    }

    /* Focus states using getting_started primary blue - exclude dropdowns */
    button:focus,
    input:not(.gradio-dropdown input):focus,
    select:not(.gradio-dropdown select):focus,
    textarea:not(.gradio-dropdown textarea):focus {
        outline: none;
        box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.3);
    }

    /* Subtle hover effects for interactive elements - exclude dropdowns */
    button:not(.gradio-dropdown button):hover {
        transform: translateY(-1px);
    }

    /* Global text styling matching getting_started */
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        line-height: 1.6;
        color: var(--text-secondary);
        background-color: var(--bg-secondary);
    }

    /* Enhanced form element styling - exclude dropdowns from global styling */
    input:not(.gradio-dropdown input), 
    textarea:not(.gradio-dropdown textarea), 
    select:not(.gradio-dropdown select) {
        border-radius: 8px;
        border: 1px solid rgba(14, 165, 233, 0.2);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        color: var(--text-primary);
        transition: all 0.3s ease;
        padding: 12px 16px;
        font-size: 0.95rem;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1);
    }

    input:not(.gradio-dropdown input):focus, 
    textarea:not(.gradio-dropdown textarea):focus, 
    select:not(.gradio-dropdown select):focus {
        border-color: var(--primary-blue);
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1), 0 2px 8px rgba(14, 165, 233, 0.15);
        background: linear-gradient(135deg, rgba(255, 255, 255, 1) 0%, rgba(240, 249, 255, 0.98) 100%);
        outline: none;
        transform: translateY(-1px);
    }

    input:not(.gradio-dropdown input):hover, 
    textarea:not(.gradio-dropdown textarea):hover, 
    select:not(.gradio-dropdown select):hover {
        border-color: rgba(14, 165, 233, 0.3);
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.12);
    }

    /* Textbox specific styling */
    .gradio-textbox {
        border-radius: 12px;
        overflow: hidden;
    }

    .gradio-textbox textarea {
        border-radius: 12px;
        resize: vertical;
        min-height: 44px;
    }

    /* Scrollbar styling matching getting_started */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--neutral-100);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--neutral-400);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-blue);
    }

    /* Enhanced button styling with Apple-style refinement */
    button {
        border-radius: 8px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 1px solid var(--neutral-200);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }

    /* Button hover glow effect */
    button::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: radial-gradient(circle, rgba(14, 165, 233, 0.1) 0%, transparent 70%);
        transition: all 0.4s ease;
        transform: translate(-50%, -50%);
        pointer-events: none;
    }

    button:hover::after {
        width: 200px;
        height: 200px;
    }

    /* Primary button using unified primary blue */
    button[variant="primary"] {
        background: var(--primary-blue);
        border-color: var(--primary-blue);
        color: white;
        box-shadow: 0 2px 6px rgba(14, 165, 233, 0.2);
    }

    button[variant="primary"]:hover {
        background: #0284c7;
        border-color: #0284c7;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
        transform: translateY(-1px);
    }

    /* Secondary buttons */
    button[variant="secondary"], .secondary-button {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%);
        border: 1px solid rgba(14, 165, 233, 0.2);
        color: var(--text-secondary);
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1);
    }

    button[variant="secondary"]:hover, .secondary-button:hover {
        background: var(--primary-blue);
        border-color: var(--primary-blue);
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.25);
    }

    /* VLM buttons with subtle, elegant styling */
    #vlm_generate_btn, #vlm_polish_btn {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%) !important;
        border: 1px solid rgba(14, 165, 233, 0.2) !important;
        color: var(--text-secondary) !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1);
        font-weight: 500;
        border-radius: 8px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }

    #vlm_generate_btn::before, #vlm_polish_btn::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(14, 165, 233, 0.1), transparent);
        transition: left 0.5s ease;
    }

    #vlm_generate_btn:hover::before, #vlm_polish_btn:hover::before {
        left: 100%;
    }

    #vlm_generate_btn:hover, #vlm_polish_btn:hover {
        background: var(--primary-blue) !important;
        border-color: var(--primary-blue) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.25);
        transform: translateY(-1px);
    }

    #vlm_generate_btn:active, #vlm_polish_btn:active {
        transform: translateY(0px);
        box-shadow: 0 2px 6px rgba(14, 165, 233, 0.2);
    }

    /* Enhanced image styling with fixed dimensions for consistency */
    .gradio-image, .gradio-imageeditor {
        height: 300px !important;
        width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    .gradio-image img,
    .gradio-imageeditor img {
        height: 300px !important;
        width: 100% !important;
        object-fit: contain !important;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.1);
    }

    .gradio-image img:hover,
    .gradio-imageeditor img:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.15);
    }

    /* Gallery CSS - contained adaptive layout with theme colors */
    #mask_gallery, #result_gallery, .custom-gallery {
        overflow: visible !important; /* Allow progress indicator to show */
        position: relative !important;
        width: 100% !important;
        height: auto !important;
        max-height: 75vh !important;
        min-height: 300px !important;
        display: flex !important;
        flex-direction: column !important;
        padding: 6px !important;
        margin: 0 !important;
        border-radius: 12px !important;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%) !important;
        border: 1px solid rgba(14, 165, 233, 0.2) !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1) !important;
    }

    /* Gallery containers with contained but flexible display */
    #mask_gallery .gradio-gallery, #result_gallery .gradio-gallery {
        width: 100% !important;
        height: auto !important;
        min-height: 280px !important;
        max-height: 70vh !important;
        padding: 8px !important;
        overflow: auto !important;
        display: flex !important;
        flex-direction: column !important;
        border-radius: 8px !important;
    }

    /* Only hide specific duplicate elements that cause the problem */
    #mask_gallery > div > div:nth-child(n+2),
    #result_gallery > div > div:nth-child(n+2) {
        display: none !important;
    }

    /* Alternative: hide duplicate grid structures only */
    #mask_gallery .gradio-gallery:nth-child(n+2),
    #result_gallery .gradio-gallery:nth-child(n+2) {
        display: none !important;
    }

    /* Ensure timing and status elements are NOT hidden by the above rules */
    #result_gallery .status,
    #result_gallery .timer,
    #result_gallery [class*="time"],
    #result_gallery [class*="status"],
    #result_gallery [class*="duration"],
    #result_gallery .gradio-status,
    #result_gallery .gradio-timer,
    #result_gallery .gradio-info,
    #result_gallery [data-testid*="timer"],
    #result_gallery [data-testid*="status"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: relative !important;
        z-index: 1000 !important;
    }

    /* Gallery images - contained adaptive display */
    #mask_gallery img, #result_gallery img {
        width: 100% !important;
        height: auto !important;
        max-width: 100% !important;
        max-height: 60vh !important;
        object-fit: contain !important;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.1);
        display: block !important;
        margin: 0 auto !important;
    }

    /* Main preview image styling - contained but responsive */
    #mask_gallery .preview-image, #result_gallery .preview-image {
        width: 100% !important;
        height: auto !important;
        max-width: 100% !important;
        max-height: 55vh !important;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.15);
        object-fit: contain !important;
        display: block !important;
        margin: 0 auto !important;
    }

    /* Gallery content wrappers - ensure no height constraints */
    #mask_gallery .gradio-gallery > div,
    #result_gallery .gradio-gallery > div {
        width: 100% !important;
        height: auto !important;
        min-height: auto !important;
        max-height: none !important;
        overflow: visible !important;
    }

    /* Gallery image containers - remove any height limits */
    #mask_gallery .image-container,
    #result_gallery .image-container,
    #mask_gallery [data-testid="image"],
    #result_gallery [data-testid="image"] {
        width: 100% !important;
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
    }

    /* Controlled gallery wrapper elements */
    #mask_gallery .image-wrapper,
    #result_gallery .image-wrapper {
        max-height: 60vh !important;
        overflow: hidden !important;
    }

    /* Specific targeting for Gradio's internal gallery elements */
    #mask_gallery .grid-wrap,
    #result_gallery .grid-wrap,
    #mask_gallery .preview-wrap,
    #result_gallery .preview-wrap {
        height: auto !important;
        max-height: 65vh !important;
        overflow: auto !important;
        border-radius: 8px !important;
    }

    /* Ensure gallery grids are properly sized within container */
    #mask_gallery .grid,
    #result_gallery .grid {
        height: auto !important;
        max-height: 60vh !important;
        display: grid !important;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)) !important;
        gap: 8px !important;
        align-items: start !important;
        overflow: auto !important;
        padding: 4px !important;
    }

    /* Custom scrollbar for gallery */
    #mask_gallery .gradio-gallery::-webkit-scrollbar,
    #result_gallery .gradio-gallery::-webkit-scrollbar,
    #mask_gallery .grid::-webkit-scrollbar,
    #result_gallery .grid::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    #mask_gallery .gradio-gallery::-webkit-scrollbar-track,
    #result_gallery .gradio-gallery::-webkit-scrollbar-track,
    #mask_gallery .grid::-webkit-scrollbar-track,
    #result_gallery .grid::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.1);
        border-radius: 3px;
    }

    #mask_gallery .gradio-gallery::-webkit-scrollbar-thumb,
    #result_gallery .gradio-gallery::-webkit-scrollbar-thumb,
    #mask_gallery .grid::-webkit-scrollbar-thumb,
    #result_gallery .grid::-webkit-scrollbar-thumb {
        background: var(--neutral-400);
        border-radius: 3px;
    }

    #mask_gallery .gradio-gallery::-webkit-scrollbar-thumb:hover,
    #result_gallery .gradio-gallery::-webkit-scrollbar-thumb:hover,
    #mask_gallery .grid::-webkit-scrollbar-thumb:hover,
    #result_gallery .grid::-webkit-scrollbar-thumb:hover {
        background: var(--primary-blue);
    }

    /* Thumbnail navigation styling in preview mode */
    #mask_gallery .thumbnail, #result_gallery .thumbnail {
        opacity: 0.7;
        transition: opacity 0.3s ease;
        border-radius: 6px;
    }

    #mask_gallery .thumbnail:hover, #result_gallery .thumbnail:hover,
    #mask_gallery .thumbnail.selected, #result_gallery .thumbnail.selected {
        opacity: 1;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.3);
    }

    /* Improved layout spacing and organization */
    #glass_card .gradio-row {
        gap: 16px !important;
        margin-bottom: 6px !important;
    }

    #glass_card .gradio-column {
        gap: 10px !important;
    }

    /* Better section spacing with theme colors */
    #glass_card .gradio-group {
        margin-bottom: 6px !important;
        padding: 10px !important;
        border-radius: 8px !important;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(240, 249, 255, 0.9) 100%) !important;
        border: 1px solid rgba(14, 165, 233, 0.15) !important;
        box-shadow: 0 1px 3px rgba(14, 165, 233, 0.05) !important;
        transition: all 0.3s ease !important;
        overflow: visible !important; /* Allow dropdown to overflow */
    }

    #glass_card .gradio-group:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%) !important;
        border-color: rgba(14, 165, 233, 0.25) !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.12) !important;
        /* transform removed to prevent layout shift that hides dropdown */
    }

    /* Enhanced button styling for improved UX */
    button[variant="secondary"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%) !important;
        border: 1px solid rgba(14, 165, 233, 0.2) !important;
        color: var(--text-secondary) !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1) !important;
        transition: all 0.3s ease !important;
    }

    button[variant="secondary"]:hover {
        background: var(--primary-blue) !important;
        border-color: var(--primary-blue) !important;
        color: white !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.25) !important;
    }

    /* Markdown header improvements */
    .gradio-markdown h1, .gradio-markdown h2, .gradio-markdown h3 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
        margin-top: 4px !important;
    }

    /* Radio button container improvements */
    #customization_mode_radio, #input_mask_mode_radio, #seg_ref_mode_radio {
        margin-bottom: 0px !important;
        margin-top: 0px !important;
    }

    /* Reduce space between markdown headers and subsequent components */
    .gradio-markdown + .gradio-group {
        margin-top: 1px !important;
    }

    .gradio-markdown + .gradio-image,
    .gradio-markdown + .gradio-imageeditor,
    .gradio-markdown + .gradio-textbox,
    .gradio-markdown + .gradio-gallery {
        margin-top: 1px !important;
    }

    /* Specific spacing adjustments for numbered sections */
    .gradio-markdown:has(h1), .gradio-markdown:has(h2), .gradio-markdown:has(h3) {
        margin-bottom: 2px !important;
    }

    /* Remove padding from image and gallery containers */
    .gradio-image, .gradio-imageeditor, .gradio-gallery {
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Image container styling with theme colors */
    .gradio-image, .gradio-imageeditor {
        border-radius: 12px;
        overflow: hidden;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%) !important;
        border: 1px solid rgba(14, 165, 233, 0.2) !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1) !important;
        transition: all 0.3s ease;
    }

    .gradio-image:hover, .gradio-imageeditor:hover {
        border-color: var(--primary-blue) !important;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.15) !important;
        transform: translateY(-1px);
    }

    /* Image upload area styling */
    .gradio-image .upload-container,
    .gradio-imageeditor .upload-container,
    .gradio-image > div,
    .gradio-imageeditor > div {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%) !important;
        border: 1px solid rgba(14, 165, 233, 0.2) !important;
        border-radius: 12px !important;
    }

    /* Image upload placeholder styling */
    .gradio-image .upload-text,
    .gradio-imageeditor .upload-text,
    .gradio-image [data-testid="upload-text"],
    .gradio-imageeditor [data-testid="upload-text"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%) !important;
        color: var(--text-secondary) !important;
    }

    /* Image preview area */
    .gradio-image .image-container,
    .gradio-imageeditor .image-container,
    .gradio-image .preview-container,
    .gradio-imageeditor .preview-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%) !important;
        border-radius: 12px !important;
    }

    /* Specific targeting for image upload areas */
    .gradio-image .wrap,
    .gradio-imageeditor .wrap,
    .gradio-image .block,
    .gradio-imageeditor .block {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%) !important;
        border: 1px solid rgba(14, 165, 233, 0.2) !important;
        border-radius: 12px !important;
    }

    /* Image drop zone styling */
    .gradio-image .drop-zone,
    .gradio-imageeditor .drop-zone,
    .gradio-image .upload-area,
    .gradio-imageeditor .upload-area {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%) !important;
        border: 2px dashed rgba(14, 165, 233, 0.3) !important;
        border-radius: 12px !important;
    }

    /* Force override any white backgrounds in image components */
    .gradio-image *,
    .gradio-imageeditor * {
        background-color: transparent !important;
    }

    .gradio-image .gradio-image,
    .gradio-imageeditor .gradio-imageeditor {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%) !important;
    }

    /* Specific styling for Reference Image and Target Images */
    #reference_image,
    #target_image_1,
    #target_image_2 {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%) !important;
        border: 1px solid rgba(14, 165, 233, 0.2) !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1) !important;
        border-radius: 12px !important;
    }

    #reference_image *,
    #target_image_1 *,
    #target_image_2 * {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%) !important;
        border-radius: 12px !important;
    }

    /* Upload area for specific image components */
    #reference_image .upload-container,
    #target_image_1 .upload-container,
    #target_image_2 .upload-container,
    #reference_image .drop-zone,
    #target_image_1 .drop-zone,
    #target_image_2 .drop-zone {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%) !important;
        border: 2px dashed rgba(14, 165, 233, 0.3) !important;
        border-radius: 12px !important;
    }

    /* Hover effects for specific image components */
    #reference_image:hover,
    #target_image_1:hover,
    #target_image_2:hover {
        border-color: var(--primary-blue) !important;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.15) !important;
        transform: translateY(-1px);
    }

    /* Group styling matching getting_started white cards */
    .group, .gradio-group {
        border-radius: 8px;
        background: var(--bg-primary);
        border: 1px solid var(--neutral-200);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    /* Subtle page background with theme colors */
    body, .gradio-container {
        background: linear-gradient(135deg, #f8fafc 0%, #f0f9ff 50%, #f8fafc 100%);
        min-height: 100vh;
    }

    /* Global glass container with subtle Apple-style gradient */
    #global_glass_container {
        position: relative;
        border-radius: 20px;
        padding: 16px;
        margin: 12px auto;
        max-width: 1400px;
        background: linear-gradient(145deg, 
            rgba(248, 250, 252, 0.98), 
            rgba(241, 245, 249, 0.95));
        box-shadow: 
            0 20px 40px rgba(15, 23, 42, 0.08),
            0 8px 24px rgba(15, 23, 42, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(226, 232, 240, 0.7);
        transition: all 0.3s ease;
        overflow: visible !important; /* Allow dropdown to overflow */
    }

    /* Subtle gradient overlay for Apple effect */
    #global_glass_container::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 250px;
        background: linear-gradient(135deg, 
            rgba(14, 165, 233, 0.08) 0%,
            rgba(6, 182, 212, 0.06) 25%,
            rgba(16, 185, 129, 0.08) 50%,
            rgba(139, 92, 246, 0.06) 75%,
            rgba(14, 165, 233, 0.08) 100%);
        background-size: 300% 300%;
        animation: subtleGradientShift 15s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes subtleGradientShift {
        0%, 100% { 
            background-position: 0% 50%;
            opacity: 0.8;
        }
        50% { 
            background-position: 100% 50%;
            opacity: 1;
        }
    }

    /* Ensure content is above the gradient overlay */
    #global_glass_container > * {
        position: relative;
        z-index: 1;
    }

    /* Hover effect for global container - transform disabled to avoid dropdown reposition */
    #global_glass_container:hover {
        /* transform: translateY(-2px); */
        box-shadow: 
            0 25px 50px rgba(15, 23, 42, 0.08),
            0 12px 30px rgba(15, 23, 42, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
        border-color: rgba(226, 232, 240, 0.8);
    }

    /* Subtle border highlight for global container */
    #global_glass_container::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 20px;
        padding: 1px;
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.8), 
            rgba(226, 232, 240, 0.4), 
            rgba(255, 255, 255, 0.6),
            rgba(226, 232, 240, 0.3)
        );
        -webkit-mask: 
            linear-gradient(#fff 0 0) content-box, 
            linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        pointer-events: none;
        z-index: 0;
    }

    /* Inner glassmorphism container with theme colors */
    #glass_card {
        position: relative;
        border-radius: 16px;
        padding: 16px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.6) 0%, rgba(240, 249, 255, 0.5) 100%);
        box-shadow: 
            0 8px 24px rgba(14, 165, 233, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(14, 165, 233, 0.2);
        margin-bottom: 12px;
        transition: all 0.3s ease;
        overflow: visible !important; /* Allow dropdown to overflow */
    }

    #glass_card:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.7) 0%, rgba(240, 249, 255, 0.6) 100%);
        border-color: rgba(14, 165, 233, 0.3);
        box-shadow: 
            0 12px 32px rgba(14, 165, 233, 0.12),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
    }

    /* Subtle inner border gradient for liquid glass feel */
    #glass_card::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 16px;
        padding: 1px;
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.6), 
            rgba(226, 232, 240, 0.2));
        -webkit-mask: 
            linear-gradient(#fff 0 0) content-box, 
            linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        pointer-events: none;
    }

    /* Preserve the airy layout inside the cards */
    #global_glass_container .gradio-column { gap: 12px; }
    #glass_card .gradio-row { gap: 16px; }
    #glass_card .gradio-column { gap: 12px; }
    #glass_card .gradio-group { margin: 8px 0; }

    /* Text selection matching getting_started colors */
    ::selection {
        background: var(--badge-blue-bg);
        color: var(--badge-blue-text);
    }

    /* Placeholder styling */
    ::placeholder {
        color: var(--text-muted);
        opacity: 0.8;
    }

    /* Improved error state styling */
    .error {
        border-color: #ef4444 !important;
        box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.1) !important;
    }

    /* Success state using getting_started green */
    .success-state {
        border-color: var(--primary-green) !important;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.1) !important;
    }











    /* Label styling */
    .gradio-label {
        color: var(--text-primary);
        font-weight: 600;
    }

    /* Markdown content styling */
    .markdown-body {
        color: var(--text-secondary);
        line-height: 1.6;
    }

    .markdown-body h1, .markdown-body h2, .markdown-body h3 {
        color: var(--text-primary);
    }

    /* Step indicators styling */
    .gradio-markdown h1, .gradio-markdown h2, .gradio-markdown h3,
    .gradio-markdown p {
        margin: 0.25rem 0;
    }

    /* Enhanced step indicators with numbers */
    .gradio-markdown:contains("1."), .gradio-markdown:contains("2."), 
    .gradio-markdown:contains("3."), .gradio-markdown:contains("4."),
    .gradio-markdown:contains("5."), .gradio-markdown:contains("6."),
    .gradio-markdown:contains("7.") {
        position: relative;
        padding-left: 2.5rem;
        color: var(--text-primary);
        font-weight: 600;
    }

    /* Specific button styling */
    #undo_btnSEG, #dilate_btn, #erode_btn, #bounding_box_btn {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 249, 255, 0.95) 100%);
        border: 1px solid rgba(14, 165, 233, 0.2);
        color: var(--text-secondary);
        font-weight: 500;
        padding: 8px 16px;
        border-radius: 6px;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1);
        transition: all 0.3s ease;
    }

    #undo_btnSEG:hover, #dilate_btn:hover, #erode_btn:hover, #bounding_box_btn:hover {
        background: var(--primary-blue);
        border-color: var(--primary-blue);
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
    }

    /* Submit button enhanced styling - unified with primary blue */
    button[variant="primary"], .gradio-button.primary {
        background: var(--primary-blue);
        border-color: var(--primary-blue);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 12px 24px;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.25);
        transition: all 0.3s ease;
    }

    button[variant="primary"]:hover, .gradio-button.primary:hover {
        background: #0284c7;
        border-color: #0284c7;
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.4);
        transform: translateY(-2px);
    }

    /* Improved button states */
    button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
        transform: none !important;
        box-shadow: none !important;
    }

    button:disabled::after {
        display: none;
    }

    button.processing {
        background: var(--neutral-400) !important;
        border-color: var(--neutral-400) !important;
        cursor: wait;
        animation: processingPulse 2s ease-in-out infinite;
    }

    @keyframes processingPulse {
        0%, 100% { opacity: 0.8; }
        50% { opacity: 1; }
    }

    /* Responsive improvements */
    @media (max-width: 768px) {
        .header-content {
            padding: 1.2rem 1.8rem;
            margin: 0 1rem;
        }
        
        .main-title {
            font-size: 2rem;
        }
        
        .title-icon {
            font-size: 1.8rem;
        }
        
        .subtitle {
            font-size: 0.9rem;
        }
        
        .header-badges {
            gap: 0.6rem;
        }
        
        .badge-link {
            padding: 0.4rem 0.8rem;
            font-size: 0.85rem;
        }
        
        .header-badge {
            padding: 0.4rem 0.8rem;
            font-size: 0.85rem;
        }

        /* Getting Started responsive */
        .getting-started-container {
            padding: 1rem;
            margin: 0 0.5rem;
        }
        
        .guide-title {
            font-size: 1.1rem;
        }
        
        .guide-subtitle {
            font-size: 0.85rem;
        }
        
        .step-card {
            padding: 0.8rem;
            margin-bottom: 1rem;
        }
        
        .step-header {
            font-size: 0.9rem;
        }
        
        .step-number {
            width: 22px;
            height: 22px;
            font-size: 0.7rem;
        }
        
        .step-list {
            font-size: 0.8rem;
            padding-left: 1rem;
        }
        
        .tips-card {
            padding: 0.6rem;
        }
        
        .tips-content {
            font-size: 0.75rem;
        }
        
        .final-message {
            padding: 0.6rem;
        }
        
        .final-text {
            font-size: 0.8rem;
        }
        
        button {
            min-height: 44px;
        }
        
        input, textarea, select {
            min-height: 44px;
        }

        /* Mobile optimization for subtle effects */
        #global_glass_container {
            padding: 16px;
            margin: 8px;
            border-radius: 16px;
        }

        #global_glass_container::after {
            height: 180px;
            animation-duration: 18s;
        }

        #glass_card {
            padding: 20px;
            margin: 10px;
            border-radius: 12px;
        }

        #glass_card .gradio-row { gap: 12px; }
        #glass_card .gradio-column { gap: 12px; }


    }

    /* Ensure gallery works properly in all screen sizes */
    @media (min-width: 1200px) {
        #mask_gallery .gradio-gallery, #result_gallery .gradio-gallery {
            min-height: 300px !important;
            max-height: 80vh !important;
        }
        
        .responsive-gallery .grid-container {
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)) !important;
        }
    }

    /* Fix for gallery duplicate content issue - ensure clean display */
    #mask_gallery > div > div:nth-child(n+2),
    #result_gallery > div > div:nth-child(n+2) {
        display: none !important;
    }

    #mask_gallery .gradio-gallery:nth-child(n+2),
    #result_gallery .gradio-gallery:nth-child(n+2) {
        display: none !important;
    }

    """
