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
    
    /* Keep your existing radio button styles exactly as they are */
    #my_radio .wrap {
        display: flex;
        flex-wrap: nowrap;
        justify-content: center;
        align-items: center;
    }

    #my_radio .wrap label {
        display: flex;
        width: 50%;
        justify-content: center;
        align-items: center;
        margin: 0;
        padding: 10px 0;
        box-sizing: border-box;
    }

    /* Global smooth transitions - affects all elements */
    * {
        transition: all 0.2s ease;
    }

    /* Better focus states for accessibility */
    button:focus,
    input:focus,
    select:focus,
    textarea:focus {
        outline: none;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
    }

    /* Subtle hover effects for interactive elements */
    button:hover {
        transform: translateY(-1px);
    }

    /* Improved text readability */
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        line-height: 1.6;
        color: #374151;
    }

    /* Better form element styling */
    input, textarea, select {
        border-radius: 4px;
        border: 1px solid #d1d5db;
    }

    input:focus, textarea:focus, select:focus {
        border-color: #3b82f6;
    }

    /* Custom scrollbar for better aesthetics */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }

    /* Better table styling */
    table {
        border-radius: 6px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    th {
        background-color: #f8fafc;
        font-weight: 600;
    }

    /* Improved progress bars */
    .progress {
        border-radius: 4px;
        overflow: hidden;
    }

    /* Better tooltip styling */
    [title]:hover::after {
        background: #1f2937;
        color: #f9fafb;
        border-radius: 4px;
        padding: 0.5rem 0.75rem;
        font-size: 0.875rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Loading state improvements */
    .loading {
        opacity: 0.7;
        transition: opacity 0.2s ease;
    }

    /* Success state improvements */
    .success {
        animation: subtle-pulse 0.6s ease;
    }

    @keyframes subtle-pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.005); }
        100% { transform: scale(1); }
    }

    /* Better button styling */
    button {
        border-radius: 6px;
        font-weight: 500;
        cursor: pointer;
    }

    /* Improved checkbox styling */
    input[type="checkbox"] {
        border-radius: 4px;
    }

    input[type="checkbox"]:checked {
        background-color: #3b82f6;
        border-color: #3b82f6;
    }

    /* Better radio button styling */
    input[type="radio"] {
        border-radius: 50%;
    }

    input[type="radio"]:checked {
        background-color: #3b82f6;
        border-color: #3b82f6;
    }

    /* Improved slider styling */
    input[type="range"] {
        accent-color: #3b82f6;
    }

    /* Better image styling */
    img {
        border-radius: 6px;
        transition: all 0.2s ease;
    }

    img:hover {
        transform: scale(1.01);
    }

    /* Improved accordion styling */
    .accordion {
        border-radius: 6px;
        overflow: hidden;
    }

    /* Better group styling */
    .group {
        border-radius: 6px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }

    /* Enhanced focus visible for keyboard navigation */
    *:focus-visible {
        outline: 2px solid #3b82f6;
        outline-offset: 2px;
    }

    /* Better selection styling */
    ::selection {
        background: rgba(59, 130, 246, 0.2);
        color: #1f2937;
    }

    /* Improved placeholder styling */
    ::placeholder {
        color: #9ca3af;
        opacity: 1;
    }

    /* Better disabled state styling */
    button:disabled,
    input:disabled,
    select:disabled,
    textarea:disabled {
        opacity: 0.6;
        cursor: not-allowed;
    }

    /* Improved error state styling */
    .error {
        border-color: #ef4444 !important;
        box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.1) !important;
    }

    /* Better success state styling */
    .success-state {
        border-color: #10b981 !important;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.1) !important;
    }

    /* Responsive improvements */
    @media (max-width: 768px) {
        button {
            min-height: 44px;
        }
        
        input, textarea, select {
            min-height: 44px;
        }
    }
    """


def get_js() -> str:
    return r"""
    // Global Optimization Effects - No Layout Changes
    document.addEventListener('DOMContentLoaded', function() {
        // Smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });

        // Enhanced form validation feedback
        document.querySelectorAll('input, textarea, select').forEach(input => {
            input.addEventListener('blur', function() {
                if (this.value.trim() === '') {
                    this.classList.add('error');
                    this.classList.remove('success-state');
                } else {
                    this.classList.remove('error');
                    this.classList.add('success-state');
                }
            });
            
            input.addEventListener('input', function() {
                if (this.value.trim() !== '') {
                    this.classList.remove('error');
                    this.classList.add('success-state');
                }
            });
        });

        // Subtle button loading states
        document.querySelectorAll('button').forEach(button => {
            button.addEventListener('click', function() {
                if (this.textContent.includes('Run') || this.textContent.includes('Submit')) {
                    this.classList.add('loading');
                    this.textContent = 'Processing...';
                    
                    setTimeout(() => {
                        this.classList.remove('loading');
                        this.textContent = this.textContent.includes('Run') ? '💫 Run' : 'Submit';
                    }, 5000);
                }
            });
        });

        // Gallery success animations
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    mutation.addedNodes.forEach(function(node) {
                        if (node.nodeType === 1 && node.classList.contains('gallery')) {
                            node.classList.add('success');
                            setTimeout(() => node.classList.remove('success'), 600);
                        }
                    });
                }
            });
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        // Keyboard shortcuts for power users
        document.addEventListener('keydown', function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                const submitBtn = document.querySelector('button[data-testid="submit"]') || 
                                document.querySelector('button:contains("Run")') ||
                                document.querySelector('button:contains("Submit")');
                if (submitBtn) {
                    submitBtn.click();
                }
            }
            
            if (e.key === 'Escape') {
                const clearBtn = document.querySelector('button[data-testid="clear"]') ||
                               document.querySelector('button:contains("Clear")');
                if (clearBtn) {
                    clearBtn.click();
                }
            }
        });

        // Enhanced focus management
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Tab') {
                document.body.classList.add('keyboard-navigation');
            }
        });

        document.addEventListener('mousedown', function() {
            document.body.classList.remove('keyboard-navigation');
        });

        // Keyboard navigation styles
        const keyboardStyle = document.createElement('style');
        keyboardStyle.textContent = `
            .keyboard-navigation *:focus {
                outline: 2px solid #3b82f6 !important;
                outline-offset: 2px !important;
            }
        `;
        document.head.appendChild(keyboardStyle);

        // Subtle progress indicators
        function addProgressIndicator(container) {
            const progress = document.createElement('div');
            progress.style.cssText = `
                width: 100%;
                height: 4px;
                background: #e5e7eb;
                border-radius: 2px;
                overflow: hidden;
                margin-top: 0.5rem;
            `;
            
            const progressFill = document.createElement('div');
            progressFill.style.cssText = `
                height: 100%;
                background: #3b82f6;
                width: 0%;
                transition: width 0.3s ease;
                border-radius: 2px;
            `;
            
            progress.appendChild(progressFill);
            container.appendChild(progress);
            
            let width = 0;
            const interval = setInterval(() => {
                if (width >= 90) {
                    clearInterval(interval);
                } else {
                    width += Math.random() * 4;
                    progressFill.style.width = width + '%';
                }
            }, 200);
        }

        // Add progress bars to submit buttons
        document.querySelectorAll('button[data-testid="submit"], button:contains("Run")').forEach(btn => {
            btn.addEventListener('click', function() {
                addProgressIndicator(this.parentElement);
            });
        });

        // Better image loading
        document.querySelectorAll('img').forEach(img => {
            img.addEventListener('load', function() {
                this.style.opacity = '1';
            });
            
            img.style.opacity = '0';
            img.style.transition = 'opacity 0.3s ease';
        });

        // Improved form submission feedback
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', function() {
                const submitBtn = this.querySelector('button[type="submit"]');
                if (submitBtn) {
                    submitBtn.classList.add('loading');
                }
            });
        });

        // Better error handling
        window.addEventListener('error', function(e) {
            console.error('Global error:', e.error);
        });

        // Performance monitoring
        if ('performance' in window) {
            window.addEventListener('load', function() {
                const perfData = performance.getEntriesByType('navigation')[0];
                console.log('Page load time:', perfData.loadEventEnd - perfData.loadEventStart, 'ms');
            });
        }
    });
    """
