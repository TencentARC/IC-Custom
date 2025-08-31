#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Constants and default values for IC-Custom application.
"""
from aspect_ratio_template import ASPECT_RATIO_TEMPLATE

# Aspect ratio constants
ASPECT_RATIO_LABELS = list(ASPECT_RATIO_TEMPLATE)
DEFAULT_ASPECT_RATIO = ASPECT_RATIO_LABELS[0]

# Colors and markers for segmentation
# OpenCV expects BGR colors; keep tuples as (R, G, B) for consistency across code.
SEGMENTATION_COLORS = [(255, 0, 0), (0, 255, 0)]
SEGMENTATION_MARKERS = [1, 5]
RGBA_COLORS = [(255, 0, 255, 255), (0, 255, 0, 255), (0, 0, 255, 255)]

# Magic-number constants
DEFAULT_BACKGROUND_BLEND_THRESHOLD = 0.5
DEFAULT_NUM_STEPS = 32
DEFAULT_GUIDANCE = 40
DEFAULT_TRUE_GS = 1
DEFAULT_NUM_IMAGES = 1
DEFAULT_SEED = -1  # -1 indicates random seed
DEFAULT_DILATION_KERNEL_SIZE = 7

# UI constants
DEFAULT_BRUSH_SIZE = 30
DEFAULT_MARKER_SIZE = 20
DEFAULT_MARKER_THICKNESS = 5
DEFAULT_MASK_ALPHA = 0.3
DEFAULT_COLOR_ALPHA = 0.7

# File naming
TIMESTAMP_FORMAT = "%Y%m%d_%H%M"
