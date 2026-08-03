"""
Flat-field correction for photos of paintings on a wall, using the
astronomical flat-fielding technique: divide the raw photo by a photo of
a "blank" reference (a clean patch of wall) to cancel out uneven lighting,
vignetting, and sensor artifacts.

This is a direct generalization of the user's original division-based
script -- same math, just packaged as a reusable function that accepts
the flat field as either a color or grayscale image.
"""

import cv2
import numpy as np


def apply_flat_field(color_bgr, flat_bgr_or_gray):
    """
    color_bgr: the raw photo to correct (3-channel BGR, uint8)
    flat_bgr_or_gray: photo of a clean/blank wall patch, same lighting
                       conditions, same session (3-channel BGR or 1-channel gray)

    Returns the corrected image (3-channel BGR, uint8).
    """
    color_float = color_bgr.astype(np.float32)

    if flat_bgr_or_gray.ndim == 3:
        flat_gray = cv2.cvtColor(flat_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        flat_gray = flat_bgr_or_gray
    flat_float = flat_gray.astype(np.float32)
    flat_float[flat_float == 0] = 0.001  # avoid divide-by-zero

    flat_3ch = cv2.merge([flat_float, flat_float, flat_float])

    result = cv2.divide(color_float, flat_3ch) * 255.0
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result
