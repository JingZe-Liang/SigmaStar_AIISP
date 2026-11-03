import cv2
import numpy as np


def suppress_color_difference_artifacts(camera_rgb, kernel_size=5, strength=0.85):
    camera = camera_rgb.astype(np.float32, copy=False)
    green = camera[..., 1]
    red_diff = camera[..., 0] - green
    blue_diff = camera[..., 2] - green

    red_filtered = cv2.medianBlur(red_diff, kernel_size)
    blue_filtered = cv2.medianBlur(blue_diff, kernel_size)
    red_diff = red_diff * (1.0 - strength) + red_filtered * strength
    blue_diff = blue_diff * (1.0 - strength) + blue_filtered * strength

    out = camera.copy()
    out[..., 0] = green + red_diff
    out[..., 2] = green + blue_diff
    return np.clip(out, 0.0, None)
