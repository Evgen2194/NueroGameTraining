import numpy as np
from PIL import Image
import cv2
import pyautogui
import ctypes

def get_scaling_factor():
    try:
        # Query DPI awareness
        awareness = ctypes.c_int()
        ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
        # Query DPI for the main monitor
        dpi = ctypes.windll.user32.GetDpiForSystem()
        # The default DPI is 96. Scaling factor is the current DPI / 96.
        return dpi / 96.0
    except (AttributeError, TypeError):
        # For non-Windows platforms or if anything goes wrong, assume no scaling.
        return 1.0

def preprocess_frame(frame, resolution):
    """
    Convert to grayscale, resize, and normalize a single frame.
    """
    img = Image.fromarray(frame).convert('L') # Convert to grayscale
    img = img.resize(resolution, Image.Resampling.LANCZOS)
    img_np = np.array(img)
    # Add a channel dimension
    img_np = np.expand_dims(img_np, axis=0)
    return img_np.astype(np.float32) / 255.0

def stack_frames(stacked_frames, new_frame, stack_size):
    """
    Stack frames for temporal information.
    """
    if stacked_frames is None:
        # Initial setup: stack the new_frame stack_size times
        stacked_frames = np.concatenate([new_frame] * stack_size, axis=0)
    else:
        # Append new_frame and remove the oldest frame
        stacked_frames = np.concatenate((stacked_frames[1:], new_frame), axis=0)
    return stacked_frames

def capture_screen(region=None):
    """
    Capture the screen or a region of it.
    """
    screenshot = pyautogui.screenshot(region=region)
    return np.array(screenshot)
