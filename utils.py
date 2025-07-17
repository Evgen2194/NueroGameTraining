import numpy as np
from PIL import Image
import cv2

def preprocess_frame(frame, resolution):
    """
    Resize and normalize a single frame.
    """
    img = Image.fromarray(frame)
    img = img.resize(resolution, Image.Resampling.LANCZOS)
    img_np = np.array(img)
    return img_np.astype(np.float32) / 255.0

def stack_frames(stacked_frames, new_frame, stack_size):
    """
    Stack frames for temporal information.
    """
    if stacked_frames is None:
        stacked_frames = np.stack([new_frame] * stack_size, axis=0)
    else:
        stacked_frames = np.roll(stacked_frames, -1, axis=0)
        stacked_frames[-1] = new_frame
    return stacked_frames

def capture_screen(region=None):
    """
    Capture the screen or a region of it.
    """
    screenshot = pyautogui.screenshot(region=region)
    return np.array(screenshot)
