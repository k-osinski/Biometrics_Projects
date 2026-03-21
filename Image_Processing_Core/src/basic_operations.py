import cv2
import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def adjust_brightness(img: np.ndarray, value: int) -> np.ndarray:
    result = img.astype(np.int16) + value
    return np.clip(result, 0, 255).astype(np.uint8)


def adjust_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    mean = np.mean(img)
    result = (img.astype(np.float64) - mean) * factor + mean
    return np.clip(result, 0, 255).astype(np.uint8)


def negative(img: np.ndarray) -> np.ndarray:
    return 255 - img
