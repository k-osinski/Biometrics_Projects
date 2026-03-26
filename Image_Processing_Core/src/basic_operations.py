import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)


def adjust_brightness(img: np.ndarray, value: int) -> np.ndarray:
    result = img.astype(np.int16) + value
    return np.clip(result, 0, 255).astype(np.uint8)


def adjust_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    mean = np.mean(img)
    result = (img.astype(np.float64) - mean) * factor + mean
    return np.clip(result, 0, 255).astype(np.uint8)


def negative(img: np.ndarray) -> np.ndarray:
    return 255 - img
