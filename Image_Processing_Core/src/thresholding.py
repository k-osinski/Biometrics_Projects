import numpy as np
from src.basic_operations import to_grayscale


def binarize(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    if len(img.shape) == 3:
        gray = to_grayscale(img)
    else:
        gray = img
    return ((gray >= threshold) * 255).astype(np.uint8)
