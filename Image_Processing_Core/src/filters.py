import cv2
import numpy as np


def mean_filter(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    return cv2.blur(img, (kernel_size, kernel_size))


def gaussian_filter(img: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


def sharpen_filter(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    kernel = np.array([
        [ 0, -strength,  0],
        [-strength, 1 + strength * 4, -strength],
        [ 0, -strength,  0]
    ])
    result = cv2.filter2D(img, -1, kernel)
    return np.clip(result, 0, 255).astype(np.uint8)
