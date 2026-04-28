import numpy as np
from .convolution import convolve2d


def mean_filter(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) / (kernel_size * kernel_size)
    return np.clip(convolve2d(img, kernel), 0, 255).astype(np.uint8)


def gaussian_filter(img: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    ax = np.arange(kernel_size) - kernel_size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return np.clip(convolve2d(img, kernel), 0, 255).astype(np.uint8)


def sharpen_filter(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    kernel = np.array([
        [ 0, -strength,  0],
        [-strength, 1 + strength * 4, -strength],
        [ 0, -strength,  0]
    ])
    return np.clip(convolve2d(img, kernel), 0, 255).astype(np.uint8)


def custom_filter(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.clip(convolve2d(img, kernel), 0, 255).astype(np.uint8)
