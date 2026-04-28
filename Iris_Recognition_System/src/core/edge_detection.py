import numpy as np
from .basic_operations import to_grayscale
from .convolution import convolve2d


def roberts_cross(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        gray = to_grayscale(img)
    else:
        gray = img

    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float64)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float64)

    gx = convolve2d(gray.astype(np.float64), kernel_x)
    gy = convolve2d(gray.astype(np.float64), kernel_y)

    magnitude = np.sqrt(gx**2 + gy**2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def sobel_operator(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        gray = to_grayscale(img)
    else:
        gray = img

    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    gx = convolve2d(gray.astype(np.float64), kernel_x)
    gy = convolve2d(gray.astype(np.float64), kernel_y)

    magnitude = np.sqrt(gx**2 + gy**2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)
