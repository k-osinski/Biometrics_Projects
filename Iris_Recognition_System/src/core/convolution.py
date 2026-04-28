import numpy as np


def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    if len(img.shape) == 3:
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="edge")
        h, w, c = img.shape
        result = np.zeros_like(img, dtype=np.float64)
        for i in range(kh):
            for j in range(kw):
                result += kernel[i, j] * padded[i:i+h, j:j+w, :]
    else:
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
        h, w = img.shape
        result = np.zeros_like(img, dtype=np.float64)
        for i in range(kh):
            for j in range(kw):
                result += kernel[i, j] * padded[i:i+h, j:j+w]

    return result
