import numpy as np
import cv2
from .basic_operations import to_grayscale


def binarize(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    if len(img.shape) == 3:
        gray = to_grayscale(img)
    else:
        gray = img
    return ((gray >= threshold) * 255).astype(np.uint8)

# Progowanie globalne (Metoda Otsu)
def otsu_threshold(img: np.ndarray) -> int:
    """
    Wyznacza optymalny próg binaryzacji metodą Otsu.
    """
    if len(img.shape) == 3:
        gray = to_grayscale(img)
    else:
        gray = img

    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    p = hist / hist.sum()
    omega = np.cumsum(p)
    omega = np.clip(omega, 1e-8, 1 - 1e-8)
    mu = np.cumsum(p * np.arange(256))
    mu_total = mu[-1]
    variance = (mu_total * omega - mu)**2 / (omega * (1 - omega))
    optimal_t = np.argmax(variance)
    
    return optimal_t

def binarize_otsu(img: np.ndarray) -> np.ndarray:
    """
    Wykonuje binaryzację obrazu przy użyciu progu wyznaczonego metodą Otsu.
    """
    optimal_t = otsu_threshold(img)
    return binarize(img, threshold=optimal_t)

# Progowanie lokalne (Metoda Bernsena)
def threshold_bernsen_base(img: np.ndarray, window_size: int = 15) -> np.ndarray:
    if len(img.shape) == 3: img = to_grayscale(img)
    
    pad = window_size // 2
    padded = np.pad(img, pad, mode='edge')
    result = np.zeros_like(img)

    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(padded, (window_size, window_size))
    
    local_min = np.min(windows, axis=(2, 3))
    local_max = np.max(windows, axis=(2, 3))
    local_threshold = (local_min.astype(np.float64) + local_max.astype(np.float64)) / 2
    
    return ((img >= local_threshold) * 255).astype(np.uint8)

# Wersja zoptymalizowana (ze względu na problemy z alokacją pamięci)
def threshold_bernsen(img: np.ndarray, window_size: int = 15) -> np.ndarray:
    if len(img.shape) == 3: 
        from src.basic_operations import to_grayscale
        img = to_grayscale(img)
    
    kernel = np.ones((window_size, window_size), np.uint8)
    
    local_max = cv2.dilate(img, kernel).astype(np.float64)
    local_min = cv2.erode(img, kernel).astype(np.float64)

    local_threshold = (local_max + local_min) / 2.0

    return ((img >= local_threshold) * 255).astype(np.uint8)


# Progowanie adaptacyjne (Metoda Niblacka)
def threshold_niblack_base(img: np.ndarray, window_size: int = 15, k: float = -0.2) -> np.ndarray:
    if len(img.shape) == 3: img = to_grayscale(img)
    
    from numpy.lib.stride_tricks import sliding_window_view
    pad = window_size // 2
    padded = np.pad(img, pad, mode='edge')
    windows = sliding_window_view(padded, (window_size, window_size))
    
    local_mean = np.mean(windows, axis=(2, 3))
    local_std = np.std(windows, axis=(2, 3))
    
    local_threshold = local_mean + k * local_std
    return ((img >= local_threshold) * 255).astype(np.uint8)

# Wersja zoptymalizowana (ze względu na problemy z alokacją pamięci)
def threshold_niblack(img: np.ndarray, window_size: int = 15, k: float = -0.2) -> np.ndarray:
    if len(img.shape) == 3: 
        from src.basic_operations import to_grayscale
        img = to_grayscale(img)
        
    img_float = img.astype(np.float64)
    
    local_mean = cv2.blur(img_float, (window_size, window_size))
    local_mean_sq = cv2.blur(img_float**2, (window_size, window_size))
    variance = local_mean_sq - (local_mean**2)
    variance[variance < 0] = 0 
    local_std = np.sqrt(variance)
    
    local_threshold = local_mean + k * local_std
    
    return ((img >= local_threshold) * 255).astype(np.uint8)


# Wieloprogowanie (Multi-thresholding)
def multi_threshold(img: np.ndarray, low: int, high: int) -> np.ndarray:
    if len(img.shape) == 3: img = to_grayscale(img)
    mask = (img >= low) & (img <= high)
    return (mask * 255).astype(np.uint8)