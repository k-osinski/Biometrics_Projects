import numpy as np
from basic_operations import to_grayscale


def binarize(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    if len(img.shape) == 3:
        gray = to_grayscale(img)
    else:
        gray = img
    return ((gray >= threshold) * 255).astype(np.uint8)

def otsu_threshold(img: np.ndarray) -> int:
    """
    Wyznacza optymalny próg binaryzacji metodą Otsu.
    """
    if len(img.shape) == 3:
        gray = to_grayscale(img)
    else:
        gray = img

    # Obliczenie histogramu (ile pikseli ma daną jasność)
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    
    # Normalizacja histogramu do prawdopodobieństw
    p = hist / hist.sum()
    
    # Skumulowane sumy prawdopodobieństw (wagi klas omega)
    omega = np.cumsum(p)
    omega = np.clip(omega, 1e-8, 1 - 1e-8)
    
    # Skumulowane średnie (mu)
    mu = np.cumsum(p * np.arange(256))
    mu_total = mu[-1] # Średnia jasność całego obrazu
    
    # Obliczenie wariancji międzyklasowej dla każdego progu T
    variance = (mu_total * omega - mu)**2 / (omega * (1 - omega))
    
    # Próg z maksymalną wariancją
    optimal_t = np.argmax(variance)
    
    return optimal_t

def binarize_otsu(img: np.ndarray) -> np.ndarray:
    """
    Wykonuje binaryzację obrazu przy użyciu progu wyznaczonego metodą Otsu.
    """
    optimal_t = otsu_threshold(img)
    return binarize(img, threshold=optimal_t)