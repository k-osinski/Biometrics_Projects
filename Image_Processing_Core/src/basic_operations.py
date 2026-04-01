import numpy as np

def to_grayscale(img: np.ndarray, method: str = "luminance_bt601") -> np.ndarray:
    if len(img.shape) == 2:
        return img
    
    r = img[:, :, 0].astype(np.float64)
    g = img[:, :, 1].astype(np.float64)
    b = img[:, :, 2].astype(np.float64)
    
    if method == "average":
        gray = (r + g + b) / 3
    elif method == "luminance_bt709":
        gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    elif method == "desaturation":
        gray = (np.maximum(np.maximum(r, g), b) + np.minimum(np.minimum(r, g), b)) / 2
    else:
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        
    return np.clip(gray, 0, 255).astype(np.uint8)

def adjust_brightness(img: np.ndarray, value: int) -> np.ndarray:
    result = img.astype(np.int16) + value
    return np.clip(result, 0, 255).astype(np.uint8)


def adjust_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    mean = np.mean(img)
    result = (img.astype(np.float64) - mean) * factor + mean
    return np.clip(result, 0, 255).astype(np.uint8)


def negative(img: np.ndarray) -> np.ndarray:
    return 255 - img

def adjust_gamma(img: np.ndarray, alpha: float) -> np.ndarray:
    """
    Nieliniowa poprawa kontrastu - potęgowanie (korekta gamma).
    Wzór: J_w(x,y) = 255 * (J(x,y) / 255)^alpha
    Dla alpha < 1 rozjaśnia, dla alpha > 1 przyciemnia.
    """
    img_normalized = img.astype(np.float64) / 255.0
    result = 255.0 * (img_normalized ** alpha)
    return np.clip(result, 0, 255).astype(np.uint8)

def adjust_logarithmic(img: np.ndarray) -> np.ndarray:
    """
    Nieliniowa poprawa kontrastu - logarytmowanie.
    Wzór: J_w(x,y) = 255 * (log(1 + J(x,y)) / log(1 + 255))
    Rozjaśnia obraz z większym zróżnicowaniem ciemnych partii.
    """
    img_float = img.astype(np.float64)
    result = 255.0 * (np.log(1 + img_float) / np.log(1 + 255.0))
    return np.clip(result, 0, 255).astype(np.uint8)

def equalize_histogram(img: np.ndarray) -> np.ndarray:
    """
    Wyrównywanie histogramu (spłaszczenie) dla obrazów 8-bitowych.
    Zwiększa kontrast poprzez równomierne rozłożenie poziomów jasności.
    """
    if len(img.shape) == 3:
        img = to_grayscale(img)
    # Zliczanie wystąpień każdego poziomu jasności (0-255)
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
    # Suma skumulowana
    cdf = hist.cumsum()
    # Normalizacja
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_normalized = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    cdf_final = np.ma.filled(cdf_normalized, 0).astype(np.uint8)
    return cdf_final[img]