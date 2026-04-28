"""
Operacje morfologiczne dla obrazów binarnych.

Konwencja:
    - obraz binarny zawiera wartości 0 (tło) oraz 255 (obiekt)
    - element strukturalny (kernel) to ndarray z wartościami 0/1
    - dla obrazów po binaryzacji (binarize_*) "obiekt" = wartości 255

Operacje implementowane są ręcznie (bez wywołań cv2.erode/dilate),
zgodnie z definicjami z wykładów oraz treścią Projektu 2.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _to_bool(img: np.ndarray) -> np.ndarray:
    """Sprowadza obraz binarny (0/255) lub (0/1) do tablicy bool."""
    return img > 0


def _from_bool(mask: np.ndarray) -> np.ndarray:
    """Zwraca obraz typu uint8 z wartościami 0/255."""
    return (mask.astype(np.uint8)) * 255


def get_structuring_element(shape: str = "square", size: int = 3) -> np.ndarray:
    """
    Tworzy element strukturalny.

    Parametry:
        shape : "square" | "cross" | "disk"
        size  : nieparzysty rozmiar kernela (np. 3, 5, 7).
    """
    if size % 2 == 0:
        size += 1
    if shape == "cross":
        kernel = np.zeros((size, size), dtype=np.uint8)
        c = size // 2
        kernel[c, :] = 1
        kernel[:, c] = 1
        return kernel
    if shape == "disk":
        c = size // 2
        y, x = np.ogrid[-c:c + 1, -c:c + 1]
        return ((x * x + y * y) <= c * c).astype(np.uint8)
    return np.ones((size, size), dtype=np.uint8)


def erode(img: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    """
    Erozja: A ⊖ B = { a : a + b ∈ A dla każdego b ∈ B }.

    Implementacja: piksel wynikowy = 1 wtedy i tylko wtedy, gdy
    wszystkie sąsiednie piksele wskazane przez kernel = 1.
    """
    if kernel is None:
        kernel = get_structuring_element("square", 3)

    mask = _to_bool(img)
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=False)

    windows = sliding_window_view(padded, (kh, kw))
    k = kernel.astype(bool)
    # piksel = AND po wszystkich aktywnych pozycjach kernela
    eroded = np.all(windows[:, :] | (~k), axis=(2, 3))
    return _from_bool(eroded)


def dilate(img: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    """
    Dylatacja: A ⊕ B = { p : p = a + b, a ∈ A, b ∈ B }.

    Piksel wynikowy = 1 wtedy i tylko wtedy, gdy w sąsiedztwie
    (zgodnie z kernelem) jest co najmniej jeden piksel obiektu.
    """
    if kernel is None:
        kernel = get_structuring_element("square", 3)

    mask = _to_bool(img)
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=False)

    windows = sliding_window_view(padded, (kh, kw))
    k = kernel.astype(bool)
    dilated = np.any(windows & k, axis=(2, 3))
    return _from_bool(dilated)


def opening(img: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    """Otwarcie: erozja, a następnie dylatacja. Usuwa drobne obiekty."""
    return dilate(erode(img, kernel), kernel)


def closing(img: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    """Zamknięcie: dylatacja, a następnie erozja. Wypełnia drobne dziury."""
    return erode(dilate(img, kernel), kernel)


def morphological_pipeline(img: np.ndarray,
                           operations: list[tuple[str, int]]) -> np.ndarray:
    """
    Wykonuje sekwencję operacji morfologicznych.

    operations: lista par (op_name, size), np.
        [("close", 5), ("open", 3)]
    op_name ∈ {"erode", "dilate", "open", "close"}.
    """
    fn_map = {"erode": erode, "dilate": dilate,
              "open": opening, "close": closing}
    out = img.copy()
    for op_name, size in operations:
        kernel = get_structuring_element("square", size)
        out = fn_map[op_name](out, kernel)
    return out
