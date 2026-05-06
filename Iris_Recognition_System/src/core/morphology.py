"""
Operacje morfologiczne dla obrazów binarnych.
Konwencja:
    - obraz binarny zawiera wartości 0 (tło) oraz 255 (obiekt)
    - element strukturalny (kernel) to ndarray z wartościami 0/1
    - dla obrazów po binaryzacji (binarize_*) "obiekt" = wartości 255.
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


def label_components(img: np.ndarray,
                     connectivity: int = 4) -> tuple[np.ndarray, dict[int, int]]:
    """
    Etykietowanie spójnych komponentów na obrazie binarnym.

    Algorytm: iteracyjny flood-fill (BFS) na stosie. Każdy komponent
    dostaje unikalną etykietę całkowitą >= 1; tło zostaje 0.

    Argumenty:
        img: obraz binarny (uint8, 0 = tło, >0 = obiekt) albo bool.
        connectivity: 4 (sąsiedzi N/S/E/W) lub 8 (z przekątnymi).

    Zwraca:
        (label_img, sizes), gdzie:
            label_img - tablica int32 o tym samym kształcie co `img`,
            sizes     - słownik {etykieta: liczba_pikseli}.
    """
    mask = _to_bool(img)
    h, w = mask.shape
    label_img = np.zeros((h, w), dtype=np.int32)
    sizes: dict[int, int] = {}

    if connectivity == 8:
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]
    else:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    current_label = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if label_img[y0, x0] != 0:
            continue
        current_label += 1
        size = 0
        stack = [(int(y0), int(x0))]
        while stack:
            y, x = stack.pop()
            if label_img[y, x] != 0:
                continue
            label_img[y, x] = current_label
            size += 1
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w \
                        and mask[ny, nx] and label_img[ny, nx] == 0:
                    stack.append((ny, nx))
        sizes[current_label] = size

    return label_img, sizes


def keep_largest_component(img: np.ndarray,
                           connectivity: int = 4) -> np.ndarray:
    """
    Zachowuje tylko największy spójny komponent na obrazie binarnym.

    Wszystko poza nim (mniejsze odłamki, "kępy" rzęs, ramki) zostaje
    wyzerowane. Operacja jest częścią morfologii matematycznej (tzw.
    "area opening": usuń wszystkie komponenty mniejsze od progu - tu
    progiem jest rozmiar największego z nich).

    Argumenty:
        img: obraz binarny (uint8 lub bool).
        connectivity: 4 albo 8.

    Zwraca:
        obraz uint8 (0 / 255).
    """
    label_img, sizes = label_components(img, connectivity=connectivity)
    if not sizes:
        return np.zeros_like(img, dtype=np.uint8)
    largest = max(sizes, key=sizes.get)
    return _from_bool(label_img == largest)
