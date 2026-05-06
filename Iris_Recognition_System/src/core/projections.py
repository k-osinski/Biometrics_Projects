"""
Projekcje (sumy 1D) obrazu binarnego oraz pomocnicze procedury
do wyznaczania środka i promienia obiektu na ich podstawie.

Te funkcje są wykorzystywane do detekcji źrenicy i tęczówki.
Zwracają surowe wektory liczb (NumPy), w przeciwieństwie do
modułu `analysis.py`, który zwraca wykresy do GUI.
"""
import numpy as np
from .basic_operations import to_grayscale


def horizontal_projection_1d(img: np.ndarray) -> np.ndarray:
    """
    Projekcja pozioma — suma jasności w każdym wierszu.
    Wynik: wektor o długości równej wysokości obrazu.
    Wartość projekcji[i] mówi "jak dużo jest obiektu w wierszu i".
    """
    if img.ndim == 3:
        img = to_grayscale(img)
    return np.sum(img.astype(np.int64), axis=1)


def vertical_projection_1d(img: np.ndarray) -> np.ndarray:
    """
    Projekcja pionowa — suma jasności w każdej kolumnie.
    Wynik: wektor o długości równej szerokości obrazu.
    """
    if img.ndim == 3:
        img = to_grayscale(img)
    return np.sum(img.astype(np.int64), axis=0)


def find_object_extent(projection: np.ndarray,
                       threshold_ratio: float = 0.5) -> tuple[int, int, int]:
    """
    Wyznacza środek i połowę szerokości "obiektu" na podstawie projekcji.
    Procedura:
        1. Znajdź wartość maksymalną projekcji.
        2. Wyznacz próg = threshold_ratio * max.
        3. Znajdź najdłuższy spójny przedział, w którym projekcja >= próg
           i który zawiera punkt maksimum.
        4. Środek = środek tego przedziału, promień = pół jego szerokości.
    Zwraca:
        (center, radius, peak_index)
    """
    if projection.size == 0 or projection.max() == 0:
        return (0, 0, 0)

    peak = int(np.argmax(projection))
    threshold = threshold_ratio * projection[peak]

    left = peak
    while left > 0 and projection[left - 1] >= threshold:
        left -= 1
    right = peak
    while right < projection.size - 1 and projection[right + 1] >= threshold:
        right += 1

    center = (left + right) // 2
    radius = max((right - left) // 2, 1)
    return (center, radius, peak)


def estimate_circle_from_projections(binary_img: np.ndarray,
                                     threshold_ratio: float = 0.5
                                     ) -> tuple[int, int, int]:
    """
    Z obrazu binarnego (obiekt = 255, tło = 0) szacuje (cx, cy, r).
    Korzysta z dwóch projekcji:
        - pionowa  -> środek i promień w osi x
        - pozioma  -> środek i promień w osi y
    Promień końcowy to średnia z obu wymiarów.
    """
    vert = vertical_projection_1d(binary_img)
    horiz = horizontal_projection_1d(binary_img)

    cx, rx, _ = find_object_extent(vert, threshold_ratio)
    cy, ry, _ = find_object_extent(horiz, threshold_ratio)
    r = int(round((rx + ry) / 2))
    return (cx, cy, r)


def diagonal_projection(img: np.ndarray, direction: str = "main") -> np.ndarray:
    """
    Projekcja po przekątnej (sumy wzdłuż diagonali).
    Wykład wspomina o projekcji "pod kątem" jako uzupełnieniu pionowej i
    poziomej. Implementacja zwraca wektor sum dla każdej diagonali.
    direction:
        "main" - przekątne równoległe do głównej (lewy-górny -> prawy-dolny)
        "anti" - przekątne równoległe do antydiagonali
    """
    if img.ndim == 3:
        img = to_grayscale(img)

    arr = img.astype(np.int64)
    if direction == "anti":
        arr = np.fliplr(arr)

    h, w = arr.shape
    sums = []
    for offset in range(-(h - 1), w):
        sums.append(int(np.trace(arr, offset=offset)))
    return np.asarray(sums, dtype=np.int64)
