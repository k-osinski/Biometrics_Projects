"""
Detekcja źrenicy (czarna tarczka w środku oka).

Algorytm::
    1.  Konwersja do skali szarości.
    2.  Wyznaczenie progu bazowego P:
            P = (1 / (h * w)) * Σ Σ A(i, j)
        dzielonego przez X_P, otrzymując próg P_P = P / X_P.
        Źrenica jest ciemniejsza od średniej -> binaryzujemy
        wybierając piksele o jasności < P_P jako "obiekt" (255).
    3.  Czyszczenie obrazu binarnego operacjami morfologicznymi:
            - zamknięcie  -> wypełnia "dziury" w źrenicy
            - otwarcie    -> usuwa drobne piksele rzęs / odbić
    4.  Wyznaczenie środka i promienia za pomocą projekcji
        pionowej i poziomej.
"""
from dataclasses import dataclass
import numpy as np

from ..core.basic_operations import to_grayscale
from ..core.morphology import (
    closing, opening, get_structuring_element, keep_largest_component,
)
from ..core.projections import estimate_circle_from_projections


@dataclass
class PupilResult:
    """Wynik detekcji źrenicy."""
    cx: int
    cy: int
    radius: int
    threshold: int
    binary_mask: np.ndarray


def base_mean_threshold(gray: np.ndarray) -> float:
    """
    Próg bazowy P = średnia jasność obrazu (zgodnie z wzorem z PDF).
    """
    if gray.ndim == 3:
        gray = to_grayscale(gray)
    return float(np.mean(gray))


def binarize_dark(gray: np.ndarray, threshold: float) -> np.ndarray:
    """
    Binaryzacja "odwrotna": piksele *ciemniejsze* od progu zostają
    obiektem (wartość 255). Tak izolujemy źrenicę.
    """
    return ((gray <= threshold) * 255).astype(np.uint8)


def detect_pupil(gray: np.ndarray,
                 x_p: float = 3.0,
                 close_size: int = 7,
                 open_size: int = 5,
                 projection_threshold: float = 0.5,
                 keep_largest: bool = True) -> PupilResult:
    """
    Detekcja źrenicy.

    Parametry:
        gray
            obraz w skali szarości (uint8) lub RGB - automatycznie
            zostanie przekonwertowany.
        x_p
            mianownik we wzorze P_P = P / X_P. Im większe x_p, tym
            niższy próg -> "ostrzejsze" wycinanie tylko czarnej
            źrenicy. Wartości typowo 2.5 - 4.0 dla MMU.
        close_size, open_size
            rozmiary kerneli (kwadrat) używanych w domykaniu /
            otwarciu morfologicznym.
        projection_threshold
            próg (jako ułamek maksimum projekcji), powyżej którego
            uznajemy, że dany wiersz/kolumna należy do źrenicy.
        keep_largest
            jeśli True, po cleanupie morfologicznym pozostawiany jest
            tylko *największy* spójny komponent (area opening).
            Dzięki temu projekcje nie są zakłócane przez ewentualne
            kępy rzęs / cienie / ramki obiektywu, które przetrwały
            otwarcie morfologiczne. Operacja jest "ostatnią linią
            obrony" - przy dobrze dobranych X_P, close_size, open_size
            jest no-op (komponent i tak jest jeden).

    Zwraca:
        PupilResult(cx, cy, radius, threshold, binary_mask).
    """
    if gray.ndim == 3:
        gray = to_grayscale(gray)

    base = base_mean_threshold(gray)
    threshold = base / max(x_p, 1e-6)

    binary = binarize_dark(gray, threshold)

    if close_size and close_size > 1:
        binary = closing(binary, get_structuring_element("square", close_size))
    if open_size and open_size > 1:
        binary = opening(binary, get_structuring_element("square", open_size))

    if keep_largest:
        binary = keep_largest_component(binary, connectivity=4)

    cx, cy, r = estimate_circle_from_projections(binary, projection_threshold)
    return PupilResult(cx=cx, cy=cy, radius=r,
                       threshold=int(round(threshold)),
                       binary_mask=binary)
