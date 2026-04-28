"""
Detekcja zewnętrznej granicy tęczówki.

Algorytm jest analogiczny do detekcji źrenicy, ale:
    - używa innego progu binaryzacji P_I = P / X_I
      (typowo X_I bliskie 1, bo tęczówka jest tylko nieco ciemniejsza
       niż średnia jasność obrazu),
    - zamiast szukać dowolnego ciemnego obiektu w obrazie, ogranicza
      się do otoczenia źrenicy (wykorzystując znany środek źrenicy),
    - radius wyznaczamy szukając największego skoku jasności w
      profilu radialnym wzdłuż linii poziomej i pionowej, zaczynając
      od krawędzi źrenicy.

Dzięki temu unikamy "wpadania" w rzęsy lub powieki, a procedura jest
odporna nawet wtedy, gdy granica tęczówka/twardówka jest słabo
widoczna.
"""
from dataclasses import dataclass
import numpy as np

from ..core.basic_operations import to_grayscale
from ..core.filters import gaussian_filter
from .pupil_detector import (
    PupilResult, base_mean_threshold, binarize_dark
)


@dataclass
class IrisResult:
    """Wynik detekcji zewnętrznej granicy tęczówki."""
    cx: int
    cy: int
    radius: int
    threshold: int


def _radial_profile(gray: np.ndarray,
                    cx: int, cy: int,
                    direction: tuple[int, int],
                    max_distance: int) -> np.ndarray:
    """
    Wyciąga jasności wzdłuż prostej startującej w (cx, cy)
    w kierunku (dx, dy) (jednostkowy wektor).

    Zwraca tablicę dla d = 0, 1, ..., max_distance - 1.
    Punkty poza obrazem otrzymują NaN.
    """
    h, w = gray.shape
    dx, dy = direction
    profile = np.full(max_distance, np.nan, dtype=np.float64)
    for d in range(max_distance):
        x = cx + dx * d
        y = cy + dy * d
        if 0 <= x < w and 0 <= y < h:
            profile[d] = gray[y, x]
    return profile


def detect_iris(gray: np.ndarray,
                pupil: PupilResult,
                x_i: float = 1.15,
                margin: float = 1.4,
                smoothing_sigma: float = 2.0,
                min_radius_factor: float = 1.6,
                max_radius_factor: float = 4.0) -> IrisResult:
    """
    Detekcja granicy tęczówki.

    Parametry:
        gray
            obraz wejściowy (RGB lub L).
        pupil
            wynik detekcji źrenicy (potrzebny środek i promień).
        x_i
            mianownik progu binaryzacji P_I = P / X_I.
            Wartość ~1.0 - 1.3 odpowiada za "rozluźnione" progowanie
            (tęczówka jest jaśniejsza niż źrenica, ale ciemniejsza niż
            twardówka, więc próg musi być wyższy).
        margin
            tolerancja: maksymalny promień tęczówki rozważany w
            poszukiwaniu krawędzi = max_radius_factor * pupil.radius,
            ale nie mniej niż min_radius_factor * pupil.radius.
        smoothing_sigma
            sigma w filtrze Gaussa stosowanym przed obliczeniem
            gradientu radialnego (tłumi szum / rzęsy).
        min_radius_factor, max_radius_factor
            ograniczenia na wynikowy promień tęczówki w jednostkach
            promienia źrenicy.

    Zwraca:
        IrisResult(cx, cy, radius, threshold).
    """
    if gray.ndim == 3:
        gray = to_grayscale(gray)

    # Próg binaryzacji - obliczany analogicznie do źrenicy
    base = base_mean_threshold(gray)
    threshold = base / max(x_i, 1e-6)

    # Wygładzenie obrazu - kluczowe, bo gradient w obrazie surowym
    # jest dominowany przez rzęsy i tekstury tęczówki.
    smoothed = gaussian_filter(gray, kernel_size=7, sigma=smoothing_sigma)

    # Średnia jasność w pierścieniach o danym promieniu, liczona z
    # czterech głównych kierunków (lewo, prawo, góra, dół). Zewnętrzna
    # granica tęczówki to lokalizacja największego skoku tej średniej
    # (przejście z ciemnej tęczówki do jasnej twardówki).
    h, w = smoothed.shape
    max_distance = int(min(h, w) / 2)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    profiles = []
    for d in directions:
        profiles.append(_radial_profile(smoothed, pupil.cx, pupil.cy,
                                        d, max_distance))
    stacked = np.vstack(profiles)
    mean_profile = np.nanmean(stacked, axis=0)
    # uzupełniamy NaN-y wartością z wnętrza
    if np.isnan(mean_profile).any():
        valid = ~np.isnan(mean_profile)
        mean_profile[~valid] = np.interp(
            np.flatnonzero(~valid),
            np.flatnonzero(valid),
            mean_profile[valid]
        ) if valid.any() else 0.0

    # gradient radialny
    gradient = np.diff(mean_profile)

    # ograniczamy zakres poszukiwania do przedziału
    # [min_radius_factor * r_p,  max_radius_factor * r_p]
    r_min = max(int(min_radius_factor * pupil.radius), pupil.radius + 3)
    r_max = min(int(max_radius_factor * pupil.radius), gradient.size)
    r_max = max(r_max, r_min + 1)

    # Tęczówka -> twardówka: wzrost jasności, więc szukamy maksimum gradientu.
    search = gradient[r_min:r_max]
    if search.size == 0 or np.all(np.isnan(search)):
        radius = int(min_radius_factor * pupil.radius)
    else:
        radius = r_min + int(np.argmax(search))

    # Środek tęczówki utożsamiamy ze środkiem źrenicy (algorytm Daugmana
    # zakłada współśrodkowość tych okręgów - patrz książka, rys. 6.12a).
    return IrisResult(cx=pupil.cx, cy=pupil.cy, radius=radius,
                      threshold=int(round(threshold)))
