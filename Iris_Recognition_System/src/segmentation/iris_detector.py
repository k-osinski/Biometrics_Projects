"""
Detekcja zewnętrznej granicy tęczówki.

Algorytm:
    1. Wygładzamy obraz filtrem Gaussa, by stłumić rzęsy i szum.
    2. Dla każdego promienia kandydackiego r ∈ [r_min, r_max] liczymy
       *średnią jasność po okręgu* o środku w środku źrenicy.
    3. Punktem granicznym jest największe r, dla którego średnia
       po okręgu jest jeszcze poniżej progu P_I = P / X_I (czyli
       okrąg leży jeszcze w obszarze ciemnej tęczówki).
       Jeśli żaden okrąg nie spełnia tego warunku, w ramach
       fallbacku bierzemy r o największym gradiencie radialnym.

Dzięki temu:
    * X_I jest faktycznym parametrem detektora,
    * używamy projekcji-jak (uśrednianie po okręgu),
    * cały detektor jest dwuwymiarowy, a nie jednoliniowy
      (mniej wrażliwy na rzęsy / refleksy).
"""
from dataclasses import dataclass
import numpy as np

from ..core.basic_operations import to_grayscale
from ..core.filters import gaussian_filter
from .pupil_detector import (
    PupilResult, base_mean_threshold,
)


@dataclass
class IrisResult:
    """Wynik detekcji zewnętrznej granicy tęczówki."""
    cx: int
    cy: int
    radius: int
    threshold: int           # P_I = P / X_I (zaokrąglony)
    method_used: str = ""    # "threshold" lub "gradient" (fallback)


def _circle_mean_intensity(img: np.ndarray, cx: int, cy: int,
                           radius: int) -> float:
    """
    Średnia jasność próbkowana wzdłuż okręgu o promieniu `radius`.
    Punkty wypadające poza obraz są pomijane. Sampling bilinearny.
    """
    h, w = img.shape
    n = max(64, int(2 * np.pi * radius))
    thetas = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xs = cx + radius * np.cos(thetas)
    ys = cy + radius * np.sin(thetas)

    valid = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
    if not valid.any():
        return float("nan")

    xs, ys = xs[valid], ys[valid]
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    dx = xs - x0
    dy = ys - y0

    Ia = img[y0, x0].astype(np.float64)
    Ib = img[y0, x1].astype(np.float64)
    Ic = img[y1, x0].astype(np.float64)
    Id = img[y1, x1].astype(np.float64)
    top = Ia * (1 - dx) + Ib * dx
    bot = Ic * (1 - dx) + Id * dx
    return float(np.mean(top * (1 - dy) + bot * dy))


def detect_iris(gray: np.ndarray,
                pupil: PupilResult,
                x_i: float = 1.15,
                smoothing_sigma: float = 2.0,
                min_radius_factor: float = 1.5,
                max_radius_factor: float = 4.5) -> IrisResult:
    """
    Detekcja granicy tęczówki.

    Parametry:
        gray
            obraz wejściowy (RGB lub L).
        pupil
            wynik detekcji źrenicy.
        x_i
            mianownik progu P_I = P / X_I. Zwykle 0.9 - 1.5;
            mniejsze wartości -> wyższy próg -> szerszy zakres "ciemnych"
            promieni; większe wartości -> niższy próg -> ostrzejsza
            tęczówka (mniej szansy "wpaść" w sclerę).
        smoothing_sigma
            sigma w filtrze Gaussa stosowanym przed pomiarami radialnymi.
        min_radius_factor, max_radius_factor
            ograniczenia na promień tęczówki w jednostkach promienia
            źrenicy. Domyślnie [1.5·r_p, 4.5·r_p].
    """
    if gray.ndim == 3:
        gray = to_grayscale(gray)

    base = base_mean_threshold(gray)
    threshold = base / max(x_i, 1e-6)

    smoothed = gaussian_filter(gray, kernel_size=7, sigma=smoothing_sigma)
    h, w = smoothed.shape
    cx, cy = pupil.cx, pupil.cy
    rp = max(pupil.radius, 1)

    radial_room = min(cx, w - 1 - cx, cy, h - 1 - cy)
    r_min = max(int(min_radius_factor * rp), rp + 3)
    r_max = min(int(max_radius_factor * rp), int(radial_room))
    if r_max <= r_min + 1:
        return IrisResult(cx=cx, cy=cy, radius=max(r_min, rp + 3),
                          threshold=int(round(threshold)),
                          method_used="fallback")

    radii = np.arange(r_min, r_max + 1, dtype=np.int32)
    means = np.array([_circle_mean_intensity(smoothed, cx, cy, int(r))
                      for r in radii], dtype=np.float64)

    if np.isnan(means).any():
        valid = ~np.isnan(means)
        if not valid.any():
            return IrisResult(cx=cx, cy=cy, radius=r_min,
                              threshold=int(round(threshold)),
                              method_used="fallback")
        means[~valid] = np.interp(np.flatnonzero(~valid),
                                  np.flatnonzero(valid),
                                  means[valid])

    # 1) Metoda progowa: największe r, dla którego okrąg jest jeszcze "ciemny"
    below = means < threshold
    if below.any():
        contiguous_end = 0
        for k in range(below.size):
            if below[k]:
                contiguous_end = k
            else:
                break
        radius = int(radii[contiguous_end])
        method = "threshold"
    else:
        # 2) Fallback: największy gradient (przejście tęczówka -> twardówka).
        gradient = np.diff(means)
        if gradient.size == 0:
            radius = int(radii[0])
        else:
            radius = int(radii[int(np.argmax(gradient))])
        method = "gradient"

    return IrisResult(cx=cx, cy=cy, radius=radius,
                      threshold=int(round(threshold)),
                      method_used=method)
