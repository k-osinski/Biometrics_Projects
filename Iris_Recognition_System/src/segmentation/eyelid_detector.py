"""
Detekcja powiek - górnej i dolnej - jako parabole y = a·x² + b·x + c.

Idea (klasyczne podejście Daugmanowskie):
    1. Wygładzić obraz Gaussem.
    2. Operatorem Sobela liczymy gradient pionowy (powieka jest w
       przybliżeniu poziomą krawędzią, więc gradient w kierunku y
       jest na niej duży).
    3. W okolicy *nad* środkiem źrenicy (region górnej powieki) i
       *pod* nim (region dolnej powieki), dla każdej kolumny x
       szukamy y o największej wartości |∂I/∂y|. Bierzemy też pod
       uwagę magnitudę - jeśli krawędź jest słaba, kandydata
       odrzucamy.
    4. Robimy *RANSAC-podobny* fit paraboli (kilka iteracji: losuj
       3 punkty, fituj, policz inliers, zapamiętaj najlepszy model).
    5. Końcowy fit: least squares na zbiorze inliers.

Wynik:
    EyelidResult(upper, lower) - każdy z nich jest obiektem `Parabola`
    (lub None, jeśli detekcja się nie powiodła).

Uwagi:
    * Detekcja działa wyłącznie w obrębie tęczówki (granica zewnętrzna)
      - punkty poza nią są ignorowane.
    * Parametry detektora są konfigurowalne; domyślne wartości
      sprawdzają się na MMU.
    * Detekcja jest opcjonalna: w pipeline można ją wyłączyć i wtedy
      maskowanie powiek odbywa się tylko statyczną maską 360/226/180°.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..core.basic_operations import to_grayscale
from ..core.filters import gaussian_filter
from ..core.convolution import convolve2d
from .pupil_detector import PupilResult
from .iris_detector import IrisResult


@dataclass
class Parabola:
    """y(x) = a·x² + b·x + c"""
    a: float
    b: float
    c: float

    def __call__(self, x: np.ndarray | float) -> np.ndarray | float:
        x_arr = np.asarray(x, dtype=np.float64)
        return self.a * x_arr * x_arr + self.b * x_arr + self.c

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.a, self.b, self.c)


@dataclass
class EyelidResult:
    """Parabole górnej i dolnej powieki (mogą być None, jeśli nie wykryto)."""
    upper: Parabola | None
    lower: Parabola | None

    @property
    def detected(self) -> bool:
        return self.upper is not None or self.lower is not None


def _sobel_y(gray: np.ndarray) -> np.ndarray:
    """Pionowa składowa gradientu (operator Sobela)."""
    ky = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=np.float64)
    return convolve2d(gray.astype(np.float64), ky)


def _fit_parabola_lstsq(xs: np.ndarray, ys: np.ndarray) -> Parabola | None:
    """LSQ fit y = a·x² + b·x + c. Zwraca None, jeśli za mało punktów."""
    if xs.size < 3:
        return None
    A = np.vstack([xs * xs, xs, np.ones_like(xs)]).T
    try:
        coeffs, *_ = np.linalg.lstsq(A.astype(np.float64),
                                     ys.astype(np.float64), rcond=None)
    except np.linalg.LinAlgError:
        return None
    return Parabola(a=float(coeffs[0]),
                    b=float(coeffs[1]),
                    c=float(coeffs[2]))


def _ransac_parabola(xs: np.ndarray, ys: np.ndarray,
                     iterations: int = 300,
                     inlier_dist: float = 4.0,
                     rng: np.random.Generator | None = None
                     ) -> tuple[Parabola | None, np.ndarray]:
    """
    RANSAC: w każdej iteracji losuje 3 punkty, fituje parabolę, liczy
    inliers (|y - parabola(x)| < inlier_dist). Zwraca najlepszy fit
    + maskę inlierów.

    Uwaga: 300 iteracji jest dość, żeby z prawdopodobieństwem >0.99
    wylosować przynajmniej raz trójkę z samych inlierów nawet przy
    50% udziale outlierów - z zapasem.
    """
    n = xs.size
    if n < 5:
        para = _fit_parabola_lstsq(xs, ys)
        return para, np.ones(n, dtype=bool)

    if rng is None:
        rng = np.random.default_rng(seed=42)

    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0

    for _ in range(iterations):
        idx = rng.choice(n, size=3, replace=False)
        para = _fit_parabola_lstsq(xs[idx], ys[idx])
        if para is None:
            continue
        residuals = np.abs(ys - para(xs))
        inliers = residuals < inlier_dist
        cnt = int(inliers.sum())
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers

    if best_count < 3:
        return None, best_inliers

    final = _fit_parabola_lstsq(xs[best_inliers], ys[best_inliers])
    return final, best_inliers


def _detect_one_eyelid(grad_y: np.ndarray,
                       cx: int, cy: int,
                       iris_radius: int,
                       side: str,
                       min_gradient: float,
                       horizontal_extent: float = 0.9,
                       vertical_inner_skip: float = 0.3,
                       vertical_outer_extent: float = 1.0,
                       inlier_dist: float = 4.0,
                       min_inlier_fraction: float = 0.2,
                       ) -> Parabola | None:
    """
    Detekcja jednej powieki (górnej lub dolnej).

    side : "upper" lub "lower"

    Trzy mechanizmy odporności (Daugmanowski duch + praktyka):

    1. **Gradient ze znakiem.** Szukamy konkretnej *orientacji*
       krawędzi powieki, nie samego modułu gradientu:
         - górna powieka: schodząc w dół przechodzimy ze skóry (jasna)
           w pas rzęs/powieki (ciemne) -> intensywność maleje, więc
           ∂I/∂y jest **ujemny**. Argmin gradientu po kolumnie
           wskazuje krawędź skóra/rzęsy (czyli faktyczną linię powieki),
           a nie sam pas rzęs.
         - dolna powieka: schodząc w dół z rzęs (ciemne) w skórę
           (jasna) -> ∂I/∂y jest **dodatni**. Argmax po kolumnie.

    2. **Zawężone okno pionowe.** Szukamy w pierścieniu
       y ∈ [cy - r, cy - inner_skip · r]   (lub odpowiednio dla dołu),
       czyli wykluczamy *wnętrze* tęczówki - tam są pojedyncze rzęsy
       i melanina, które potrafią dawać duże |gradient|.

    3. **Próg jakości fitu (frakcja inlierów).** Po RANSAC-u sprawdzamy
       ile kandydackich kolumn jest inlierami. Jeśli mniej niż
       min_inlier_fraction (np. 20%) - uznajemy że detekcja nie udana
       i zwracamy None. Wtedy maska kątowa 360°/226°/180° z książki
       sama bierze rolę zabezpieczenia.
    """
    h, w = grad_y.shape

    half_w = int(iris_radius * horizontal_extent)
    x_lo = max(0, cx - half_w)
    x_hi = min(w, cx + half_w + 1)

    inner_offset = int(iris_radius * vertical_inner_skip)
    outer_offset = int(iris_radius * vertical_outer_extent)
    if side == "upper":
        y_lo = max(0, cy - outer_offset)
        y_hi = max(0, cy - inner_offset)
    else:
        y_lo = min(h, cy + inner_offset)
        y_hi = min(h, cy + outer_offset + 1)

    if y_hi - y_lo < 3 or x_hi - x_lo < 5:
        return None

    region = grad_y[y_lo:y_hi, x_lo:x_hi]

    if side == "upper":
        col_extreme = region.min(axis=0)
        col_argextreme = region.argmin(axis=0)
        valid_cols = col_extreme <= -min_gradient
    else:
        col_extreme = region.max(axis=0)
        col_argextreme = region.argmax(axis=0)
        valid_cols = col_extreme >= min_gradient

    if int(valid_cols.sum()) < 5:
        return None

    xs_local = np.arange(x_hi - x_lo)[valid_cols]
    ys_local = col_argextreme[valid_cols]

    xs = (xs_local + x_lo).astype(np.float64)
    ys = (ys_local + y_lo).astype(np.float64)

    para, inliers = _ransac_parabola(xs, ys, inlier_dist=inlier_dist)
    if para is None:
        return None

    inlier_frac = float(inliers.sum()) / max(xs.size, 1)
    if inlier_frac < min_inlier_fraction:
        return None

    return para


def detect_eyelids(gray: np.ndarray,
                   pupil: PupilResult,
                   iris: IrisResult,
                   smoothing_sigma: float = 2.0,
                   min_gradient: float = 80.0,
                   inlier_dist: float = 4.0,
                   min_inlier_fraction: float = 0.2,
                   vertical_inner_skip: float = 0.3,
                   detect_upper: bool = True,
                   detect_lower: bool = True) -> EyelidResult:
    """
    Detekcja parabolicznych powiek (górnej i dolnej) na obrazie oka.

    Argumenty:
        gray
            obraz wejściowy (RGB lub L).
        pupil, iris
            wyniki segmentacji.
        smoothing_sigma
            sigma w preprocesującym filtrze Gaussa.
        min_gradient
            minimalna wartość |∂I/∂y|, żeby kolumna była brana pod
            uwagę jako kandydat na powiekę. Zbyt niska -> dużo
            outlierów; zbyt wysoka -> brakuje punktów do dopasowania.
        inlier_dist
            tolerancja w pikselach przy RANSAC-podobnym fit'cie.
        detect_upper, detect_lower
            wyłączniki niezależne dla każdej powieki.
    """
    if gray.ndim == 3:
        gray = to_grayscale(gray)

    smoothed = gaussian_filter(gray, kernel_size=7, sigma=smoothing_sigma)
    grad_y = _sobel_y(smoothed)

    upper = None
    lower = None
    if detect_upper:
        upper = _detect_one_eyelid(grad_y, pupil.cx, pupil.cy,
                                   iris.radius, side="upper",
                                   min_gradient=min_gradient,
                                   inlier_dist=inlier_dist,
                                   min_inlier_fraction=min_inlier_fraction,
                                   vertical_inner_skip=vertical_inner_skip)
    if detect_lower:
        lower = _detect_one_eyelid(grad_y, pupil.cx, pupil.cy,
                                   iris.radius, side="lower",
                                   min_gradient=min_gradient,
                                   inlier_dist=inlier_dist,
                                   min_inlier_fraction=min_inlier_fraction,
                                   vertical_inner_skip=vertical_inner_skip)

    return EyelidResult(upper=upper, lower=lower)


def eyelid_mask(eyelids: EyelidResult,
                xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Dla zbioru punktów (xs, ys) zwraca tablicę bool tej samej formy,
    True = punkt jest *poza* powiekami (czyli wewnątrz tęczówki).
    """
    valid = np.ones_like(xs, dtype=bool)
    if eyelids.upper is not None:
        valid &= ys > eyelids.upper(xs)
    if eyelids.lower is not None:
        valid &= ys < eyelids.lower(xs)
    return valid
