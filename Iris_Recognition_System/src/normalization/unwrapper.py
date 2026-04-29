"""
Rozwinięcie tęczówki (pierścienia) do prostokąta.

Stosujemy "rubber sheet model" Daugmana:
    dla każdej współrzędnej (r, θ), gdzie r ∈ [0, 1], θ ∈ [0, 2π),
    pobieramy piksel z punktu okręgu interpolowanego między
    granicą wewnętrzną (źrenica) a zewnętrzną (tęczówka):

        x(r, θ) = (1 - r) * x_p(θ) + r * x_i(θ)
        y(r, θ) = (1 - r) * y_p(θ) + r * y_i(θ)

    gdzie:
        x_p(θ) = pup_cx + pup_r * cos(θ)
        y_p(θ) = pup_cy + pup_r * sin(θ)
        x_i(θ) = iris_cx + iris_r * cos(θ)
        y_i(θ) = iris_cy + iris_r * sin(θ)

Zaletą tej parametryzacji jest niezależność opisu od:
    - rozmiaru źrenicy (kompensacja zmian oświetlenia),
    - położenia źrenicy względem tęczówki,
    - rozdzielczości akwizycji.

Implementacja używa interpolacji bilinearnej (ręcznej, bez OpenCV).
"""
from dataclasses import dataclass
import numpy as np

from ..core.basic_operations import to_grayscale
from ..segmentation.pupil_detector import PupilResult
from ..segmentation.iris_detector import IrisResult
from ..segmentation.eyelid_detector import EyelidResult, eyelid_mask


@dataclass
class UnwrapResult:
    """Rozwinięta tęczówka."""
    image: np.ndarray   # uint8, kształt (radial_res, angular_res)
    mask: np.ndarray    # bool, True = piksel "ważny" (wewnątrz obrazu)


def _bilinear_sample(img: np.ndarray,
                     xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolacja bilinearna w punktach (xs, ys).

    Zwraca:
        values - tablicę o tym samym kształcie co xs, dtype=float64,
        valid  - maskę, w której True oznacza, że punkt mieści się
                 w granicach obrazu.
    """
    h, w = img.shape
    valid = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)

    xs_c = np.clip(xs, 0, w - 1)
    ys_c = np.clip(ys, 0, h - 1)

    x0 = np.floor(xs_c).astype(np.int32)
    y0 = np.floor(ys_c).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    dx = xs_c - x0
    dy = ys_c - y0

    Ia = img[y0, x0].astype(np.float64)
    Ib = img[y0, x1].astype(np.float64)
    Ic = img[y1, x0].astype(np.float64)
    Id = img[y1, x1].astype(np.float64)

    top = Ia * (1 - dx) + Ib * dx
    bot = Ic * (1 - dx) + Id * dx
    return top * (1 - dy) + bot * dy, valid


def unwrap_iris(img: np.ndarray,
                pupil: PupilResult,
                iris: IrisResult,
                radial_res: int = 64,
                angular_res: int = 256,
                start_angle: float = 0.0,
                eyelids: EyelidResult | None = None) -> UnwrapResult:
    """
    Rozwija pierścień tęczówki do prostokąta.

    Parametry:
        img
            obraz wejściowy (RGB lub L).
        pupil, iris
            wyniki segmentacji (środki + promienie).
        radial_res
            wysokość rozwiniętego prostokąta (liczba próbek po promieniu);
            wiersz 0 leży tuż obok źrenicy, wiersz radial_res-1 leży tuż
            przy granicy tęczówka/twardówka.
        angular_res
            szerokość rozwiniętego prostokąta (liczba próbek po kącie).
        start_angle
            kąt początkowy (w radianach). Pozwala obrócić rozwinięcie -
            wykorzystywane przy kompensacji rotacji w fazie matchingu
            można też zrobić poprzez przesunięcie kodu, ale daje to
            elastyczność.

    Zwraca:
        UnwrapResult(image, mask).
    """
    if img.ndim == 3:
        gray = to_grayscale(img)
    else:
        gray = img

    # siatka kątów i promieni
    thetas = np.linspace(start_angle, start_angle + 2 * np.pi,
                         angular_res, endpoint=False)
    rs = np.linspace(0.0, 1.0, radial_res, endpoint=True)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # punkty na okręgu źrenicy i tęczówki dla każdego θ
    xp = pupil.cx + pupil.radius * cos_t   # shape (angular_res,)
    yp = pupil.cy + pupil.radius * sin_t
    xi = iris.cx + iris.radius * cos_t
    yi = iris.cy + iris.radius * sin_t

    # interpolacja w kierunku radialnym: macierze rs[r] * (xi - xp) itp.
    # broadcasting -> (radial_res, angular_res)
    R = rs[:, None]
    xs = (1 - R) * xp[None, :] + R * xi[None, :]
    ys = (1 - R) * yp[None, :] + R * yi[None, :]

    values, mask = _bilinear_sample(gray, xs, ys)
    image = np.clip(values, 0, 255).astype(np.uint8)

    # Maska powiek - jeśli wykryto parabole, oznacz piksele leżące za nimi
    # jako "nieważne" (False).
    if eyelids is not None and eyelids.detected:
        valid_eyelids = eyelid_mask(eyelids, xs, ys)
        mask = mask & valid_eyelids

    return UnwrapResult(image=image, mask=mask)
