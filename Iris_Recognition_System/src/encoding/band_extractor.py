"""
Podział rozwiniętej tęczówki na pasy radialne i zamiana każdego pasa
na wektor 1D (uśrednianie radialne z oknem Gaussa).

Założenia (rys. 6.14 w książce + treść Projektu 2):
    - rozwinięta tęczówka ma kształt (radial_res, angular_res),
      gdzie wiersze odpowiadają coraz większemu promieniowi
      (od źrenicy do twardówki),
    - dzielimy ją na N pasów (domyślnie 8) o równej wysokości,
    - dla każdego pasa wykonujemy uśrednianie kolumnowe z wagami
      Gaussa (preferujemy środkowy wiersz pasa),
    - wynik resamplujemy do P punktów (domyślnie 128) - "rozwiniętą"
      kolumnę dla każdego pasa.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class BandSignals:
    """Wynik konwersji rozwiniętej tęczówki na 8 sygnałów 1D."""
    signals: np.ndarray  # shape (num_bands, points_per_band), float64
    masks: np.ndarray    # shape (num_bands, points_per_band), bool
    band_height: int


def _gaussian_weights(length: int, sigma: float | None = None) -> np.ndarray:
    """Wagi Gaussa scentrowane na środku okna o danej długości."""
    if length <= 1:
        return np.ones(length, dtype=np.float64)
    if sigma is None:
        sigma = max(length / 4.0, 1.0)
    xs = np.arange(length) - (length - 1) / 2.0
    w = np.exp(-(xs ** 2) / (2.0 * sigma ** 2))
    w /= w.sum()
    return w


def gaussian_radial_average(band: np.ndarray,
                            sigma: float | None = None,
                            mask: np.ndarray | None = None) -> np.ndarray:
    """
    Uśrednianie kolumnowe (po wierszach) z wagami Gaussa.

    band   - shape (band_height, angular_res)
    mask   - opcjonalna maska bool tej samej formy co band; True = piksel
             ważny. Jeśli podana, uśrednianie idzie tylko po ważnych
             pikselach (wagi Gaussa są przeskalowywane per kolumna).
    return - shape (angular_res,) - jedna liczba per kolumna.
    """
    h = band.shape[0]
    w = _gaussian_weights(h, sigma=sigma)
    if mask is None:
        return (band.astype(np.float64) * w[:, None]).sum(axis=0)

    valid = mask.astype(np.float64)
    weighted = band.astype(np.float64) * w[:, None] * valid
    norm = (w[:, None] * valid).sum(axis=0)
    eps = 1e-9
    return weighted.sum(axis=0) / np.maximum(norm, eps)


def _resample_1d(signal: np.ndarray, new_length: int) -> np.ndarray:
    """Liniowa interpolacja sygnału 1D do zadanej długości."""
    if signal.size == new_length:
        return signal.copy()
    xs_old = np.linspace(0.0, 1.0, signal.size)
    xs_new = np.linspace(0.0, 1.0, new_length)
    return np.interp(xs_new, xs_old, signal)


def _angular_mask(num_points: int, angular_extent_deg: float,
                  start_offset_deg: float = 0.0) -> np.ndarray:
    """
    Tworzy maskę kątową o zadanej szerokości (np. 360°, 226°, 180°).
    Maska jest "centrowana" wokół 90° (kierunek "do góry" w obrazie
    rozwiniętym), tj. wycina symetrycznie fragmenty z lewej i prawej
    strony obrazu, gdzie bywają powieki/rzęsy.

    angular_extent_deg = 360 oznacza "wszystkie punkty są ważne".
    """
    extent = max(0.0, min(360.0, angular_extent_deg))
    if extent >= 360.0:
        return np.ones(num_points, dtype=bool)
    valid_count = int(round(num_points * (extent / 360.0)))
    drop = num_points - valid_count
    left_drop = drop // 2
    right_drop = drop - left_drop
    mask = np.ones(num_points, dtype=bool)
    if left_drop > 0:
        mask[:left_drop] = False
    if right_drop > 0:
        mask[-right_drop:] = False
    if start_offset_deg:
        shift = int(round(num_points * (start_offset_deg / 360.0)))
        mask = np.roll(mask, shift)
    return mask


# domyślny układ ważnych obszarów dla 8 pasów (od wewnętrznego do zewnętrznego)
# zgodny z opisem z książki: pasy 0-3 - 360°, 4-5 - 226°, 6-7 - 180°.
DEFAULT_BAND_ANGULAR_EXTENT = (360.0, 360.0, 360.0, 360.0,
                               226.0, 226.0, 180.0, 180.0)


def split_into_bands(unwrapped: np.ndarray,
                     num_bands: int = 8) -> list[np.ndarray]:
    """
    Dzieli rozwinięty obraz tęczówki na poziome pasy o równej wysokości.

    Zwraca listę pasów (każdy o kształcie (band_h, angular_res)).
    Wiersze pominięte przy podziale (gdy radial_res nie jest podzielne
    przez num_bands) są równomiernie odrzucane z góry/dołu.
    """
    if unwrapped.ndim != 2:
        raise ValueError("Oczekiwano obrazu 2D (rozwiniętej tęczówki).")
    h, _ = unwrapped.shape
    band_h = h // num_bands
    if band_h == 0:
        raise ValueError("Za niski obraz, by podzielić na zadaną liczbę pasów.")
    extra = h - band_h * num_bands
    top = extra // 2
    bands = []
    for i in range(num_bands):
        y0 = top + i * band_h
        y1 = y0 + band_h
        bands.append(unwrapped[y0:y1])
    return bands


def _resample_mask(mask_1d: np.ndarray, new_length: int) -> np.ndarray:
    """Resampling maski bool do zadanej długości."""
    if mask_1d.size == new_length:
        return mask_1d.astype(bool)
    xs_old = np.linspace(0.0, 1.0, mask_1d.size)
    xs_new = np.linspace(0.0, 1.0, new_length)
    return np.interp(xs_new, xs_old, mask_1d.astype(np.float64)) >= 0.5


def build_band_signals(unwrapped: np.ndarray,
                       num_bands: int = 8,
                       points_per_band: int = 128,
                       gaussian_sigma: float | None = None,
                       angular_extents=DEFAULT_BAND_ANGULAR_EXTENT,
                       extra_mask: np.ndarray | None = None,
                       min_valid_fraction: float = 0.5,
                       ) -> BandSignals:
    """
    Cała ścieżka 6.14 (książka): obrazek -> 8 sygnałów 1D + maski.

    Argumenty:
        unwrapped
            obraz rozwiniętej tęczówki (radial_res, angular_res).
        num_bands
            liczba pasów (domyślnie 8).
        points_per_band
            liczba próbek per pas (domyślnie 128).
        gaussian_sigma
            sigma w uśrednianiu radialnym - domyślnie band_h/4.
        angular_extents
            statyczna maska kątowa per pas wg DEFAULT_BAND_ANGULAR_EXTENT
            (360/360/360/360/226/226/180/180 z książki).
        extra_mask
            opcjonalna maska 2D (radial_res, angular_res), True = piksel
            ważny. Używana m.in. do nakładania detekcji powiek z
            `unwrap.mask`. Łączona AND z maską angular_extents.
        min_valid_fraction
            kolumna wewnątrz pasa jest oznaczana jako ważna (po stronie
            extra_mask) tylko gdy >= min_valid_fraction pikseli w tej
            kolumnie pasa jest ważne.
    """
    bands = split_into_bands(unwrapped, num_bands=num_bands)
    band_h, angular_res = bands[0].shape

    if extra_mask is not None:
        if extra_mask.shape != unwrapped.shape:
            raise ValueError("extra_mask musi mieć ten sam kształt co unwrapped.")
        mask_bands = split_into_bands(extra_mask.astype(bool),
                                      num_bands=num_bands)
    else:
        mask_bands = [None] * num_bands

    signals = np.zeros((num_bands, points_per_band), dtype=np.float64)
    masks = np.zeros((num_bands, points_per_band), dtype=bool)

    if angular_extents is None:
        angular_extents = (360.0,) * num_bands

    for i, band in enumerate(bands):
        bm = mask_bands[i]
        avg = gaussian_radial_average(band, sigma=gaussian_sigma, mask=bm)
        signals[i] = _resample_1d(avg, points_per_band)

        # Statyczna maska kątowa wg książki.
        static_mask = _angular_mask(angular_res, angular_extents[i])

        # Maska "wystarczającej liczby ważnych pikseli" wg extra_mask.
        if bm is not None:
            col_valid_frac = bm.astype(np.float64).mean(axis=0)
            dynamic_mask = col_valid_frac >= min_valid_fraction
            full_mask = static_mask & dynamic_mask
        else:
            full_mask = static_mask

        # Resampling do liczby punktów per pas.
        masks[i] = _resample_mask(full_mask, points_per_band)

    return BandSignals(signals=signals, masks=masks, band_height=band_h)
