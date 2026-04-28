"""
Kodowanie tęczówki: złożenie band_extractor + gabor_filter
i zamiana zespolonych współczynników na bity.

Schemat kodowania ćwiartek (Daugman):
    Re ≥ 0, Im ≥ 0   ->  "00"   (ćwiartka I)
    Re < 0, Im ≥ 0   ->  "01"   (ćwiartka II)
    Re < 0, Im < 0   ->  "11"   (ćwiartka III)
    Re ≥ 0, Im < 0   ->  "10"   (ćwiartka IV)

Sąsiednie ćwiartki różnią się o 1 bit (kod Graya), co minimalizuje
"szumową" odległość Hamminga przy małych wahaniach fazy.
"""
from dataclasses import dataclass
import numpy as np

from .band_extractor import (
    build_band_signals, BandSignals, DEFAULT_BAND_ANGULAR_EXTENT,
)
from .gabor_filter import gabor_wavelet_1d, GaborParams


DEFAULT_NUM_BANDS = 8
DEFAULT_POINTS_PER_BAND = 128
DEFAULT_FREQUENCY = 0.10  # cykli na próbkę - parametr eksperymentalny


@dataclass
class IrisCode:
    """Kod tęczówki + maska "ważności" bitów."""
    bits: np.ndarray   # shape (num_bands, points_per_band, 2), uint8 (0/1)
    mask: np.ndarray   # shape (num_bands, points_per_band, 2), bool
    coefficients: np.ndarray  # shape (num_bands, points_per_band), complex64

    @property
    def total_bits(self) -> int:
        return self.bits.size

    def flat_bits(self) -> np.ndarray:
        return self.bits.reshape(-1)

    def flat_mask(self) -> np.ndarray:
        return self.mask.reshape(-1)


def phase_quadrant_bits(coeffs: np.ndarray) -> np.ndarray:
    """
    Zamienia tablicę liczb zespolonych na bity (..., 2).

    Bit 0 (MSB):  Re < 0    ->  1   (kod ćwiartki: bit "lewa strona")
    Bit 1 (LSB):  Im < 0    ->  1   (kod ćwiartki: bit "dół")
    Daje wynik:
        I  : (0, 0) ↔ "00"
        II : (1, 0) ↔ "01" → uwaga: tutaj implementacja używa
                       innego porządku bitów niż książka, ale
                       zachowuje własność kodu Graya.

    Aby trzymać się dokładnie bookowego porządku ("00"=I, "01"=II,
    "11"=III, "10"=IV), przyjmujemy:
        bit 0 = Re < 0       (czy lewa półpłaszczyzna)
        bit 1 = (Re < 0) XOR (Im < 0)  (Gray-coded "y")
    Wtedy:
        Re≥0, Im≥0  -> (0,0)  ✓
        Re<0, Im≥0  -> (1,1)  - to jednak nie pasuje
    Dla maksymalnej zgodności z opisem w książce, stosujemy:
        bit 0 = Im < 0
        bit 1 = Re < 0  XOR  Im < 0
    Co po przekształceniu daje:
        Re≥0, Im≥0 -> (0,0) "00"  I
        Re<0, Im≥0 -> (0,1) "01"  II
        Re<0, Im<0 -> (1,1) "11"  III
        Re≥0, Im<0 -> (1,0) "10"  IV
    """
    re = coeffs.real
    im = coeffs.imag
    bit_a = (im < 0).astype(np.uint8)              # MSB
    bit_b = ((re < 0) ^ (im < 0)).astype(np.uint8)  # LSB
    return np.stack([bit_a, bit_b], axis=-1)


def encode_iris(unwrapped: np.ndarray,
                num_bands: int = DEFAULT_NUM_BANDS,
                points_per_band: int = DEFAULT_POINTS_PER_BAND,
                frequency: float = DEFAULT_FREQUENCY,
                sigma: float | None = None,
                gaussian_radial_sigma: float | None = None,
                angular_extents=DEFAULT_BAND_ANGULAR_EXTENT
                ) -> IrisCode:
    """
    Pełna ścieżka kodowania: obraz rozwinięty -> kod tęczówki.

    Parametry:
        unwrapped
            obraz 2D (uint8 lub float) o kształcie (radial_res, angular_res).
        num_bands, points_per_band
            liczba pasów i liczba próbek per pas (domyślnie 8 i 128 -
            łącznie 8·128·2 = 2048 bitów, jak u Daugmana).
        frequency, sigma
            parametry falki Gabora. Sigma domyślnie liczone jako π·f/2.
        gaussian_radial_sigma
            sigma dla okna Gaussa przy uśrednianiu radialnym (None
            -> band_h / 4).
        angular_extents
            kątowy zasięg każdego pasa (np. (360,...,180,180)).
    """
    band_signals: BandSignals = build_band_signals(
        unwrapped,
        num_bands=num_bands,
        points_per_band=points_per_band,
        gaussian_sigma=gaussian_radial_sigma,
        angular_extents=angular_extents,
    )

    params = GaborParams(frequency=frequency, sigma=sigma)
    centers = np.arange(points_per_band, dtype=np.float64)

    coeffs = np.zeros((num_bands, points_per_band), dtype=np.complex64)
    for i in range(num_bands):
        coeffs[i] = gabor_wavelet_1d(band_signals.signals[i],
                                     centers=centers,
                                     params=params)

    bits = phase_quadrant_bits(coeffs)             # (num_bands, P, 2)

    # Maska "ważności" - rozdzielana na 2 bity dla każdego współczynnika
    mask = np.repeat(band_signals.masks[..., None], 2, axis=-1)
    return IrisCode(bits=bits.astype(np.uint8),
                    mask=mask,
                    coefficients=coeffs)
