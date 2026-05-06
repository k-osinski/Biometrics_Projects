"""
Jednowymiarowa transformata falkowa Gabora.
Postać analityczna:
    G(x; x_k, σ, f) = exp(-(x - x_k)^2 / σ^2) · exp(-i · 2π · f · (x - x_k))
Współczynnik dekompozycji sygnału I(x) względem falki o pozycji x_k:
    c_k = Σ_{i=0}^{n-1}  I(x_i) · G(x_i; x_k, σ, f)
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class GaborParams:
    """Parametry falki Gabora używane podczas kodowania."""
    frequency: float            # f - częstotliwość falki
    sigma: float | None = None  # jeśli None, liczone ze wzoru σ = π·f/2

    @property
    def effective_sigma(self) -> float:
        if self.sigma is not None:
            return self.sigma
        return 0.5 * np.pi * self.frequency


def gabor_wavelet_1d(signal: np.ndarray,
                     centers: np.ndarray,
                     params: GaborParams) -> np.ndarray:
    """
    Liczy zbiór współczynników dekompozycji sygnału 1D na bazie falek Gabora.
    Argumenty:
        signal   - 1D ndarray (real-valued).
        centers  - 1D ndarray, współrzędne x_k (w jednostkach próbek).
        params   - GaborParams.
    Zwraca:
        ndarray complex64 o długości len(centers) - jeden współczynnik
        per pozycja x_k.
    """
    sigma = params.effective_sigma
    if sigma <= 0:
        raise ValueError("σ musi być dodatnia.")

    n = signal.size
    xs = np.arange(n, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)

    dx = xs[None, :] - centers[:, None]

    envelope = np.exp(-(dx ** 2) / (sigma ** 2))
    carrier = np.exp(-1j * 2.0 * np.pi * params.frequency * dx)

    kernel = envelope * carrier
    coeffs = (signal.astype(np.float64))[None, :] * kernel
    return np.sum(coeffs, axis=1).astype(np.complex64)
