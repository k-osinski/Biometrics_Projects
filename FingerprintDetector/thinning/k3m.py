"""Algorytm scieniania K3M.

Algorytm K3M jest sekwencyjny i iteracyjny. Sklada sie z czesci iteracyjnej
(7 faz: 0..6) oraz dodatkowego przebiegu koncowego (A1pix) ktory redukuje
szkielet do szerokosci jednego piksela.

Wagi sasiadow s_i sa kodowane wedlug macierzy N:

        | 128   1   2 |
    N = |  64   0   4 |
        |  32  16   8 |

Suma wag czarnych sasiadow daje wartosc w(x,y) z zakresu 0..255. Wartosc ta
jest porownywana z odpowiednia tablica A_i okreslajaca czy piksel ma byc
usuniety w danej fazie.
"""
from __future__ import annotations

import numpy as np


# Tablice A_i
A0_SET = [
    3, 6, 7, 12, 14, 15, 24, 28, 30, 31, 48, 56, 60, 62, 63, 96, 112, 120,
    124, 126, 127, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207,
    223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252,
    253, 254,
]
A1_SET = [7, 14, 28, 56, 112, 131, 193, 224]
A2_SET = [
    7, 14, 15, 28, 30, 56, 60, 112, 120, 131, 135, 193, 195, 224, 225, 240,
]
A3_SET = [
    7, 14, 15, 28, 30, 31, 56, 60, 62, 112, 120, 124, 131, 135, 143, 193,
    195, 199, 224, 225, 227, 240, 241, 248,
]
A4_SET = [
    7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120, 124, 126, 131, 135, 143,
    159, 193, 195, 199, 207, 224, 225, 227, 231, 240, 241, 243, 248, 249, 252,
]
A5_SET = [
    7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120, 124, 126, 131, 135, 143,
    159, 191, 193, 195, 199, 207, 224, 225, 227, 231, 239, 240, 241, 243, 248,
    249, 251, 252, 254,
]
A1PIX_SET = [
    3, 6, 7, 12, 14, 15, 24, 28, 30, 31, 48, 56, 60, 62, 63, 96, 112, 120,
    124, 126, 127, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207,
    223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252,
    253, 254,
]


def _set_to_lut(values):
    lut = np.zeros(256, dtype=bool)
    lut[values] = True
    return lut


A0_LUT = _set_to_lut(A0_SET)
A_PHASE_LUTS = [_set_to_lut(s) for s in (A1_SET, A2_SET, A3_SET, A4_SET, A5_SET)]
A1PIX_LUT = _set_to_lut(A1PIX_SET)


def _compute_weights(img):
    """Wagi sasiedztwa dla calego obrazu (czysty numpy, padding + shift)."""
    p = np.pad(img.astype(np.int32), 1, mode="constant", constant_values=0)
    H, W = img.shape
    w = (
        128 * p[0:H, 0:W]
        + 1 * p[0:H, 1:W + 1]
        + 2 * p[0:H, 2:W + 2]
        + 64 * p[1:H + 1, 0:W]
        + 4 * p[1:H + 1, 2:W + 2]
        + 32 * p[2:H + 2, 0:W]
        + 16 * p[2:H + 2, 1:W + 1]
        + 8 * p[2:H + 2, 2:W + 2]
    )
    return w.astype(np.int32)


def _weight_at(img, r, c):
    H, W = img.shape
    if 0 < r < H - 1 and 0 < c < W - 1:
        return int(
            128 * img[r - 1, c - 1] + img[r - 1, c] + 2 * img[r - 1, c + 1]
            + 64 * img[r, c - 1] + 4 * img[r, c + 1]
            + 32 * img[r + 1, c - 1] + 16 * img[r + 1, c] + 8 * img[r + 1, c + 1]
        )
    w = 0
    offsets = (
        (-1, -1, 128), (-1, 0, 1), (-1, 1, 2),
        (0, -1, 64), (0, 1, 4),
        (1, -1, 32), (1, 0, 16), (1, 1, 8),
    )
    for dr, dc, wi in offsets:
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W:
            w += wi * int(img[rr, cc])
    return w


def _k3m_sequential(binary, max_iters=60):
    img = (binary > 0).astype(np.uint8)
    for _ in range(max_iters):
        modified = False
        weights = _compute_weights(img)
        is_border = (img == 1) & A0_LUT[weights]
        borders = np.argwhere(is_border)
        if borders.size == 0:
            break
        for phase_lut in A_PHASE_LUTS:
            for r, c in borders:
                if img[r, c] != 1:
                    continue
                w = _weight_at(img, r, c)
                if phase_lut[w]:
                    img[r, c] = 0
                    modified = True
        if not modified:
            break

    for _ in range(max_iters):
        modified = False
        weights = _compute_weights(img)
        is_border = (img == 1) & A0_LUT[weights]
        borders = np.argwhere(is_border)
        if borders.size == 0:
            break
        for r, c in borders:
            if img[r, c] != 1:
                continue
            w = _weight_at(img, r, c)
            if A1PIX_LUT[w]:
                img[r, c] = 0
                modified = True
        if not modified:
            break
    return img


def _k3m_parallel(binary, max_iters=60):
    img = (binary > 0).astype(np.uint8)
    for _ in range(max_iters):
        modified = False
        for phase_lut in A_PHASE_LUTS:
            weights = _compute_weights(img)
            is_border = (img == 1) & A0_LUT[weights]
            to_delete = is_border & phase_lut[weights]
            if to_delete.any():
                img[to_delete] = 0
                modified = True
        if not modified:
            break

    for _ in range(max_iters):
        weights = _compute_weights(img)
        is_border = (img == 1) & A0_LUT[weights]
        to_delete = is_border & A1PIX_LUT[weights]
        if not to_delete.any():
            break
        img[to_delete] = 0
    return img


def k3m_thin(binary, mode="sequential", max_iters=60):
    """Scienianie obrazu binarnego algorytmem K3M.

    Parametry:
    ----------
    binary : ndarray
        Obraz binarny {0, 1} (uint8). 1 = obiekt.
    mode : str
        'sequential' (wg PDF) lub 'parallel' (szybszy).
    max_iters : int
        Maksymalna liczba iteracji petli zewnetrznej.
    """
    if mode not in {"sequential", "parallel"}:
        raise ValueError("mode must be 'sequential' or 'parallel'")
    if mode == "sequential":
        return _k3m_sequential(binary, max_iters)
    return _k3m_parallel(binary, max_iters)
