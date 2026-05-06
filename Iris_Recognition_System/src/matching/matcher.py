"""
Porównywanie kodów tęczówki.

Miara: znormalizowana odległość Hamminga (HD).
    HD(A, B) = sum( a_k XOR b_k ) / N
gdzie sumujemy tylko po pozycjach uznanych za "ważne" w obu kodach
(tj. mask_A[k] AND mask_B[k] = True).

Aby skompensować rotację oka (i tym samym przesunięcie kąta), kod B
jest porównywany z kilkoma wersjami przesuniętymi w lewo/prawo
(o ±k punktów kątowych). Zwracana jest najmniejsza otrzymana HD.
"""
from __future__ import annotations
import numpy as np
from ..encoding.iris_code import IrisCode


def _xor_count(a_bits: np.ndarray, b_bits: np.ndarray,
               valid_mask: np.ndarray) -> float:
    """
    Liczy znormalizowaną odległość Hamminga na "ważnych" pozycjach.
    a_bits, b_bits, valid_mask muszą mieć ten sam kształt.
    """
    diff = (a_bits ^ b_bits) & valid_mask.astype(np.uint8)
    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        return 1.0
    return float(diff.sum()) / valid_count


def hamming_distance(code_a: IrisCode, code_b: IrisCode) -> float:
    """Znormalizowana HD bez rotacji (dla identycznego ułożenia)."""
    valid = code_a.mask & code_b.mask
    return _xor_count(code_a.bits.astype(np.uint8),
                      code_b.bits.astype(np.uint8),
                      valid)


def _shift_columns(bits: np.ndarray, mask: np.ndarray,
                   shift: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Cyklicznie przesuwa kod (i jego maskę) wzdłuż osi kątowej (axis=1).
    bits/mask kształtu (num_bands, points_per_band, 2) - przesuwamy
    drugą oś.
    """
    return np.roll(bits, shift, axis=1), np.roll(mask, shift, axis=1)


def hamming_distance_with_rotation(code_a: IrisCode, code_b: IrisCode,
                                   max_shift: int = 8
                                   ) -> tuple[float, int]:
    """
    HD z kompensacją obrotu - najlepsze (minimalne) HD wśród
    przesunięć (-max_shift, ..., +max_shift).
    Zwraca (best_hd, best_shift).
    """
    a_bits = code_a.bits.astype(np.uint8)
    a_mask = code_a.mask
    b_bits = code_b.bits.astype(np.uint8)
    b_mask = code_b.mask

    best_hd = 1.0
    best_shift = 0
    for shift in range(-max_shift, max_shift + 1):
        bb, bm = _shift_columns(b_bits, b_mask, shift)
        valid = a_mask & bm
        hd = _xor_count(a_bits, bb, valid)
        if hd < best_hd:
            best_hd = hd
            best_shift = shift
    return best_hd, best_shift


def compare(code_a: IrisCode, code_b: IrisCode,
            max_shift: int = 8) -> dict:
    """
    Wrapper do GUI - zwraca komplet informacji.
    """
    hd_zero = hamming_distance(code_a, code_b)
    hd_min, shift = hamming_distance_with_rotation(code_a, code_b,
                                                   max_shift=max_shift)
    return {
        "hd_no_rotation": hd_zero,
        "hd_min": hd_min,
        "best_shift": shift,
    }
