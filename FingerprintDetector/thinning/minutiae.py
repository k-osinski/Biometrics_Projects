"""Detekcja minucji na szkielecie odcisku palca.

Klasyczna metoda Crossing Number (CN):

    CN(p) = 0.5 * sum_{i=1..8} |P_i - P_{i+1}|

  - CN = 1  -> zakonczenie (ending),
  - CN = 3  -> bifurkacja (rozwidlenie).

Dodatkowo modul oferuje 'deduplicate_minutiae' ktora usuwa skupiska
falszywych minucji z szumu (NMS + filtr gestwiny).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class Minutia:
    row: int
    col: int
    kind: str  # "ending" lub "bifurcation"


_NEIGHBOUR_OFFSETS = (
    (-1,  0), (-1,  1), ( 0,  1), ( 1,  1),
    ( 1,  0), ( 1, -1), ( 0, -1), (-1, -1),
)


def _crossing_number(skel, r, c):
    vals = []
    H, W = skel.shape
    for dr, dc in _NEIGHBOUR_OFFSETS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W:
            vals.append(int(skel[rr, cc]))
        else:
            vals.append(0)
    s = 0
    for i in range(8):
        s += abs(vals[i] - vals[(i + 1) % 8])
    return s // 2


def detect_minutiae(skel, mask=None, border=10):
    """Wyznacz minucje na szkielecie binarnym (CN=1 / CN=3)."""
    skel_b = (skel > 0).astype(np.uint8)
    H, W = skel_b.shape

    if mask is not None:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * border + 1, 2 * border + 1))
        valid_mask = cv2.erode(mask.astype(np.uint8), kernel)
    else:
        valid_mask = np.zeros((H, W), dtype=np.uint8)
        valid_mask[border:H - border, border:W - border] = 1

    minutiae = []
    ys, xs = np.where(skel_b == 1)
    for r, c in zip(ys, xs):
        if valid_mask[r, c] == 0:
            continue
        cn = _crossing_number(skel_b, r, c)
        if cn == 1:
            minutiae.append(Minutia(int(r), int(c), "ending"))
        elif cn == 3:
            minutiae.append(Minutia(int(r), int(c), "bifurcation"))
    return minutiae


def deduplicate_minutiae(minutiae, min_distance=8, density_threshold=3):
    """Usuwa minucje znajdujace sie blizej niz min_distance pikseli.

    Algorytm dwustopniowy:

    1. **Filtr gestwiny zakonczen** - jezeli zakonczenie ma co najmniej
       'density_threshold' innych zakonczen w promieniu 'min_distance',
       odrzucamy je (typowy szum z poszarpanych ridges).
       Ustaw density_threshold=0 aby wylaczyc.

    2. **Greedy Non-Maximum Suppression** - bifurkacje sa preferowane nad
       zakonczeniami. Dla pozostalych minucji pierwsza w kolejnosci zostaje,
       kolejne odrzucamy jezeli sa blizej niz min_distance od juz
       zatwierdzonej.

    Ustaw min_distance=0 aby wylaczyc deduplikacje.
    """
    if min_distance <= 0 or len(minutiae) <= 1:
        return list(minutiae)

    md2 = min_distance * min_distance

    # 1. Filtr gestwiny dla endpoints
    if density_threshold > 0:
        coords = np.array([(m.row, m.col) for m in minutiae], dtype=np.int32)
        kinds = np.array([m.kind for m in minutiae])
        keep = np.ones(len(minutiae), dtype=bool)
        for i, m in enumerate(minutiae):
            if m.kind != "ending":
                continue
            d2 = ((coords[:, 0] - m.row) ** 2 + (coords[:, 1] - m.col) ** 2)
            close_ends = ((d2 > 0) & (d2 <= md2) & (kinds == "ending")).sum()
            if close_ends >= density_threshold:
                keep[i] = False
        filtered = [m for m, k in zip(minutiae, keep) if k]
    else:
        filtered = list(minutiae)

    # 2. Greedy NMS - bifurkacje pierwsze
    filtered.sort(key=lambda m: 0 if m.kind == "bifurcation" else 1)
    accepted = []
    for m in filtered:
        ok = True
        for a in accepted:
            d2 = (m.row - a.row) ** 2 + (m.col - a.col) ** 2
            if d2 < md2:
                ok = False
                break
        if ok:
            accepted.append(m)
    return accepted


def draw_minutiae(skel, minutiae, background=None,
                   ending_color=(0, 0, 255),
                   bifurcation_color=(0, 200, 0),
                   radius=4):
    """Wizualizacja minucji."""
    if background is None:
        base = (skel * 255).astype(np.uint8) if skel.max() <= 1 else skel.copy()
        canvas = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        if background.ndim == 2:
            canvas = cv2.cvtColor(background.astype(np.uint8),
                                   cv2.COLOR_GRAY2BGR)
        else:
            canvas = background.copy()

    for m in minutiae:
        color = ending_color if m.kind == "ending" else bifurcation_color
        if m.kind == "ending":
            cv2.circle(canvas, (m.col, m.row), radius, color, 1)
        else:
            cv2.rectangle(canvas,
                          (m.col - radius, m.row - radius),
                          (m.col + radius, m.row + radius),
                          color, 1)
    return canvas


def count_by_kind(minutiae):
    counts = {"ending": 0, "bifurcation": 0}
    for m in minutiae:
        counts[m.kind] += 1
    return counts
