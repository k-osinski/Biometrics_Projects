"""Postprocessing szkieletu odcisku palca.

Po szkieletyzacji typowo pojawiaja sie:
  - krotkie pseudoboki (spurs) wystajace ze szkieletu - powoduja falszywe
    minucje,
  - przerwy w liniach papilarnych - powoduja falszywe zakonczenia,
  - male wyspy odlaczonych pikseli.

Modul implementuje proste operacje morfologiczne i operacje na sasiedztwie
polepszajace jakosc szkieletu przed detekcja minucji.
"""
from __future__ import annotations

import cv2
import numpy as np


def _neighbour_count(skel):
    """Liczba sasiadow obiektu w 8-sasiedztwie dla kazdego piksela."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    s = skel.astype(np.uint8)
    p = np.pad(s, 1, mode='constant', constant_values=0)
    H, W = s.shape
    nb = (
        p[0:H, 0:W] + p[0:H, 1:W + 1] + p[0:H, 2:W + 2]
        + p[1:H + 1, 0:W] + p[1:H + 1, 2:W + 2]
        + p[2:H + 2, 0:W] + p[2:H + 2, 1:W + 1] + p[2:H + 2, 2:W + 2]
    )
    return nb.astype(np.int32)


def remove_spurs(skel, max_length=3):
    """Iteracyjnie usuwa zakonczenia (CN=1) krotsze niz max_length."""
    out = (skel > 0).astype(np.uint8)
    for _ in range(max(1, max_length)):
        nb = _neighbour_count(out)
        endpoints = (out == 1) & (nb == 1)
        if not endpoints.any():
            break
        out[endpoints] = 0
    return out


def remove_small_components(skel, min_size=8):
    """Usuwa male, izolowane skladowe spojne."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        skel.astype(np.uint8), connectivity=8)
    out = np.zeros_like(skel)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            out[labels == i] = 1
    return out


def bridge_gaps(skel, max_gap=4):
    """Probuje polaczyc poprzerywane linie laczac bliskie zakonczenia (CN=1).

    Dla kazdego zakonczenia szuka najblizszego innego zakonczenia w promieniu
    max_gap i rysuje miedzy nimi krotki odcinek prosty.
    """
    out = (skel > 0).astype(np.uint8)
    nb = _neighbour_count(out)
    endpoints = np.argwhere((out == 1) & (nb == 1))
    if len(endpoints) < 2:
        return out

    used = set()
    for i, (r1, c1) in enumerate(endpoints):
        key1 = (int(r1), int(c1))
        if key1 in used:
            continue
        best = None
        best_d2 = (max_gap + 1) ** 2
        for j, (r2, c2) in enumerate(endpoints):
            if j == i:
                continue
            key2 = (int(r2), int(c2))
            if key2 in used:
                continue
            d2 = int((r1 - r2) ** 2 + (c1 - c2) ** 2)
            if 1 < d2 <= max_gap ** 2 and d2 < best_d2:
                best_d2 = d2
                best = key2
        if best is not None:
            cv2.line(out, (c1, r1), (best[1], best[0]), 1, 1)
            used.add(key1)
            used.add(best)
    return out


def postprocess_skeleton(skel,
                          spur_length=3,
                          min_component_size=8,
                          gap_size=4,
                          do_bridge=True):
    """Lagodny postprocessing szkieletu.

    Parametry:
      - spur_length          - max dlugosc usuwanego "kolca" (CN=1),
      - min_component_size   - usun skladowe spojne mniejsze niz N pikseli,
      - gap_size             - max dystans miedzy laczonymi zakonczeniami,
      - do_bridge            - czy probowac laczyc przerwane linie.
    """
    out = (skel > 0).astype(np.uint8)
    if do_bridge and gap_size >= 2:
        out = bridge_gaps(out, max_gap=gap_size)
    if spur_length >= 1:
        out = remove_spurs(out, max_length=spur_length)
    if min_component_size >= 2:
        out = remove_small_components(out, min_size=min_component_size)
    return out
