"""Wczytywanie i wstepne przetwarzanie odciskow palcow.

Funkcje zwracaja obrazy binarne w formacie numpy uint8 z wartosciami
{0, 1} (1 = linia papilarna, 0 = tlo) - takie kodowanie pasuje do algorytmow
scieniania KMM/K3M.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_fingerprint(path):
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Nie udalo sie wczytac pliku: {path}")
    return img


def _segment_fingerprint(gray, block_size=16, var_threshold=60.0):
    """Maska obszaru palca na podstawie lokalnej wariancji w blokach."""
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    gray_f = gray.astype(np.float32)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = gray_f[y:y + block_size, x:x + block_size]
            if block.size == 0:
                continue
            if block.var() > var_threshold:
                mask[y:y + block_size, x:x + block_size] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def preprocess_fingerprint(gray,
                            clahe_clip=3.0,
                            clahe_tile=8,
                            block_size=17,
                            C=7,
                            do_segmentation=True,
                            close_kernel=3,
                            open_kernel=3):
    """Pelen pipeline preprocessingu odcisku.

    Parametry:
      - clahe_clip / clahe_tile  - parametry CLAHE (wyrownanie kontrastu),
      - block_size / C           - parametry adaptacyjnej binaryzacji,
      - close_kernel             - rozmiar elementu zamykajacego przerwy
                                    pomiedzy fragmentami linii,
      - open_kernel              - rozmiar elementu usuwajacego drobny szum,
      - do_segmentation          - czy wyznaczac maske obszaru palca.

    Zwraca slownik:
      - gray         (orginalny obraz w skali szarosci)
      - enhanced     (po CLAHE)
      - mask         (maska palca)
      - binary       (wynik {0,1})
      - binary_uint8 (wynik 0/255 do wyswietlania)
    """
    if gray.ndim != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                             tileGridSize=(clahe_tile, clahe_tile))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    bs = block_size if block_size % 2 == 1 else block_size + 1
    binary_inv = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        bs, C,
    )

    if do_segmentation:
        mask = _segment_fingerprint(gray)
        binary_inv = cv2.bitwise_and(binary_inv, binary_inv, mask=mask)
    else:
        mask = np.ones_like(gray, dtype=np.uint8)

    if open_kernel >= 2:
        kop = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (open_kernel, open_kernel))
        cleaned = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kop)
    else:
        cleaned = binary_inv

    if close_kernel >= 2:
        kcl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (close_kernel, close_kernel))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kcl)

    binary = (cleaned > 0).astype(np.uint8)

    return {
        "gray": gray,
        "enhanced": enhanced,
        "binary": binary,
        "mask": mask,
        "binary_uint8": (binary * 255).astype(np.uint8),
    }


def normalize_for_display(img, invert=False):
    arr = np.asarray(img)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    elif arr.max() <= 1:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    if invert:
        arr = 255 - arr
    return arr
