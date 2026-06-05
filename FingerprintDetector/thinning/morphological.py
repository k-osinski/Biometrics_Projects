"""Szkieletyzacja morfologiczna.

Implementacja klasycznego wzoru Lantuejoula:

    S(X) = U_{n>=0} [ (X (-) nB) \ ((X (-) nB) o B) ]

gdzie:
  - (-)   to erozja,
  - o     to otwarcie (open = erozja + dylatacja),
  - U     to suma teoriomnogosciowa,
  - nB    to n-krotna erozja elementem strukturalnym B (kwadrat 3x3).

Iteracje sa wykonywane az do uzyskania pustego obrazu po n-tej erozji.
Algorytm ten zostal przedstawiony na wykladzie i jest podstawowym
algorytmem szkieletyzacji morfologicznej.
"""
from __future__ import annotations

import cv2
import numpy as np


def morphological_skeleton(binary: np.ndarray,
                            kernel_shape: int = cv2.MORPH_CROSS,
                            kernel_size: int = 3,
                            max_iter: int = 200) -> np.ndarray:
    """Szkieletyzacja morfologiczna wedlug wzoru Lantuejoula.

    Parametry
    ----------
    binary : ndarray
        Obraz binarny {0, 1} (uint8) gdzie 1 oznacza obiekt.
    kernel_shape : int
        Ksztalt elementu strukturalnego (cv2.MORPH_CROSS lub cv2.MORPH_RECT).
    kernel_size : int
        Rozmiar elementu strukturalnego (zwykle 3).
    max_iter : int
        Maksymalna liczba iteracji (zabezpieczenie).

    Zwraca
    -------
    skeleton : ndarray
        Obraz binarny {0, 1} ze szkieletem.
    """
    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)
    work = (binary > 0).astype(np.uint8) * 255

    skeleton = np.zeros_like(work)
    element = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))

    for _ in range(max_iter):
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        diff = cv2.subtract(work, opened)
        skeleton = cv2.bitwise_or(skeleton, diff)
        work = cv2.erode(work, element)
        if cv2.countNonZero(work) == 0:
            break

    return (skeleton > 0).astype(np.uint8)
