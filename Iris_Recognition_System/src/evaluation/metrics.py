"""
Metryki ewaluacyjne: rozkłady odległości Hamminga oraz krzywe FAR/FRR.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..encoding.iris_code import IrisCode
from ..matching.matcher import hamming_distance_with_rotation


@dataclass
class DistanceDistributions:
    intra: np.ndarray   # HD dla par z tej samej klasy
    inter: np.ndarray   # HD dla par z różnych klas


def intra_inter_distances(codes: list[IrisCode],
                          labels: list[str],
                          max_shift: int = 8
                          ) -> DistanceDistributions:
    """
    Liczy odległości Hamminga dla wszystkich par kodów:
        - intra-class: odległości między kodami z tej samej klasy,
        - inter-class: odległości między kodami z różnych klas.
    """
    n = len(codes)
    intra: list[float] = []
    inter: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            hd, _ = hamming_distance_with_rotation(codes[i], codes[j],
                                                   max_shift=max_shift)
            if labels[i] == labels[j]:
                intra.append(hd)
            else:
                inter.append(hd)
    return DistanceDistributions(intra=np.asarray(intra),
                                 inter=np.asarray(inter))


@dataclass
class ThresholdMetrics:
    threshold: float
    far: float           # False Acceptance Rate
    frr: float           # False Rejection Rate


def decision_threshold_metrics(dist: DistanceDistributions,
                               thresholds: np.ndarray | None = None
                               ) -> list[ThresholdMetrics]:
    """
    Dla zadanych progów odległości Hamminga oblicza FAR i FRR.

    FAR = P(inter <= t)  (akceptacja przy różnych klasach = błąd)
    FRR = P(intra >  t)  (odrzucenie przy tej samej klasie = błąd)
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    results = []
    for t in thresholds:
        far = float(np.mean(dist.inter <= t)) if dist.inter.size else 0.0
        frr = float(np.mean(dist.intra > t)) if dist.intra.size else 0.0
        results.append(ThresholdMetrics(threshold=float(t), far=far, frr=frr))
    return results
