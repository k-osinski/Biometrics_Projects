"""
Wrapper łączący detekcję źrenicy, tęczówki i powiek w jeden krok.
"""
from dataclasses import dataclass, field
import numpy as np

from ..core.basic_operations import to_grayscale
from .pupil_detector import detect_pupil, PupilResult
from .iris_detector import detect_iris, IrisResult
from .eyelid_detector import detect_eyelids, EyelidResult


@dataclass
class SegmentationResult:
    pupil: PupilResult
    iris: IrisResult
    gray: np.ndarray
    eyelids: EyelidResult | None = None


def segment_eye(img: np.ndarray,
                x_p: float = 3.0,
                x_i: float = 1.15,
                pupil_close: int = 7,
                pupil_open: int = 5,
                pupil_keep_largest: bool = True,
                projection_threshold: float = 0.5,
                iris_smoothing_sigma: float = 2.0,
                iris_min_radius_factor: float = 1.5,
                iris_max_radius_factor: float = 4.5,
                detect_eyelids_flag: bool = True,
                eyelid_min_gradient: float = 80.0,
                eyelid_inlier_dist: float = 4.0,
                eyelid_min_inlier_fraction: float = 0.2,
                eyelid_vertical_inner_skip: float = 0.3,
                ) -> SegmentationResult:
    """
    Cała ścieżka segmentacji: obraz wejściowy -> źrenica -> tęczówka
    (opcjonalnie -> powieki).
    """
    gray = to_grayscale(img) if img.ndim == 3 else img
    pupil = detect_pupil(gray, x_p=x_p,
                         close_size=pupil_close,
                         open_size=pupil_open,
                         projection_threshold=projection_threshold,
                         keep_largest=pupil_keep_largest)
    iris = detect_iris(gray, pupil,
                       x_i=x_i,
                       smoothing_sigma=iris_smoothing_sigma,
                       min_radius_factor=iris_min_radius_factor,
                       max_radius_factor=iris_max_radius_factor)
    eyelids = None
    if detect_eyelids_flag:
        eyelids = detect_eyelids(
            gray, pupil, iris,
            min_gradient=eyelid_min_gradient,
            inlier_dist=eyelid_inlier_dist,
            min_inlier_fraction=eyelid_min_inlier_fraction,
            vertical_inner_skip=eyelid_vertical_inner_skip)
    return SegmentationResult(pupil=pupil, iris=iris, gray=gray,
                              eyelids=eyelids)
