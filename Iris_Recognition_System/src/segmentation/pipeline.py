"""
Wygodny wrapper łączący detekcję źrenicy i tęczówki w jeden krok.
"""
from dataclasses import dataclass
import numpy as np

from ..core.basic_operations import to_grayscale
from .pupil_detector import detect_pupil, PupilResult
from .iris_detector import detect_iris, IrisResult


@dataclass
class SegmentationResult:
    pupil: PupilResult
    iris: IrisResult
    gray: np.ndarray  # obraz wejściowy w skali szarości (uint8)


def segment_eye(img: np.ndarray,
                x_p: float = 3.0,
                x_i: float = 1.15,
                pupil_close: int = 7,
                pupil_open: int = 5,
                projection_threshold: float = 0.5) -> SegmentationResult:
    """
    Cała ścieżka segmentacji: obraz wejściowy -> źrenica -> tęczówka.
    """
    gray = to_grayscale(img) if img.ndim == 3 else img
    pupil = detect_pupil(gray, x_p=x_p,
                         close_size=pupil_close,
                         open_size=pupil_open,
                         projection_threshold=projection_threshold)
    iris = detect_iris(gray, pupil, x_i=x_i)
    return SegmentationResult(pupil=pupil, iris=iris, gray=gray)
