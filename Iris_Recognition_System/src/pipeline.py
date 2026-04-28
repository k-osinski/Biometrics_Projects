"""
Wysokopoziomowa funkcja end-to-end:
    obraz oka -> segmentacja -> rozwinięcie -> kod tęczówki.
"""
from dataclasses import dataclass
import numpy as np

from .segmentation import segment_eye, SegmentationResult
from .normalization import unwrap_iris, UnwrapResult
from .encoding import encode_iris, IrisCode, DEFAULT_FREQUENCY


@dataclass
class IrisPipelineResult:
    segmentation: SegmentationResult
    unwrap: UnwrapResult
    code: IrisCode


def run_iris_pipeline(img: np.ndarray,
                      x_p: float = 3.0,
                      x_i: float = 1.15,
                      pupil_close: int = 7,
                      pupil_open: int = 5,
                      radial_res: int = 64,
                      angular_res: int = 256,
                      gabor_frequency: float = DEFAULT_FREQUENCY,
                      gabor_sigma: float | None = None,
                      ) -> IrisPipelineResult:
    """Pełen pipeline rozpoznawania tęczówki dla pojedynczego obrazu oka."""
    seg = segment_eye(img, x_p=x_p, x_i=x_i,
                      pupil_close=pupil_close,
                      pupil_open=pupil_open)
    unwrap = unwrap_iris(seg.gray, seg.pupil, seg.iris,
                         radial_res=radial_res,
                         angular_res=angular_res)
    code = encode_iris(unwrap.image,
                       frequency=gabor_frequency,
                       sigma=gabor_sigma)
    return IrisPipelineResult(segmentation=seg, unwrap=unwrap, code=code)


# kompatybilność z importami w stylu `from src import run_iris_pipeline`
from .segmentation import (  # noqa: E402,F401
    detect_pupil, detect_iris, PupilResult, IrisResult,
)
