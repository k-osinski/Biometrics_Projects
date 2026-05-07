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
                      pupil_keep_largest: bool = True,
                      iris_smoothing_sigma: float = 2.0,
                      iris_min_radius_factor: float = 1.5,
                      iris_max_radius_factor: float = 4.5,
                      detect_eyelids_flag: bool = True,
                      eyelid_min_gradient: float = 80.0,
                      eyelid_inlier_dist: float = 4.0,
                      eyelid_min_inlier_fraction: float = 0.2,
                      eyelid_vertical_inner_skip: float = 0.3,
                      radial_res: int = 64,
                      angular_res: int = 256,
                      gabor_frequency: float = DEFAULT_FREQUENCY,
                      gabor_sigma: float | None = None,
                      ) -> IrisPipelineResult:
    """Pełen pipeline rozpoznawania tęczówki dla pojedynczego obrazu oka."""
    seg = segment_eye(img,
                      x_p=x_p, x_i=x_i,
                      pupil_close=pupil_close,
                      pupil_open=pupil_open,
                      pupil_keep_largest=pupil_keep_largest,
                      iris_smoothing_sigma=iris_smoothing_sigma,
                      iris_min_radius_factor=iris_min_radius_factor,
                      iris_max_radius_factor=iris_max_radius_factor,
                      detect_eyelids_flag=detect_eyelids_flag,
                      eyelid_min_gradient=eyelid_min_gradient,
                      eyelid_inlier_dist=eyelid_inlier_dist,
                      eyelid_min_inlier_fraction=eyelid_min_inlier_fraction,
                      eyelid_vertical_inner_skip=eyelid_vertical_inner_skip)
    unwrap = unwrap_iris(seg.gray, seg.pupil, seg.iris,
                         radial_res=radial_res,
                         angular_res=angular_res,
                         eyelids=seg.eyelids)
    code = encode_iris(unwrap.image,
                       frequency=gabor_frequency,
                       sigma=gabor_sigma,
                       unwrap_mask=unwrap.mask)
    return IrisPipelineResult(segmentation=seg, unwrap=unwrap, code=code)


from .segmentation import (
    detect_pupil, detect_iris, PupilResult, IrisResult,
)
