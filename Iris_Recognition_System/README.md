# Iris Recognition System

Iris-based biometric identification implemented from scratch in Python, following
the algorithm proposed by **John Daugman**. Built as Project 2 for the *Biometria*
course (Warsaw University of Technology, 2026).

The system performs the full pipeline on an eye image:

1. **Segmentation** — locate the pupil and the iris boundary,
2. **Normalization** — unwrap the iris ring into a fixed-size rectangle
   (Daugman's *rubber sheet model*),
3. **Encoding** — split the rectangle into 8 radial bands, apply 1D Gabor
   wavelets, and encode the phase response as a 2048-bit *iris code*,
4. **Matching** — compare two iris codes via normalized Hamming distance with
   rotation compensation.

The implementation deliberately avoids high-level computer-vision libraries
(no `cv2.HoughCircles`, no `scikit-image` segmentation). Morphology, projections
and Gabor filters are written by hand. OpenCV is only used in optional
optimized variants of Bernsen / Niblack thresholding inherited from
[`Image_Processing_Core`](../Image_Processing_Core).

---

## Project structure

```
Iris_Recognition_System/
├── app/
│   ├── main_app.py            # Streamlit GUI (single image + pairwise comparison)
│   └── evaluation_app.py      # MMU-wide evaluation + FAR/FRR curves
├── src/
│   ├── core/                  # Image-processing primitives (reused from Project 1)
│   │   ├── basic_operations.py
│   │   ├── convolution.py
│   │   ├── filters.py
│   │   ├── thresholding.py
│   │   ├── edge_detection.py
│   │   ├── analysis.py
│   │   ├── morphology.py      # erode / dilate / open / close (manual)
│   │   └── projections.py     # 1D projections + circle estimator
│   ├── segmentation/
│   │   ├── pupil_detector.py  # mean-threshold + morphology + projections
│   │   ├── iris_detector.py   # radial gradient search around the pupil
│   │   └── pipeline.py
│   ├── normalization/
│   │   └── unwrapper.py       # Daugman rubber-sheet polar unwrap
│   ├── encoding/
│   │   ├── band_extractor.py  # 8 bands × 128 points, Gaussian radial avg.
│   │   ├── gabor_filter.py    # 1D Gabor wavelet (σ = π·f / 2)
│   │   └── iris_code.py       # phase-quadrant encoding -> 2048-bit code
│   ├── matching/
│   │   └── matcher.py         # Hamming distance + rotation compensation
│   ├── evaluation/
│   │   ├── dataset.py         # MMU iterator
│   │   └── metrics.py         # intra/inter distances, FAR/FRR
│   └── pipeline.py            # end-to-end `run_iris_pipeline(...)`
├── MMU-Iris-Database/         # MMU Iris Dataset (not committed)
├── plan.md
├── requirements.txt
└── README.md
```

---

## Installation

Python 3.10+ recommended.

```bash
git clone <repo-url>
cd Biometrics_Projects/Iris_Recognition_System
pip install -r requirements.txt
```

Then download the [MMU Iris Database](https://persci.mit.edu/pub_pdfs/iris.pdf)
(or any compatible iris dataset) and place it under `MMU-Iris-Database/`
following the structure:

```
MMU-Iris-Database/
├── 1/
│   ├── left/   *.bmp
│   └── right/  *.bmp
├── 2/
│   ...
```

---

## Usage

### Interactive GUI (single image / two-image comparison)

```bash
streamlit run app/main_app.py
```

The sidebar exposes every tunable parameter — pupil/iris thresholds (`X_P`,
`X_I`), morphology kernel sizes, unwrap resolution, and the Gabor frequency `f`.

### Batch evaluation on MMU

```bash
streamlit run app/evaluation_app.py
```

For each enrolled image the script computes an iris code; pairs of codes are
then compared and bucketed into:

* **intra-class** — same subject, same eye (positive pairs),
* **inter-class** — different subject or different eye (negative pairs).

The app produces:

* histogram of normalized Hamming distances (analogue of *Fig. 6.16*
  from the textbook),
* FAR/FRR curves and the **Equal Error Rate** point.

### Programmatic use

```python
import numpy as np
from PIL import Image
from src.pipeline import run_iris_pipeline
from src.matching import compare

img_a = np.asarray(Image.open("MMU-Iris-Database/1/left/aeval1.bmp").convert("L"))
img_b = np.asarray(Image.open("MMU-Iris-Database/1/left/aeval2.bmp").convert("L"))

result_a = run_iris_pipeline(img_a)
result_b = run_iris_pipeline(img_b)

print("Pupil:", result_a.segmentation.pupil)
print("Iris:",  result_a.segmentation.iris)
print("Code shape:", result_a.code.bits.shape)   # (8, 128, 2) = 2048 bits

print("HD =", compare(result_a.code, result_b.code))
```

Expected output on the bundled MMU sample:

| Pair                                       | Hamming distance |
|--------------------------------------------|------------------|
| `1/left/aeval1` vs `1/left/aeval2` (same)  | ≈ 0.26           |
| `1/left/aeval1` vs `2/left/blval1` (diff.) | ≈ 0.42           |

---

## Algorithm details

### Segmentation

The base threshold

$$P = \frac{1}{h \cdot w} \sum_{i=0}^{h-1} \sum_{j=0}^{w-1} A(i, j)$$

is the mean intensity of the eye image. The pupil threshold is `P / X_P` and
the iris threshold is `P / X_I`, both `X_*` chosen experimentally
(Project 2 specification). Pixels below the threshold are marked as
foreground; closing fills small dark "holes" inside the pupil and opening
removes scattered eyelash pixels. The pupil center and radius are then
recovered from the **horizontal and vertical projections** of the cleaned
binary mask (the largest contiguous interval around the projection peak).

The iris boundary uses the same threshold formula but is detected by a
**radial-gradient search** around the pupil center: we average the smoothed
intensity profile in four cardinal directions and pick the radius with the
strongest dark-to-bright transition. This is more robust than thresholding
when the iris/sclera contrast is weak.

### Unwrap

For every $(r, \theta)$ with $r \in [0, 1]$ and $\theta \in [0, 2\pi)$ the
point inside the iris ring is

$$\begin{aligned}
x(r, \theta) &= (1 - r)\,x_p(\theta) + r\,x_i(\theta) \\
y(r, \theta) &= (1 - r)\,y_p(\theta) + r\,y_i(\theta)
\end{aligned}$$

with $x_p, y_p$ on the pupil circle and $x_i, y_i$ on the iris circle.
Sampling is bilinear; the default output size is **64 × 256**.

### Encoding (Daugman)

Each unwrapped iris is split into **8 radial bands**. Within a band, columns
are reduced to a 1D signal via Gaussian-weighted averaging in the radial
direction (preferring the band center — see Fig. 6.14 of the textbook).
The 1D signal is resampled to 128 equispaced points and convolved with a
**1D Gabor wavelet**:

$$G(x; x_k, \sigma, f) = \exp\!\left(-\frac{(x-x_k)^2}{\sigma^2}\right)
                       \cdot \exp(-i \cdot 2 \pi f (x - x_k))$$

> ⚠️ Important: per the project instructions the envelope width must be
> $\sigma = \tfrac{1}{2}\,\pi\,f$ — **not** $\sigma = 1/(2\pi f)$. The
> incorrect form pushes every coefficient into quadrants I & IV of the
> complex plane, breaking phase encoding.

Each complex coefficient is encoded as **2 bits** (sign of real, sign of
imaginary part) following the Gray-code phase-quadrant scheme:

| Quadrant | (Re, Im) | bits |
|----------|----------|------|
| I        | (≥0, ≥0) | `00` |
| II       | (<0, ≥0) | `01` |
| III      | (<0, <0) | `11` |
| IV       | (≥0, <0) | `10` |

→ 8 × 128 × 2 = **2048-bit iris code**, plus a per-bit reliability mask
that excludes columns occluded by eyelids/eyelashes (outer bands cover
226° / 180° instead of the full 360°).

### Matching

Normalized Hamming distance over jointly-valid bits, with **cyclic shifts of
±k columns** to compensate for head/iris rotation (default `k = 8`):

$$\text{HD}(A, B) = \min_{|s| \le k} \frac{1}{|\mathcal V|}
   \sum_{i \in \mathcal V}\bigl(a_i \oplus b_{i+s}\bigr)$$

where $\mathcal V$ is the set of bit positions valid in both codes.

A typical threshold of **0.32** separates intra- from inter-class pairs on
MMU; the textbook reports the inter-class distribution centered around
0.46 with σ ≈ 0.02 (Fig. 6.16).

---

## Notes & limitations

* The pupil detector assumes the pupil is reasonably centered in the frame.
  Severely off-center or partially-occluded pupils may need different
  `X_P` or larger morphology kernels.
* Concentric pupil/iris circles are assumed (per the textbook). A more
  general implementation would use independent centers.
* Gabor frequency `f` is exposed as a tunable parameter — values around
  `0.08`–`0.12` work well on MMU. Optimal `f` should be chosen empirically
  by minimizing intra-class HD on a held-out subset.

---

## References

* J. Daugman, *How Iris Recognition Works*, IEEE Trans. CSVT, 2004.
* Course textbook: *Wybrane zagadnienia biometrii*, Chapter 6 — iris
  recognition (sections 6.2.1–6.2.2.3).
* MMU Iris Database, Multimedia University.
