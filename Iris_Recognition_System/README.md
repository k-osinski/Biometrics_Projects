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

---

## References

* J. Daugman, *How Iris Recognition Works*, IEEE Trans. CSVT, 2004.
* Course textbook: *Wybrane zagadnienia biometrii*, Chapter 6 — iris
  recognition (sections 6.2.1–6.2.2.3).
* MMU Iris Database, Multimedia University.
