"""
Aplikacja Streamlit - Rozpoznawanie tęczówki.
Uruchomienie:
    streamlit run app/main_app.py
Funkcje:
    1. wgranie obrazu oka (BMP / PNG / JPG),
    2. interaktywna segmentacja (źrenica + tęczówka),
    3. rozwinięcie tęczówki do prostokąta,
    4. kodowanie algorytmem Daugmana (8 pasów × 128 falek Gabora),
    5. porównanie kodów dwóch obrazów (HD z kompensacją rotacji).
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pipeline import run_iris_pipeline
from src.matching import compare


st.set_page_config(page_title="Biometria – Rozpoznawanie Tęczówki",
                   layout="wide")
st.title("Rozpoznawanie tęczówki (algorytm Daugmana)")
st.caption("Segmentacja → rozwinięcie → kod tęczówki → odległość Hamminga.")


# --------------------------------------------------------------------------- #
# Sidebar (parametry)                                                         #
# --------------------------------------------------------------------------- #
st.sidebar.header("Parametry algorytmu")

with st.sidebar.expander("Segmentacja źrenicy", expanded=True):
    x_p = st.slider("X_P (próg dla źrenicy = P / X_P)",
                    min_value=1.5, max_value=6.0, value=3.0, step=0.1,
                    help="Większe X_P -> niższy próg -> wykrywamy "
                         "tylko najczarniejsze piksele.")
    pupil_close = st.slider("Rozmiar zamknięcia (pupil)", 1, 21, 7, 2)
    pupil_open = st.slider("Rozmiar otwarcia (pupil)", 1, 21, 5, 2)
    pupil_keep_largest = st.checkbox(
        "Filtr największego komponentu",
        value=True,
        help="Po cleanupie morfologicznym zostawia tylko najliczniejszy "
             "spójny obszar w masce binarnej. Eliminuje resztki rzęs/cieni.",
    )

with st.sidebar.expander("Segmentacja tęczówki", expanded=True):
    x_i = st.slider("X_I (próg dla tęczówki = P / X_I)",
                    min_value=0.5, max_value=2.5, value=1.15, step=0.05,
                    help="Mniejsze X_I -> wyższy próg -> szerszy zakres "
                         "'ciemnych' promieni. Większe -> ostrzejsza "
                         "tęczówka.")
    iris_smooth = st.slider("Sigma wygładzania (iris)", 0.5, 8.0, 2.0, 0.5)
    iris_min_factor = st.slider("Min. promień tęczówki (× r_p)",
                                1.1, 3.0, 1.5, 0.1)
    iris_max_factor = st.slider("Maks. promień tęczówki (× r_p)",
                                2.0, 6.0, 4.5, 0.1)

with st.sidebar.expander("Detekcja powiek (Daugman)", expanded=False):
    detect_eyelids_flag = st.checkbox(
        "Wykrywaj powieki (parabole)", value=True,
        help="Sobel pionowy + RANSAC fit paraboli y=ax²+bx+c, dla górnej "
             "i dolnej powieki. Wyniki maskują rozwiniętą tęczówkę "
             "i propagują się do kodu.",
    )
    eyelid_min_gradient = st.slider(
        "Min. |∂I/∂y| (próg krawędzi)", 0.0, 400.0, 80.0, 5.0,
        help="Sobel na uint8 daje wartości rzędu setek (skok jasności "
             "100→200 ≈ 400). Im wyższy próg, tym mniej kandydackich "
             "kolumn -> mniej outlierów, ale ryzyko nie wykrycia "
             "słabej powieki.",
    )
    eyelid_inlier_dist = st.slider(
        "Tolerancja RANSAC (px)", 1.0, 15.0, 4.0, 0.5,
    )
    eyelid_min_inlier_fraction = st.slider(
        "Min. frakcja inlierów (jakość fitu)", 0.05, 0.8, 0.2, 0.05,
        help="Jeśli RANSAC zwróci fit z mniejszą liczbą inlierów "
             "(względem wszystkich kandydackich kolumn), powieka jest "
             "uznana za niewykrytą i statyczna maska 360°/226°/180° "
             "bierze rolę zabezpieczenia.",
    )
    eyelid_vertical_inner_skip = st.slider(
        "Pominięty wewnętrzny pas tęczówki", 0.0, 0.6, 0.3, 0.05,
        help="Frakcja promienia tęczówki (od środka źrenicy), w której "
             "NIE szukamy powieki. Wyłącza wnętrze tęczówki gdzie często "
             "siedzą rzęsy/melanina dające fałszywe peaki gradientu.",
    )

with st.sidebar.expander("Rozwinięcie tęczówki", expanded=False):
    radial_res = st.slider("Wysokość (radial_res)", 16, 128, 64, 8)
    angular_res = st.slider("Szerokość (angular_res)", 64, 512, 256, 32)

with st.sidebar.expander("Kodowanie (Gabor)", expanded=True):
    frequency = st.slider("Częstotliwość falki f", 0.02, 0.5, 0.10, 0.01)
    use_auto_sigma = st.checkbox("σ = π·f / 2 (zalecane)",
                                 value=True)
    if use_auto_sigma:
        sigma_value: float | None = None
        st.caption(f"σ aktualnie = {0.5 * np.pi * frequency:.4f}")
    else:
        sigma_value = st.slider("Sigma (ręcznie)", 0.5, 30.0, 4.0, 0.5)


# --------------------------------------------------------------------------- #
# Funkcje pomocnicze                                                          #
# --------------------------------------------------------------------------- #
def _load_pil(uploaded_file) -> np.ndarray:
    pil = Image.open(uploaded_file)
    if pil.mode not in ("L", "RGB"):
        pil = pil.convert("RGB")
    return np.asarray(pil)


def _annotate_segmentation(img: np.ndarray, result) -> Image.Image:
    """Rysuje okrąg źrenicy/tęczówki + parabole powiek na obrazie."""
    if img.ndim == 2:
        rgb = np.stack([img] * 3, axis=-1)
    else:
        rgb = img.copy()
    pil = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(pil)
    p, i = result.segmentation.pupil, result.segmentation.iris
    draw.ellipse([p.cx - p.radius, p.cy - p.radius,
                  p.cx + p.radius, p.cy + p.radius],
                 outline="#FF6B6B", width=2)
    draw.ellipse([i.cx - i.radius, i.cy - i.radius,
                  i.cx + i.radius, i.cy + i.radius],
                 outline="#51CF66", width=2)
    draw.line([(p.cx - 5, p.cy), (p.cx + 5, p.cy)], fill="#FFEC3D", width=2)
    draw.line([(p.cx, p.cy - 5), (p.cx, p.cy + 5)], fill="#FFEC3D", width=2)

    # parabole powiek
    eyelids = result.segmentation.eyelids
    if eyelids is not None:
        h, w = pil.size[1], pil.size[0]
        x_lo = max(0, p.cx - i.radius)
        x_hi = min(w - 1, p.cx + i.radius)
        xs = np.arange(x_lo, x_hi + 1)
        for parabola, color in ((eyelids.upper, "#5C9AFF"),
                                (eyelids.lower, "#5C9AFF")):
            if parabola is None:
                continue
            ys = parabola(xs)
            pts = [(int(x), int(y)) for x, y in zip(xs, ys)
                   if 0 <= y < h]
            if len(pts) >= 2:
                draw.line(pts, fill=color, width=2)
    return pil


def _unwrap_with_mask(unwrap) -> Image.Image:
    """Rozwinięta tęczówka z półprzezroczystym ściemnieniem zamaskowanych pikseli."""
    img = unwrap.image
    rgb = np.stack([img] * 3, axis=-1).astype(np.float64)
    invalid = ~unwrap.mask
    if invalid.any():
        red_tint = np.array([200.0, 60.0, 60.0])
        alpha = 0.55
        rgb[invalid] = (1 - alpha) * rgb[invalid] + alpha * red_tint
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)


def _iris_code_visual(code) -> Image.Image:
    """Zamienia kod (num_bands, P, 2) na obraz binarny do podglądu."""
    bits = code.bits.reshape(code.bits.shape[0], -1)  # (bands, P*2)
    arr = (bits * 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode="L").resize(
        (arr.shape[1] * 2, arr.shape[0] * 16), Image.NEAREST)
    return pil


def _process(img_array: np.ndarray):
    return run_iris_pipeline(
        img_array,
        x_p=x_p, x_i=x_i,
        pupil_close=pupil_close, pupil_open=pupil_open,
        pupil_keep_largest=pupil_keep_largest,
        iris_smoothing_sigma=iris_smooth,
        iris_min_radius_factor=iris_min_factor,
        iris_max_radius_factor=iris_max_factor,
        detect_eyelids_flag=detect_eyelids_flag,
        eyelid_min_gradient=eyelid_min_gradient,
        eyelid_inlier_dist=eyelid_inlier_dist,
        eyelid_min_inlier_fraction=eyelid_min_inlier_fraction,
        eyelid_vertical_inner_skip=eyelid_vertical_inner_skip,
        radial_res=radial_res, angular_res=angular_res,
        gabor_frequency=frequency, gabor_sigma=sigma_value,
    )


# --------------------------------------------------------------------------- #
# Zakładki                                                                    #
# --------------------------------------------------------------------------- #
tab_pipeline, tab_match = st.tabs(["Pojedynczy obraz", "Porównanie dwóch obrazów"])


# --- Tab 1: pojedynczy obraz ---------------------------------------------- #
with tab_pipeline:
    uploaded = st.file_uploader("Wgraj obraz oka",
                                type=["bmp", "png", "jpg", "jpeg", "tiff"],
                                key="single")
    if uploaded is None:
        st.info("Wgraj obraz, aby uruchomić pipeline.")
    else:
        img_array = _load_pil(uploaded)
        try:
            result = _process(img_array)
        except Exception as exc:
            st.error(f"Błąd przetwarzania: {exc}")
        else:
            cols = st.columns(2)
            with cols[0]:
                st.subheader("Segmentacja")
                st.image(_annotate_segmentation(img_array, result),
                         use_container_width=True,
                         caption=(
                             f"Źrenica: ({result.segmentation.pupil.cx},"
                             f" {result.segmentation.pupil.cy}), "
                             f"r={result.segmentation.pupil.radius} | "
                             f"Tęczówka: r={result.segmentation.iris.radius}"
                             f" ({result.segmentation.iris.method_used})"))
                st.image(result.segmentation.pupil.binary_mask,
                         use_container_width=True,
                         caption="Maska binarna źrenicy (po morfologii)",
                         clamp=True)

            with cols[1]:
                st.subheader("Rozwinięcie tęczówki")
                masked_frac = float((~result.unwrap.mask).mean())
                st.image(_unwrap_with_mask(result.unwrap),
                         use_container_width=True,
                         caption=(f"Rozmiar: {result.unwrap.image.shape} | "
                                  f"zamaskowane: {masked_frac:.1%} "
                                  f"(czerwone = ignorowane)"))
                st.subheader("Kod tęczówki")
                st.image(_iris_code_visual(result.code),
                         use_container_width=True,
                         caption=f"Razem bitów: {result.code.total_bits}")

            st.markdown("---")
            with st.expander("Szczegóły progowania"):
                st.write({
                    "Próg P (średnia jasność)":
                        int(np.round(np.mean(result.segmentation.gray))),
                    "Próg dla źrenicy (P/X_P)":
                        result.segmentation.pupil.threshold,
                    "Próg dla tęczówki (P/X_I)":
                        result.segmentation.iris.threshold,
                })


# --- Tab 2: porównanie ---------------------------------------------------- #
with tab_match:
    cols = st.columns(2)
    with cols[0]:
        f_a = st.file_uploader("Obraz A", type=["bmp", "png", "jpg", "jpeg"],
                               key="match_a")
    with cols[1]:
        f_b = st.file_uploader("Obraz B", type=["bmp", "png", "jpg", "jpeg"],
                               key="match_b")

    max_shift = st.slider("Maksymalne przesunięcie (kompensacja rotacji)",
                          0, 30, 8)

    if f_a is None or f_b is None:
        st.info("Wgraj oba obrazy, aby porównać kody.")
    else:
        img_a = _load_pil(f_a)
        img_b = _load_pil(f_b)
        try:
            res_a = _process(img_a)
            res_b = _process(img_b)
        except Exception as exc:
            st.error(f"Błąd przetwarzania: {exc}")
        else:
            cmp = compare(res_a.code, res_b.code, max_shift=max_shift)

            cols2 = st.columns(2)
            with cols2[0]:
                st.image(_annotate_segmentation(img_a, res_a),
                         caption="A - segmentacja",
                         use_container_width=True)
                st.image(_unwrap_with_mask(res_a.unwrap),
                         caption="A - rozwinięcie",
                         use_container_width=True)
            with cols2[1]:
                st.image(_annotate_segmentation(img_b, res_b),
                         caption="B - segmentacja",
                         use_container_width=True)
                st.image(_unwrap_with_mask(res_b.unwrap),
                         caption="B - rozwinięcie",
                         use_container_width=True)

            st.markdown("### Wynik porównania")
            metric_cols = st.columns(3)
            metric_cols[0].metric("HD bez rotacji",
                                  f"{cmp['hd_no_rotation']:.4f}")
            metric_cols[1].metric("HD min (z rotacją)",
                                  f"{cmp['hd_min']:.4f}")
            metric_cols[2].metric("Najlepsze przesunięcie",
                                  f"{cmp['best_shift']}")

            verdict = ("✅ Te same tęczówki (HD < 0.32)" if cmp["hd_min"] < 0.32
                       else "❌ Różne tęczówki (HD ≥ 0.32)")
            st.info(verdict)


st.sidebar.markdown("---")
st.sidebar.caption("Ryszard Czarnecki, Krzysztof Osiński")
