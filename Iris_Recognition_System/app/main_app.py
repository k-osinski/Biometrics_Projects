"""
Aplikacja Streamlit - Biometria, Projekt 2 (Rozpoznawanie tęczówki).

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

# pozwala uruchamiać "streamlit run app/main_app.py" z głównego folderu projektu
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
st.title("Biometria – Projekt 2: Rozpoznawanie tęczówki (algorytm Daugmana)")
st.caption("Segmentacja → rozwinięcie → kod tęczówki → odległość Hamminga.")


# --------------------------------------------------------------------------- #
# Sidebar - parametry pipeline'u                                              #
# --------------------------------------------------------------------------- #
st.sidebar.header("Parametry algorytmu")

with st.sidebar.expander("Segmentacja źrenicy", expanded=True):
    x_p = st.slider("X_P (próg dla źrenicy = P / X_P)",
                    min_value=1.5, max_value=6.0, value=3.0, step=0.1,
                    help="Większe X_P -> niższy próg -> wykrywamy "
                         "tylko najczarniejsze piksele.")
    pupil_close = st.slider("Rozmiar zamknięcia (pupil)", 1, 21, 7, 2)
    pupil_open = st.slider("Rozmiar otwarcia (pupil)", 1, 21, 5, 2)

with st.sidebar.expander("Segmentacja tęczówki", expanded=True):
    x_i = st.slider("X_I (próg dla tęczówki = P / X_I)",
                    min_value=0.5, max_value=2.5, value=1.15, step=0.05)

with st.sidebar.expander("Rozwinięcie tęczówki", expanded=False):
    radial_res = st.slider("Wysokość (radial_res)", 16, 128, 64, 8)
    angular_res = st.slider("Szerokość (angular_res)", 64, 512, 256, 32)

with st.sidebar.expander("Kodowanie (Gabor)", expanded=True):
    frequency = st.slider("Częstotliwość falki f", 0.02, 0.5, 0.10, 0.01)
    use_auto_sigma = st.checkbox("σ = π·f / 2 (zalecane przez prowadzącego)",
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
    """Rysuje okrąg źrenicy i tęczówki na oryginalnym obrazie."""
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
    return pil


def _iris_code_visual(code) -> Image.Image:
    """Zamienia kod (num_bands, P, 2) na obraz binarny do podglądu."""
    bits = code.bits.reshape(code.bits.shape[0], -1)  # (bands, P*2)
    arr = (bits * 255).astype(np.uint8)
    # zwiększ rozmiar dla czytelności
    pil = Image.fromarray(arr, mode="L").resize(
        (arr.shape[1] * 2, arr.shape[0] * 16), Image.NEAREST)
    return pil


def _process(img_array: np.ndarray):
    return run_iris_pipeline(
        img_array,
        x_p=x_p, x_i=x_i,
        pupil_close=pupil_close, pupil_open=pupil_open,
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
                             f"Tęczówka: r={result.segmentation.iris.radius}"))
                st.image(result.segmentation.pupil.binary_mask,
                         use_container_width=True,
                         caption="Maska binarna źrenicy (po morfologii)",
                         clamp=True)

            with cols[1]:
                st.subheader("Rozwinięcie tęczówki")
                st.image(result.unwrap.image, use_container_width=True,
                         caption=f"Rozmiar: {result.unwrap.image.shape}")
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
                st.image(res_a.unwrap.image,
                         caption="A - rozwinięcie",
                         use_container_width=True)
            with cols2[1]:
                st.image(_annotate_segmentation(img_b, res_b),
                         caption="B - segmentacja",
                         use_container_width=True)
                st.image(res_b.unwrap.image,
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
st.sidebar.caption("Biometria 2026 — Projekt 2")
