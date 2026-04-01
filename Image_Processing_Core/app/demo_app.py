import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import streamlit as st
import numpy as np
from PIL import Image

from src.basic_operations import (
    to_grayscale, adjust_brightness, adjust_contrast, negative,
    adjust_gamma, adjust_logarithmic, equalize_histogram, stretch_range 
    )
from src.thresholding import binarize, binarize_otsu
from src.filters import mean_filter, gaussian_filter, sharpen_filter
from src.edge_detection import roberts_cross, sobel_operator
from src.analysis import compute_histogram, horizontal_projection, vertical_projection

st.set_page_config(page_title="Biometria – Przetwarzanie Obrazów", layout="wide")
st.title("Biometria – Projekt 1: Przetwarzanie Obrazów")

uploaded_file = st.file_uploader("Wgraj obraz", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded_file is None:
    st.info("Wgraj obraz, aby rozpocząć przetwarzanie.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
img_array = np.array(image)

st.sidebar.header("Operacje")

category = st.sidebar.radio(
    "Kategoria:",
    ["Operacje na pikselach", "Filtry graficzne", "Wykrywanie krawędzi", "Filtr własny"],
)

if category != "Filtr własny":
    for key in list(st.session_state.keys()):
        if key.startswith("kf_") or key == "kernel_size":
            del st.session_state[key]

result = None

if category == "Operacje na pikselach":
    operation = st.sidebar.selectbox(
        "Operacja:",
        ["Oryginał", "Odcienie szarości", "Jasność", "Kontrast (Liniowy)",
         "Korekta Gamma (Potęgowanie)", "Logarytmowanie", 
         "Wyrównanie histogramu", "Negatyw", "Binaryzacja"],
    )

    if operation == "Oryginał":
        result = img_array
    elif operation == "Odcienie szarości":
        method = st.sidebar.selectbox(
            "Metoda konwersji:",
            ["Luminancja (BT.601)", "Luminancja (BT.709)", "Średnia arytmetyczna", "Desaturacja"]
        )
        method_map = {
            "Luminancja (BT.601)": "luminance_bt601",
            "Luminancja (BT.709)": "luminance_bt709",
            "Średnia arytmetyczna": "average",
            "Desaturacja": "desaturation"
        }
        result = to_grayscale(img_array, method=method_map[method])
    elif operation == "Jasność":
        value = st.sidebar.slider("Wartość jasności", -150, 150, 0, step=5)
        result = adjust_brightness(img_array, value)
    elif operation == "Kontrast (Liniowy)":
        factor = st.sidebar.slider("Współczynnik kontrastu", 0.1, 3.0, 1.0, step=0.1)
        result = adjust_contrast(img_array, factor)
    elif operation == "Korekta Gamma (Potęgowanie)":
        alpha = st.sidebar.slider("Współczynnik Gamma (\u03B1)", 0.1, 5.0, 1.0, step=0.1)
        result = adjust_gamma(img_array, alpha)
    elif operation == "Logarytmowanie":
        result = adjust_logarithmic(img_array)
    elif operation == "Rozszerzenie zakresu (Liniowe)":
        result = stretch_range(img_array)
    elif operation == "Wyrównanie histogramu":
        result = equalize_histogram(img_array)
    elif operation == "Negatyw":
        result = negative(img_array)
    elif operation == "Binaryzacja":
        strategy = st.sidebar.selectbox("Strategia:", 
                                        ["Globalna (Ręczna)", "Globalna (Otsu)", 
                                        "Lokalna (Bernsen)", "Adaptacyjna (Niblack)", 
                                        "Wieloprogowanie"])
        if strategy == "Globalna (Ręczna)":
            t = st.sidebar.slider("Próg", 0, 255, 128)
            result = binarize(img_array, t)
        elif strategy == "Globalna (Otsu)":
            result = binarize_otsu(img_array)
        elif strategy == "Lokalna (Bernsen)":
            w = st.sidebar.slider("Rozmiar okna", 3, 51, 15, step=2)
            result = threshold_bernsen(img_array, w)
        elif strategy == "Adaptacyjna (Niblack)":
            w = st.sidebar.slider("Rozmiar okna", 3, 51, 15, step=2)
            k = st.sidebar.slider("Parametr k", -1.0, 1.0, -0.2, step=0.1)
            result = threshold_niblack(img_array, w, k)
        elif strategy == "Wieloprogowanie":
            range_val = st.sidebar.slider("Zakres jasności", 0, 255, (50, 200))
            result = multi_threshold(img_array, range_val[0], range_val[1])

elif category == "Filtry graficzne":
    filter_type = st.sidebar.selectbox(
        "Filtr:",
        ["Uśredniający", "Gaussa", "Wyostrzający"],
    )

    if filter_type == "Uśredniający":
        k = st.sidebar.slider("Rozmiar jądra", 3, 15, 3, step=2)
        result = mean_filter(img_array, k)
    elif filter_type == "Gaussa":
        k = st.sidebar.slider("Rozmiar jądra", 3, 15, 3, step=2)
        sigma = st.sidebar.slider("Sigma", 0.1, 5.0, 1.0, step=0.1)
        result = gaussian_filter(img_array, k, sigma)
    elif filter_type == "Wyostrzający":
        strength = st.sidebar.slider("Siła wyostrzania", 0.1, 3.0, 1.0, step=0.1)
        result = sharpen_filter(img_array, strength)

elif category == "Wykrywanie krawędzi":
    edge_type = st.sidebar.selectbox(
        "Metoda:",
        ["Krzyż Robertsa", "Operator Sobela"],
    )

    if edge_type == "Krzyż Robertsa":
        result = roberts_cross(img_array)
    elif edge_type == "Operator Sobela":
        result = sobel_operator(img_array)

elif category == "Filtr własny":
    from src.filters import custom_filter

    st.sidebar.subheader("Edytor jądra filtra")
    size = st.sidebar.selectbox("Rozmiar jądra", [3, 5, 7], index=0, key="kernel_size")

    with st.sidebar.form("kernel_form"):
        st.markdown("Wagi filtra:")
        kernel = np.zeros((size, size), dtype=np.float64)
        cols = st.columns(size)
        for i in range(size):
            for j in range(size):
                with cols[j]:
                    kernel[i, j] = st.number_input(
                        f"[{i},{j}]", value=0.0, step=0.1, format="%.1f",
                        key=f"kf_{size}_{i}_{j}"
                    )

        normalize = st.checkbox("Normalizuj jądro", value=False)
        submitted = st.form_submit_button("Zastosuj filtr")

    if normalize:
        kernel_sum = np.sum(kernel)
        if kernel_sum != 0:
            kernel = kernel / kernel_sum

    st.sidebar.markdown("**Podgląd jądra:**")
    st.sidebar.dataframe(kernel, hide_index=True)

    result = custom_filter(img_array, kernel)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Oryginał")
    st.image(img_array, use_container_width=True)

with col2:
    st.subheader("Wynik")
    if result is not None:
        if len(result.shape) == 2:
            st.image(result, use_container_width=True, clamp=True)
        else:
            st.image(result, use_container_width=True)

if result is not None:
    st.markdown("---")

    if len(result.shape) == 2:
        result_pil = Image.fromarray(result, mode="L")
    else:
        result_pil = Image.fromarray(result)

    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")

    st.download_button(
        label="Zapisz przetworzony obraz",
        data=buf.getvalue(),
        file_name="wynik.png",
        mime="image/png",
    )

    st.markdown("---")
    st.subheader("Analiza")

    tab1, tab2, tab3 = st.tabs(["Histogram", "Projekcja pozioma", "Projekcja pionowa"])

    with tab1:
        fig_hist = compute_histogram(result)
        st.pyplot(fig_hist)

    with tab2:
        fig_h = horizontal_projection(result)
        st.pyplot(fig_h)

    with tab3:
        fig_v = vertical_projection(result)
        st.pyplot(fig_v)

st.sidebar.markdown("---")
st.sidebar.caption("Biometria 2026 — Projekt 1")
