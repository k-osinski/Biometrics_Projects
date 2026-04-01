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
from src.thresholding import binarize, binarize_otsu, threshold_bernsen, threshold_niblack, multi_threshold
from src.filters import mean_filter, gaussian_filter, sharpen_filter, custom_filter
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

# Inicjalizacja obrazu wynikowego
result = img_array.copy()

st.sidebar.header("Potok Przetwarzania")
st.sidebar.caption("Operacje są nakładane z góry na dół.")

# --- KONWERSJA I ODCIENIE SZAROŚCI ---
with st.sidebar.expander("1. Konwersja kolorów", expanded=True):
    use_gray = st.checkbox("Konwertuj do skali szarości")
    if use_gray:
        method = st.selectbox(
            "Metoda konwersji:",
            ["Luminancja (BT.601)", "Luminancja (BT.709)", "Średnia arytmetyczna", "Desaturacja"]
        )
        method_map = {
            "Luminancja (BT.601)": "luminance_bt601",
            "Luminancja (BT.709)": "luminance_bt709",
            "Średnia arytmetyczna": "average",
            "Desaturacja": "desaturation"
        }
        result = to_grayscale(result, method=method_map[method])

# --- TRANSFORMACJE PUNKTOWE ---
with st.sidebar.expander("2. Transformacje punktowe", expanded=False):
    use_brightness_contrast = st.checkbox("Korekta Jasności / Kontrastu")
    if use_brightness_contrast:
        b_val = st.slider("Wartość jasności", -150, 150, 0, step=5)
        c_val = st.slider("Współczynnik kontrastu", 0.1, 3.0, 1.0, step=0.1)
        if b_val != 0:
            result = adjust_brightness(result, b_val)
        if c_val != 1.0:
            result = adjust_contrast(result, c_val)
            
    use_gamma = st.checkbox("Korekta Gamma")
    if use_gamma:
        alpha = st.slider("Współczynnik Gamma (α)", 0.1, 5.0, 1.0, step=0.1)
        result = adjust_gamma(result, alpha)
        
    if st.checkbox("Logarytmowanie"):
        result = adjust_logarithmic(result)
        
    if st.checkbox("Rozszerzenie zakresu (Liniowe)"):
        result = stretch_range(result)
        
    if st.checkbox("Wyrównanie histogramu"):
        result = equalize_histogram(result)
        
    if st.checkbox("Negatyw"):
        result = negative(result)

# --- FILTRY I KRAWĘDZIE ---
with st.sidebar.expander("3. Filtracja przestrzenna", expanded=False):
    filter_choice = st.selectbox(
        "Zastosuj filtr/krawędzie:",
        ["Brak", "Uśredniający", "Gaussa", "Wyostrzający", "Krzyż Robertsa", "Operator Sobela", "Własny"]
    )
    
    if filter_choice == "Uśredniający":
        k = st.slider("Rozmiar jądra", 3, 15, 3, step=2)
        result = mean_filter(result, k)
    elif filter_choice == "Gaussa":
        k = st.slider("Rozmiar jądra", 3, 15, 3, step=2)
        sigma = st.slider("Sigma", 0.1, 5.0, 1.0, step=0.1)
        result = gaussian_filter(result, k, sigma)
    elif filter_choice == "Wyostrzający":
        strength = st.slider("Siła wyostrzania", 0.1, 3.0, 1.0, step=0.1)
        result = sharpen_filter(result, strength)
    elif filter_choice == "Krzyż Robertsa":
        result = roberts_cross(result)
    elif filter_choice == "Operator Sobela":
        result = sobel_operator(result)
    elif filter_choice == "Własny":
        size = st.selectbox("Rozmiar jądra", [3, 5, 7], index=0)
        kernel = np.zeros((size, size), dtype=np.float64)
        st.markdown("Wagi filtra:")
        cols = st.columns(size)
        for i in range(size):
            for j in range(size):
                with cols[j]:
                    kernel[i, j] = st.number_input(f"[{i},{j}]", value=0.0, step=0.1, key=f"kf_{size}_{i}_{j}")
        if st.checkbox("Normalizuj jądro", value=False):
            k_sum = np.sum(kernel)
            if k_sum != 0: kernel = kernel / k_sum
        result = custom_filter(result, kernel)

# --- SEGMENTACJA ---
with st.sidebar.expander("4. Segmentacja (Binaryzacja)", expanded=False):
    use_thresh = st.checkbox("Zastosuj progowanie")
    if use_thresh:
        strategy = st.selectbox("Strategia:", ["Globalna (Ręczna)", "Globalna (Otsu)", "Lokalna (Bernsen)", "Adaptacyjna (Niblack)", "Wieloprogowanie"])
        
        if strategy == "Globalna (Ręczna)":
            t = st.slider("Próg", 0, 255, 128)
            result = binarize(result, t)
        elif strategy == "Globalna (Otsu)":
            result = binarize_otsu(result)
        elif strategy == "Lokalna (Bernsen)":
            w = st.slider("Rozmiar okna (B)", 3, 51, 15, step=2)
            result = threshold_bernsen(result, w)
        elif strategy == "Adaptacyjna (Niblack)":
            w = st.slider("Rozmiar okna (N)", 3, 51, 15, step=2)
            k_param = st.slider("Parametr k", -1.0, 1.0, -0.2, step=0.1)
            result = threshold_niblack(result, w, k_param)
        elif strategy == "Wieloprogowanie":
            range_val = st.slider("Zakres jasności", 0, 255, (50, 200))
            result = multi_threshold(result, range_val[0], range_val[1])

# --- WYŚWIETLANIE WYNIKÓW ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Oryginał")
    st.image(img_array, use_container_width=True)

with col2:
    st.subheader("Wynik (po potoku operacji)")
    if len(result.shape) == 2:
        st.image(result, use_container_width=True, clamp=True)
    else:
        st.image(result, use_container_width=True)

st.markdown("---")

# Zapis obrazu
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
st.subheader("Analiza końcowa")

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