# File for main app code

import streamlit as st
from PIL import Image
import numpy as np

st.title("Biometria - Projekt 1: Przetwarzanie Obrazów")

# Przesyłanie pliku
uploaded_file = st.file_uploader("Wybierz obraz...", type=['jpg', 'png', 'bmp'])

if uploaded_file is not None:
    # Konwersja do tablicy NumPy (ręczne operacje na pikselach)
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(image, caption='Oryginalny obraz', use_column_width=True)

    # Menu operacji
    option = st.selectbox(
        'Wybierz operację:',
        ['Oryginał', 'Odcienie szarości', 'Jasność', 'Negatyw', 'Binaryzacja']
    )

    # Tutaj będziemy wywoływać Twoje funkcje z src/
    st.write(f"Wybrano: {option}")