# Biometria - Projekt 1: Przetwarzanie Obrazów

Projekt realizowany w ramach zajęć z Biometrii (luty 2026), skupiający się na ręcznej implementacji algorytmów przetwarzania obrazu bez użycia gotowych bibliotek graficznych.

## O projekcie
* **Cel:** Implementacja operacji na pikselach oraz filtrów splotowych "od zera" w języku Python.
* **Interfejs:** Aplikacja okienkowa zbudowana z wykorzystaniem biblioteki Streamlit.

## Zawartość projektu
* **Operacje na pikselach:** Konwersja do odcieni szarości (różne warianty), korekta jasności i kontrastu, negatyw oraz binaryzacja.
* **Filtry graficzne:** Uśredniający, Gaussa oraz wyostrzający z możliwością definiowania własnych wag.
* **Analiza danych:** Generowanie histogramów oraz projekcji pionowych i poziomych obrazu.
* **Wykrywanie krawędzi:** Implementacja operatorów Sobela oraz krzyża Robertsa.

## Struktura katalogów
* `/src` – autorskie implementacje algorytmów (logika biznesowa).
* `/app` – kod źródłowy aplikacji Streamlit.
* `/notebooks` – proces prototypowania i testy algorytmów.
* `/data` – przykładowe obrazy testowe.
