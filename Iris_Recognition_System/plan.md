# Plan Implementacji: Projekt 2 - Rozpoznawanie człowieka na podstawie obrazu tęczówki 

## Struktura Katalogów
Główny folder projektu: `Biometrics_Projects/Iris_Recognition_System/`

* `src/`
    * `core/` - Zależności i operacje bazowe (częściowo importowane z Projektu 1).
        * `morphology.py` - Operacje morfologiczne (erozja, dylatacja, otwarcie, zamknięcie).
        * `projections.py` - Projekcje pionowe i poziome.
    * `segmentation/` - Moduł odpowiedzialny za wyodrębnienie tęczówki.
        * `pupil_detector.py` - Detekcja granic i środka źrenicy.
        * `iris_detector.py` - Detekcja zewnętrznej granicy tęczówki.
    * `normalization/`
        * `unwrapper.py` - Rozwinięcie tęczówki do prostokąta.
    * `encoding/` - Generowanie kodu tęczówki (Ocena 5.0).
        * `band_extractor.py` - Podział na pasy i wyciąganie uśrednionych wartości.
        * `gabor_filter.py` - Implementacja transformaty falkowej Gabora.
    * `matching/`
        * `matcher.py` - Porównywanie kodów.
    * `app/`
        * `main.py` - Główny skrypt integrujący i aplikacja okienkowa.

---

## 1. Moduł: `core/morphology.py` (Nowe operacje z Wykładów 4 i 5)
Implementacja operacji morfologicznych dla obrazów binarnych, gdzie piksele należące do obiektu mają wartość 1, a tło ma wartość 0. Wymagane jest użycie elementu strukturalnego $B$ (np. $3\times3$).

* **Dylatacja:** Rozszerza obiekt i wypełnia dziury. 
    * Wzór: $A\oplus B=\{p:p=a+b,a\in A,b\in B\}$.
* **Erozja:** * Wzór: $A\ominus B=\{a:a+b\in A,dla~ka\dot{z}dego~b\in B\}$.
* **Otwarcie:** Usuwa drobne szczegóły i wygładza kontury.
    * Wzór: $(A\ominus B)\oplus B$ (erozja, a potem dylatacja).
* **Zamknięcie:** Wypełnia małe dziury i wygładza linię brzegową.
    * Wzór: $(A\oplus B)\ominus B$ (dylatacja, a potem erozja).

## 2. Moduł: `segmentation/pupil_detector.py` oraz `iris_detector.py`
Proces segmentacji (na ocenę 4.0).

* **Krok 1: Przekształcenie do szarości i obliczenie bazy progowania:** * Wzór na bazowy próg binaryzacji $P$: $P = \sum_{i=0}^{h-1}\sum_{j=0}^{w-1}\frac{A(i,j)}{h\cdot w}$.
    * Gdzie $h$ to wysokość, $w$ to szerokość, a $A(i,j)$ to jasność piksela szarego.
* **Krok 2: Binaryzacja dla źrenicy:** * Wzór na próg dla źrenicy: $P_P = \frac{P}{x_P}$.
    * Zmienna $x_P$ musi zostać wyznaczona eksperymentalnie.
* **Krok 3: Czyszczenie morfologiczne:**
    * Eliminacja "dziur" wewnątrz źrenicy operacjami morfologicznymi (zamknięcie/dylatacja).
    * Eliminacja elementów poza źrenicą (szumu) operacjami morfologicznymi (otwarcie/erozja).
* **Krok 4: Znalezienie środka i promienia źrenicy:**
    * Użycie projekcji pionowej i poziomej na oczyszczonym obrazie binarnym.
    * Szukamy maksymalnych wartości na projekcjach, aby wyznaczyć środek.
    * Na podstawie projekcji wyznaczamy również promień źrenicy.
* **Krok 5: Detekcja granicy tęczówki:**
    * Algorytm analogiczny do źrenicy, ale z innym progiem binaryzacji $P_I = \frac{P}{x_I}$.
    * Wymagany będzie inny dobór operacji morfologicznych.

## 3. Moduł: `normalization/unwrapper.py`
* **Rozwinięcie tęczówki:**
    * Należy rozwinąć wyizolowaną tęczówkę do prostokąta.
    * Wymaga to zamiany współrzędnych biegunowych na kartezjańskie.

## 4. Moduł: `encoding/band_extractor.py` i `gabor_filter.py`
Rozpoznawanie na podstawie rozwiniętej tęczówki - Algorytm Daugmana (na ocenę 5.0).

* **Ekstrakcja pasów:**
    * Podział prostokąta tęczówki na 8 współśrodkowych/radialnych pasów.
    * **Ważne:** Aby uniknąć powiek i rzęs, pasy należy ograniczyć od góry i od dołu.
    * Pasy muszą mieć zadaną, mniej więcej taką samą długość/liczbę pikseli.
    * Każdy z 8 pasów dzielimy na 128 równoodległych punktów/odcinków.
* **Wyznaczanie średniej:**
    * Dla każdego ze 128 punktów wyznaczana jest średnia intensywność pikseli z użyciem okna Gaussa.
    * Należy uśredniać w kierunku radialnym (wskutek silnej korelacji na tym samym promieniu).
    * Wynik: Każdy pas staje się wektorem 1D (128 punktów to 128 liczb).
* **Transformata falkowa Gabora:** * Aplikowana do każdego punktu sygnału 1D.
    * **Krytyczna uwaga do wzoru:** Wzór określający parametr $\sigma$ należy implementować jako $\sigma=\frac{1}{2}\pi f$. 
    * **Błąd:** Zastosowanie błędnej wersji $\sigma=1/(2\pi f)$ powoduje umieszczenie wszystkich współczynników w I i IV ćwiartce układu zespolonego, co uniemożliwia prawidłowe zakodowanie.
    * Częstotliwość falki $f$ powinna być parametrem konfigurowalnym w aplikacji (dobierana eksperymentalnie, zakres np. od $1/128$ do $\pi$).
    * Wynikiem jest wyznaczony "kod tęczówki".

## 5. Moduł: `matching/matcher.py`
* Porównywanie wygenerowanych kodów tęczówek odbywa się np. na podstawie odległości Hamminga.
* Zalecane zbiory danych do testów: MMU Iris Dataset.