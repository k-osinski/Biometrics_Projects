# Projekt 3 - Porównanie algorytmów ścieniania odcisków palców

Aplikacja Streamlit porównuje dwa algorytmy ścieniania zastosowane do odcisków
palców:

1. **Szkieletyzacja morfologiczna** - klasyczny wzór Lantuejoula
   (prezentowany na wykładzie).
2. **K3M** - algorytm Khalida Saeeda, Marka Tabędzkiego, Mariusza Rybnika i
   Marcina Adamskiego (2010), iteracyjny i sekwencyjny, generujący szkielet
   o szerokości jednego piksela.

Aplikacja wykonuje preprocessing (CLAHE, adaptacyjna binaryzacja, segmentacja
i operacje morfologiczne łączące przerwy), oba ścieniania, postprocessing i
detekcję minucji metodą Crossing Number (zakończenia i bifurkacje).

## Wymagania

- Python 3.10+
- `streamlit`, `opencv-python`, `numpy`, `pandas`

Można zainstalować całość poleceniem:

```
pip install -r requirements.txt
```

## Uruchomienie

W folderze `FingerprintDetector`:

```
streamlit run app.py
```

Aplikacja oferuje dwie zakładki:

- **Pojedynczy odcisk** - interaktywny widok krok po kroku dla wybranego
  odcisku (preprocessing → ścienianie → postprocessing → minucje).
- **Statystyki zbiorcze** - przetworzenie wszystkich 30 odcisków z folderu
  `Odciski`, tabela porównawcza i wykresy.

## Struktura

```
FingerprintDetector/
├── app.py                  # aplikacja Streamlit
├── thinning/
│   ├── morphological.py    # szkieletyzacja morfologiczna (Lantuejoul)
│   ├── k3m.py              # algorytm K3M (sequential / parallel)
│   ├── postprocessing.py   # poprawa szkieletu, łączenie przerw
│   └── minutiae.py         # detekcja minucji metodą CN
├── utils/
│   └── preprocessing.py    # CLAHE, binaryzacja, segmentacja
├── Odciski/                # pliki .bmp
├── requirements.txt
└── README.md
```
