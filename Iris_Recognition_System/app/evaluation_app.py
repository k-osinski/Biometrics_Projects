"""
Skrypt ewaluacyjny - liczy odległości Hamminga dla par kodów wyznaczonych
z całego datasetu MMU i prezentuje wyniki w formie histogramów + krzywych FAR/FRR.
Uruchomienie:
    streamlit run app/evaluation_app.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pipeline import run_iris_pipeline
from src.evaluation import (
    iter_mmu_dataset, intra_inter_distances,
    decision_threshold_metrics,
)


st.set_page_config(page_title="Ewaluacja - MMU Iris", layout="wide")
st.title("Ewaluacja: rozpoznawanie tęczówki na zbiorze MMU")
st.caption("Generuje rozkład odległości Hamminga dla par "
           "intra-class i inter-class oraz krzywe FAR/FRR.")


DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "MMU-Iris-Database"
root_str = st.text_input("Ścieżka do MMU dataset:",
                         value=str(DEFAULT_ROOT))
root = Path(root_str)

cols = st.columns(4)
with cols[0]:
    limit_subjects = st.number_input("Ograniczenie liczby osób", 1, 500, 10)
with cols[1]:
    limit_per_class = st.number_input("Próbek per oko", 1, 20, 5)
with cols[2]:
    max_shift = st.number_input("Max przesunięcie (rotacja)", 0, 30, 8)
with cols[3]:
    eyes = st.multiselect("Oczy", ["left", "right"],
                          default=["left", "right"])

with st.expander("Parametry algorytmu", expanded=False):
    x_p = st.slider("X_P", 1.5, 6.0, 3.0, 0.1)
    x_i = st.slider("X_I", 0.5, 2.5, 1.15, 0.05)
    frequency = st.slider("Częstotliwość Gabora", 0.02, 0.5, 0.10, 0.01)


run = st.button("Uruchom ewaluację", type="primary")
if run:
    if not root.is_dir():
        st.error("Wskazany katalog nie istnieje.")
        st.stop()

    progress = st.progress(0.0, text="Liczenie kodów...")
    codes = []
    labels = []

    images = list(iter_mmu_dataset(root,
                                   eyes=tuple(eyes),
                                   limit_per_class=int(limit_per_class),
                                   limit_subjects=int(limit_subjects)))
    if not images:
        st.error("Brak obrazów w datasecie. Sprawdź ścieżkę.")
        st.stop()

    failures = 0
    for idx, mmu in enumerate(images):
        try:
            res = run_iris_pipeline(mmu.image,
                                    x_p=x_p, x_i=x_i,
                                    gabor_frequency=frequency)
            codes.append(res.code)
            labels.append(mmu.class_label)
        except Exception:
            failures += 1
        progress.progress((idx + 1) / len(images),
                          text=f"Liczenie kodów... {idx + 1}/{len(images)}")

    if not codes:
        st.error(f"Nie udało się policzyć żadnego kodu (błędy: {failures}).")
        st.stop()

    st.success(f"Policzono {len(codes)} kodów (błędy: {failures}).")

    progress2 = st.progress(0.0, text="Liczenie odległości...")
    dist = intra_inter_distances(codes, labels, max_shift=int(max_shift))
    progress2.progress(1.0, text="Gotowe.")

    st.write({
        "Liczba par intra-class": int(dist.intra.size),
        "Liczba par inter-class": int(dist.inter.size),
        "Średnia HD intra": float(np.mean(dist.intra)) if dist.intra.size else None,
        "Średnia HD inter": float(np.mean(dist.inter)) if dist.inter.size else None,
        "Std HD intra": float(np.std(dist.intra)) if dist.intra.size else None,
        "Std HD inter": float(np.std(dist.inter)) if dist.inter.size else None,
    })

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0.0, 1.0, 51)
    if dist.intra.size:
        ax.hist(dist.intra, bins=bins, alpha=0.6,
                label="Intra-class (te same tęczówki)", color="#4A9EE0")
    if dist.inter.size:
        ax.hist(dist.inter, bins=bins, alpha=0.6,
                label="Inter-class (różne tęczówki)", color="#FF6B6B")
    ax.set_xlabel("Znormalizowana odległość Hamminga")
    ax.set_ylabel("Liczba par")
    ax.set_title("Rozkład odległości Hamminga")
    ax.legend()
    st.pyplot(fig)

    thresholds = np.linspace(0.0, 1.0, 101)
    metrics = decision_threshold_metrics(dist, thresholds)
    fars = np.array([m.far for m in metrics])
    frrs = np.array([m.frr for m in metrics])

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(thresholds, fars, label="FAR", color="#FF6B6B")
    ax2.plot(thresholds, frrs, label="FRR", color="#4A9EE0")
    ax2.set_xlabel("Próg odległości Hamminga")
    ax2.set_ylabel("Wskaźnik błędu")
    ax2.set_title("Krzywe FAR / FRR")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    # Equal Error Rate (EER): punkt, w którym FAR = FRR
    diff = np.abs(fars - frrs)
    eer_idx = int(np.argmin(diff))
    st.metric("EER", f"{(fars[eer_idx] + frrs[eer_idx]) / 2:.4f}",
              help=f"Próg @ EER = {thresholds[eer_idx]:.3f}")
