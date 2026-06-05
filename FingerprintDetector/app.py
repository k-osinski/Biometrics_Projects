"""
Aplikacja prezentuje porównanie dwóch algorytmów ścieniania:
  - Szkieletyzacja morfologiczna
  - K3M

Dodatkowo wykonuje detekcję minucji na uzyskanych
szkieletach i pozwala porównać jakość obu metod.

Uruchomienie:
    streamlit run app.py
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from thinning.k3m import k3m_thin
from thinning.minutiae import (count_by_kind, deduplicate_minutiae,
                                detect_minutiae, draw_minutiae)
from thinning.morphological import morphological_skeleton
from thinning.postprocessing import postprocess_skeleton
from utils.preprocessing import load_fingerprint, preprocess_fingerprint


st.set_page_config(
    page_title="Projekt 3 - Ścienianie odcisków",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
ODCISKI_DIR = BASE_DIR / "Odciski"


def list_fingerprints():
    if not ODCISKI_DIR.exists():
        return []
    files = sorted(ODCISKI_DIR.glob("*.bmp"),
                   key=lambda p: int(p.stem) if p.stem.isdigit() else 999)
    return files


@st.cache_data(show_spinner=False)
def cached_preprocess(path_str, clahe_clip, clahe_tile, block_size, C,
                      close_kernel, open_kernel, do_segmentation):
    gray = load_fingerprint(path_str)
    return preprocess_fingerprint(
        gray,
        clahe_clip=clahe_clip, clahe_tile=clahe_tile,
        block_size=block_size, C=C,
        close_kernel=close_kernel, open_kernel=open_kernel,
        do_segmentation=do_segmentation,
    )


def to_rgb(arr):
    if arr is None:
        return None
    a = arr
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    elif a.max() <= 1:
        a = (a * 255).astype(np.uint8)
    else:
        a = a.astype(np.uint8)
    if a.ndim == 2:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    if a.shape[2] == 3:
        return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    return a


st.sidebar.title("⚙️ Ustawienia")
files = list_fingerprints()
if not files:
    st.error(f"Brak plików .bmp w folderze: {ODCISKI_DIR}")
    st.stop()
file_names = [p.name for p in files]
selected_name = st.sidebar.selectbox("Wybierz odcisk:", file_names, index=0,
                                       key="selected_file")
selected_path = ODCISKI_DIR / selected_name

st.sidebar.markdown("---")
st.sidebar.subheader("Preprocessing")
clahe_clip = st.sidebar.slider("CLAHE clipLimit", 0.5, 8.0, 3.0, 0.5)
clahe_tile = st.sidebar.slider("CLAHE tile size", 2, 16, 8, 1)
block_size = st.sidebar.slider("Adaptive block size", 7, 41, 17, 2)
C_val = st.sidebar.slider("Adaptive C", 0, 20, 7, 1)
close_kernel = st.sidebar.slider("Close kernel (łączenie przerw)", 1, 7, 3, 1)
open_kernel = st.sidebar.slider("Open kernel (usuw. szumu)", 1, 7, 3, 1)
do_segmentation = st.sidebar.checkbox("Segmentacja palca", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("K3M")
k3m_mode = st.sidebar.radio("Tryb K3M:", ["parallel", "sequential"], index=0,
                              help="'sequential' według PDF, 'parallel' szybsze")
k3m_max_iters = st.sidebar.slider("Max iteracji K3M", 5, 80, 30, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("Postprocessing szkieletu")
spur_length = st.sidebar.slider("Długość usuwanych kolców", 0, 10, 3, 1)
min_component = st.sidebar.slider("Min. rozmiar składowej", 0, 50, 8, 1)
gap_size = st.sidebar.slider("Połącz przerwy (px)", 0, 10, 4, 1)
do_bridge = st.sidebar.checkbox("Łącz przerwane linie", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Minucje")
border_margin = st.sidebar.slider("Odstęp od brzegu (px)", 0, 40, 15, 1)
dedup_distance = st.sidebar.slider(
    "Min. odległość między minucjami (px)", 0, 30, 8, 1,
    help="Minucje bliżej niż X px są odrzucane (NMS). Bifurkacje preferowane "
         "nad zakończeniami. 0 = wyłącz.")
dedup_density_min = st.sidebar.slider(
    "Filtr gęstwiny zakończeń", 0, 10, 3, 1,
    help="Jeżeli zakończenie ma >=N innych zakończeń w promieniu, odrzucamy "
         "je jako szum. 0 = wyłącz.")


st.title("Projekt 3 - Porównanie algorytmów ścieniania odcisków palców")
st.markdown(
    "Implementacja **szkieletyzacji morfologicznej** "
    "(algorytm Lantuejoula z wykładu) oraz algorytmu **K3M**. "
    "Aplikacja wykonuje preprocessing, dwa różne ścieniania, postprocessing "
    "i detekcję minucji metodą Crossing Number."
)

tab_interactive, tab_batch = st.tabs(
    ["Pojedynczy odcisk", "Statystyki zbiorcze"])


# Tab 1
with tab_interactive:
    pre = cached_preprocess(
        str(selected_path), clahe_clip, clahe_tile, block_size, C_val,
        close_kernel, open_kernel, do_segmentation,
    )

    st.subheader("Krok 1 - Preprocessing")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image(to_rgb(pre["gray"]), caption="Oryginał", width="stretch")
    with c2:
        st.image(to_rgb(pre["enhanced"]), caption="Po CLAHE", width="stretch")
    with c3:
        st.image(to_rgb(pre["binary_uint8"]),
                 caption=f"Binaryzacja ({int(pre['binary'].sum())} px)",
                 width="stretch")
    with c4:
        st.image(to_rgb(pre["mask"] * 255), caption="Maska palca",
                 width="stretch")

    st.subheader("Krok 2 - Ścienianie")
    t0 = time.perf_counter()
    sk_morph = morphological_skeleton(pre["binary"])
    t_morph = time.perf_counter() - t0

    t0 = time.perf_counter()
    sk_k3m = k3m_thin(pre["binary"], mode=k3m_mode, max_iters=k3m_max_iters)
    t_k3m = time.perf_counter() - t0

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Szkieletyzacja morfologiczna**  "
                    f"⏱ {t_morph*1000:.0f} ms · pikseli: {int(sk_morph.sum())}")
        st.image(to_rgb(sk_morph * 255), width="stretch",
                 caption="Wzór Lantuejoula")
    with c2:
        st.markdown(f"**K3M ({k3m_mode})**  "
                    f"⏱ {t_k3m*1000:.0f} ms · pikseli: {int(sk_k3m.sum())}")
        st.image(to_rgb(sk_k3m * 255), width="stretch",
                 caption="K3M (Saeed i in., 2010)")

    st.subheader("Krok 3 - Postprocessing")
    t0 = time.perf_counter()
    post_morph = postprocess_skeleton(
        sk_morph, spur_length=spur_length,
        min_component_size=min_component, gap_size=gap_size, do_bridge=do_bridge)
    t_post_morph = time.perf_counter() - t0
    t0 = time.perf_counter()
    post_k3m = postprocess_skeleton(
        sk_k3m, spur_length=spur_length,
        min_component_size=min_component, gap_size=gap_size, do_bridge=do_bridge)
    t_post_k3m = time.perf_counter() - t0

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Morfologiczny po postprocessing**  "
                    f"⏱ {t_post_morph*1000:.0f} ms · "
                    f"pikseli: {int(post_morph.sum())}")
        st.image(to_rgb(post_morph * 255), width="stretch")
    with c2:
        st.markdown(f"**K3M po postprocessing**  "
                    f"⏱ {t_post_k3m*1000:.0f} ms · "
                    f"pikseli: {int(post_k3m.sum())}")
        st.image(to_rgb(post_k3m * 255), width="stretch")

    st.subheader("Krok 4 - Detekcja minucji (Crossing Number)")
    mins_morph_raw = detect_minutiae(post_morph, mask=pre["mask"],
                                      border=border_margin)
    mins_k3m_raw = detect_minutiae(post_k3m, mask=pre["mask"],
                                    border=border_margin)
    mins_morph = deduplicate_minutiae(
        mins_morph_raw, min_distance=dedup_distance,
        density_threshold=dedup_density_min)
    mins_k3m = deduplicate_minutiae(
        mins_k3m_raw, min_distance=dedup_distance,
        density_threshold=dedup_density_min)
    cm_morph = count_by_kind(mins_morph)
    cm_k3m = count_by_kind(mins_k3m)
    raw_morph_n = len(mins_morph_raw)
    raw_k3m_n = len(mins_k3m_raw)

    bg = cv2.cvtColor(pre["gray"], cv2.COLOR_GRAY2BGR)
    vis_morph = draw_minutiae(post_morph, mins_morph, background=bg, radius=4)
    vis_k3m = draw_minutiae(post_k3m, mins_k3m, background=bg, radius=4)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"**Morfologiczny** (przed dedup: {raw_morph_n})  "
            f"🔴 zakończeń: {cm_morph['ending']}  ·  "
            f"🟢 bifurkacji: {cm_morph['bifurcation']}")
        st.image(cv2.cvtColor(vis_morph, cv2.COLOR_BGR2RGB), width="stretch",
                 caption="Czerwone kółka = zakończenia, "
                          "zielone kwadraty = bifurkacje")
    with c2:
        st.markdown(
            f"**K3M** (przed dedup: {raw_k3m_n})  "
            f"🔴 zakończeń: {cm_k3m['ending']}  ·  "
            f"🟢 bifurkacji: {cm_k3m['bifurcation']}")
        st.image(cv2.cvtColor(vis_k3m, cv2.COLOR_BGR2RGB), width="stretch")

    st.subheader("Podsumowanie")
    summary = pd.DataFrame({
        "Metoda": ["Morfologiczna", "K3M"],
        "Czas ścieniania [ms]": [round(t_morph * 1000, 1),
                                  round(t_k3m * 1000, 1)],
        "Pikseli (raw)": [int(sk_morph.sum()), int(sk_k3m.sum())],
        "Pikseli (post)": [int(post_morph.sum()), int(post_k3m.sum())],
        "Minucje przed dedup": [raw_morph_n, raw_k3m_n],
        "Zakończeń": [cm_morph["ending"], cm_k3m["ending"]],
        "Bifurkacji": [cm_morph["bifurcation"], cm_k3m["bifurcation"]],
        "Razem (po dedup)": [cm_morph["ending"] + cm_morph["bifurcation"],
                              cm_k3m["ending"] + cm_k3m["bifurcation"]],
    })
    st.dataframe(summary, hide_index=True, width="stretch")


# Tab 2 - Statystyki zbiorcze
with tab_batch:
    st.subheader("Statystyki zbiorcze (wszystkie odciski)")
    st.write("Przetworzenie wszystkich plików z folderu *Odciski* obydwoma "
             "algorytmami i porównanie wyników.")
    if st.button("▶️ Uruchom przetwarzanie wsadowe", type="primary"):
        rows = []
        progress = st.progress(0.0)
        status = st.empty()
        total = len(files)
        for i, fp in enumerate(files, 1):
            status.text(f"Przetwarzanie {fp.name} ({i}/{total})...")
            pre_b = cached_preprocess(
                str(fp), clahe_clip, clahe_tile, block_size, C_val,
                close_kernel, open_kernel, do_segmentation)
            t0 = time.perf_counter()
            skm = morphological_skeleton(pre_b["binary"])
            tm = time.perf_counter() - t0
            t0 = time.perf_counter()
            sk3 = k3m_thin(pre_b["binary"], mode=k3m_mode,
                            max_iters=k3m_max_iters)
            t3 = time.perf_counter() - t0
            pm = postprocess_skeleton(
                skm, spur_length=spur_length,
                min_component_size=min_component,
                gap_size=gap_size, do_bridge=do_bridge)
            p3 = postprocess_skeleton(
                sk3, spur_length=spur_length,
                min_component_size=min_component,
                gap_size=gap_size, do_bridge=do_bridge)
            mm_raw = detect_minutiae(pm, mask=pre_b["mask"],
                                      border=border_margin)
            m3_raw = detect_minutiae(p3, mask=pre_b["mask"],
                                      border=border_margin)
            mm = count_by_kind(deduplicate_minutiae(
                mm_raw, min_distance=dedup_distance,
                density_threshold=dedup_density_min))
            m3 = count_by_kind(deduplicate_minutiae(
                m3_raw, min_distance=dedup_distance,
                density_threshold=dedup_density_min))
            rows.append({
                "Plik": fp.name,
                "Czas morf [ms]": round(tm * 1000, 1),
                "Czas K3M [ms]": round(t3 * 1000, 1),
                "Px morf skel": int(skm.sum()),
                "Px K3M skel": int(sk3.sum()),
                "Px morf post": int(pm.sum()),
                "Px K3M post": int(p3.sum()),
                "Morf raw": len(mm_raw),
                "K3M raw": len(m3_raw),
                "Morf zakon.": mm["ending"],
                "Morf bifur.": mm["bifurcation"],
                "K3M zakon.": m3["ending"],
                "K3M bifur.": m3["bifurcation"],
            })
            progress.progress(i / total)
        status.empty()
        progress.empty()
        df = pd.DataFrame(rows)
        st.session_state["batch_df"] = df

    if "batch_df" in st.session_state:
        df = st.session_state["batch_df"]
        st.dataframe(df, hide_index=True, width="stretch")

        st.markdown("### Wykresy porównawcze")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Średni czas ścieniania**")
            st.bar_chart(
                df[["Czas morf [ms]", "Czas K3M [ms]"]].mean().to_frame("ms"))
            st.markdown("**Średni rozmiar szkieletu (px)**")
            st.bar_chart(
                df[["Px morf skel", "Px K3M skel"]].mean().to_frame("px"))
        with c2:
            st.markdown("**Średnia liczba minucji (po dedup)**")
            st.bar_chart(pd.DataFrame({
                "n": [df["Morf zakon."].mean(), df["Morf bifur."].mean(),
                       df["K3M zakon."].mean(), df["K3M bifur."].mean()]
            }, index=["Morf zakończenia", "Morf bifurkacje",
                       "K3M zakończenia", "K3M bifurkacje"]))
            st.markdown("**Suma minucji per odcisk (po dedup)**")
            chart = pd.DataFrame({
                "Morfologiczne": df["Morf zakon."] + df["Morf bifur."],
                "K3M": df["K3M zakon."] + df["K3M bifur."],
            })
            chart.index = df["Plik"]
            st.line_chart(chart)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Pobierz wyniki jako CSV", data=csv,
                            file_name="statystyki_scieniania.csv",
                            mime="text/csv")
