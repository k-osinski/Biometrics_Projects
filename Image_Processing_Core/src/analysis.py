import numpy as np
from basic_operations import to_grayscale
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACCENT = "#4A9EE0"
TEXT_COLOR = "#CCCCCC"
GRID_COLOR = "#333333"


def _style_ax(fig, ax):
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.tick_params(colors=TEXT_COLOR, which="both")
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def compute_histogram(img: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 3))

    if len(img.shape) == 2:
        hist, _ = np.histogram(img, bins=256, range=(0, 256))
        ax.fill_between(range(256), hist, color=ACCENT, alpha=0.6)
        ax.set_xlim([0, 255])
    else:
        colors = ("#FF6B6B", "#51CF66", "#5C9AFF")
        labels = ("Czerwony", "Zielony", "Niebieski")
        for i, (col, label) in enumerate(zip(colors, labels)):
            hist, _ = np.histogram(img[:, :, i], bins=256, range=(0, 256))
            ax.plot(hist, color=col, label=label, alpha=0.8)
        ax.set_xlim([0, 255])
        ax.legend(fontsize=8, facecolor="none", edgecolor="none", labelcolor=TEXT_COLOR)

    ax.set_xlabel("Wartość piksela")
    ax.set_ylabel("Liczba pikseli")
    ax.set_title("Histogram")
    _style_ax(fig, ax)
    plt.tight_layout()
    return fig


def horizontal_projection(img: np.ndarray):
    if len(img.shape) == 3:
        gray = to_grayscale(img)
    else:
        gray = img

    projection = np.sum(gray, axis=1)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(range(len(projection)), projection, height=1, color=ACCENT, alpha=0.7)
    ax.set_ylim([len(projection), 0])
    ax.set_xlabel("Suma jasności")
    ax.set_ylabel("Wiersz")
    ax.set_title("Projekcja pozioma")
    _style_ax(fig, ax)
    plt.tight_layout()
    return fig


def vertical_projection(img: np.ndarray):
    if len(img.shape) == 3:
        gray = to_grayscale(img)
    else:
        gray = img

    projection = np.sum(gray, axis=0)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(projection)), projection, width=1, color=ACCENT, alpha=0.7)
    ax.set_xlim([0, len(projection)])
    ax.set_xlabel("Kolumna")
    ax.set_ylabel("Suma jasności")
    ax.set_title("Projekcja pionowa")
    _style_ax(fig, ax)
    plt.tight_layout()
    return fig
