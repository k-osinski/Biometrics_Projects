"""
Pomoce do iteracji po zbiorze MMU Iris Database.

Oczekiwana struktura:

    <dataset_root>/
        1/
            left/   *.bmp
            right/  *.bmp
        2/
            left/
            right/
        ...

Każdy obraz ma "klasę" jednoznacznie identyfikowaną parą
    (subject_id, eye_side)
ponieważ tęczówki lewego i prawego oka tej samej osoby nie są identyczne.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import numpy as np
from PIL import Image


@dataclass
class MMUImage:
    subject_id: str
    eye: str          # "left" lub "right"
    path: Path
    image: np.ndarray  # uint8 (H, W) - skala szarości

    @property
    def class_label(self) -> str:
        return f"{self.subject_id}_{self.eye}"


def list_mmu_subjects(root: Path) -> list[str]:
    """Lista identyfikatorów osób (np. ["1", "2", ...])."""
    return sorted(
        [p.name for p in Path(root).iterdir() if p.is_dir()],
        key=lambda s: int(s) if s.isdigit() else s,
    )


def _load_gray(path: Path) -> np.ndarray:
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    return np.asarray(img, dtype=np.uint8)


def iter_mmu_dataset(root: str | Path,
                     eyes: tuple[str, ...] = ("left", "right"),
                     limit_per_class: int | None = None,
                     limit_subjects: int | None = None
                     ) -> Iterator[MMUImage]:
    """
    Iteruje po obrazach datasetu w deterministycznej kolejności.

    Argumenty:
        root: katalog główny datasetu (np. ".../MMU-Iris-Database").
        eyes: które oczy uwzględnić.
        limit_per_class: ograniczenie liczby próbek na klasę.
        limit_subjects: ograniczenie liczby osób (przyspiesza ewaluację).
    """
    root = Path(root)
    subjects = list_mmu_subjects(root)
    if limit_subjects is not None:
        subjects = subjects[:limit_subjects]

    for subject in subjects:
        for eye in eyes:
            eye_dir = root / subject / eye
            if not eye_dir.is_dir():
                continue
            files = sorted(p for p in eye_dir.iterdir()
                           if p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"})
            if limit_per_class is not None:
                files = files[:limit_per_class]
            for f in files:
                try:
                    img = _load_gray(f)
                except Exception:
                    continue
                yield MMUImage(subject_id=subject, eye=eye,
                               path=f, image=img)
