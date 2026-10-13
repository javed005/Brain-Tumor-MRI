"""Shared, identical image decoding for training and Streamlit inference."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DATASET_URL = "https://drive.google.com/drive/folders/1C9ww4JnZ2sh22I-hbt45OR16o4ljGxju?usp=sharing"


def load_image_array(source, size: int) -> np.ndarray:
    """RGB, EXIF orientation correction, bilinear resize; return RAW 0..255."""
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32).copy()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def safe_extract_zip(archive, destination):
    """Extract dataset ZIPs without allowing path traversal or symlink entries."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("The ZIP contains a path outside its destination.")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symlinks are not supported in dataset ZIPs.")
        z.extractall(destination)


def discover_dataset_root(path):
    """Find the assigned train/valid/test dataset, even inside a ZIP wrapper."""
    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError(f"Dataset directory does not exist: {path}")
    candidates = [path] + [p for p in path.rglob("*") if p.is_dir()]
    for candidate in candidates:
        children = {p.name.lower(): p for p in candidate.iterdir() if p.is_dir()}
        if "train" in children and "test" in children and any(
            x in children for x in ("valid", "validation", "val")
        ):
            return candidate
    raise ValueError(
        "Could not find train/, valid/ (or validation/), and test/ folders. "
        "Download the complete assigned dataset folder and extract its ZIP."
    )
