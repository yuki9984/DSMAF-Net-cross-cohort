"""Frozen PAD split and five-feature age/sex encoding; no external test I/O."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import LABELS, PAD_SPLIT_SHA256, pad_images, sha256, split_dir

META_COLUMNS = ("age_z", "age_missing", "sex_female", "sex_male", "sex_unknown")


@dataclass(frozen=True)
class AgeStats:
    mean: float
    std: float
    median: float

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std, "median": self.median}


def load_pad_train_valid() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    folder = split_dir()
    manifest_path = folder / "split_manifest.json"
    if not manifest_path.is_file() or sha256(manifest_path) != PAD_SPLIT_SHA256:
        raise RuntimeError("Frozen PAD split manifest is missing or changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = {}
    for role in ("train", "valid"):
        expected = manifest["roles"][role]
        path = folder / expected["file"]
        if not path.is_file() or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Frozen {role} CSV hash mismatch")
        frame = pd.read_csv(path)
        if len(frame) != expected["images"]:
            raise RuntimeError(f"Frozen {role} image count mismatch")
        if frame["img_id"].duplicated().any():
            raise RuntimeError(f"Duplicate PAD image IDs in {role}")
        found = frame.groupby("diagnostic_number")["diagnostic"].unique().to_dict()
        if set(found) != set(range(6)) or any(
            set(found[index]) != {LABELS[index]} for index in range(6)
        ):
            raise RuntimeError(f"PAD class mapping changed in {role}")
        frames[role] = frame
    train, valid = frames["train"], frames["valid"]
    for key in ("patient_id", "img_id"):
        if set(train[key]) & set(valid[key]):
            raise RuntimeError(f"PAD train/valid overlap in {key}")
    image_root = pad_images()
    if not image_root.is_dir():
        raise FileNotFoundError(f"PAD images unavailable: {image_root}")
    for role, frame in frames.items():
        missing = [str(name) for name in frame["img_id"] if not (image_root / str(name)).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing PAD {role} images: {missing[:3]}")
    return train, valid, PAD_SPLIT_SHA256


def fit_age_stats(train: pd.DataFrame) -> AgeStats:
    age = pd.to_numeric(train["age"], errors="coerce")
    valid = age[(age >= 0) & (age <= 120)]
    if valid.empty or valid.count() < len(train) * 0.8:
        raise RuntimeError("PAD train age values insufficient")
    std = float(valid.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        raise RuntimeError("PAD train age standard deviation invalid")
    return AgeStats(mean=float(valid.mean()), std=std, median=float(valid.median()))


def _age_features(values: pd.Series, stats: AgeStats) -> tuple[np.ndarray, np.ndarray]:
    age = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    invalid = ~np.isfinite(age) | (age < 0) | (age > 120)
    if np.any(np.isfinite(age) & ((age < 0) | (age > 120))):
        raise RuntimeError("Age outside the locked [0,120] range")
    age[invalid] = stats.median
    return ((age - stats.mean) / stats.std).astype(np.float32), invalid.astype(np.float32)


def encode_pad_metadata(frame: pd.DataFrame, stats: AgeStats) -> np.ndarray:
    age_z, age_missing = _age_features(frame["age"], stats)
    female = frame["gender_FEMALE"].to_numpy(dtype=np.float32)
    male = frame["gender_MALE"].to_numpy(dtype=np.float32)
    if not np.isin(female, [0, 1]).all() or not np.isin(male, [0, 1]).all():
        raise RuntimeError("Invalid PAD sex encoding")
    if np.any(female + male > 1):
        raise RuntimeError("Conflicting PAD sex encoding")
    unknown = 1 - female - male
    result = np.column_stack((age_z, age_missing, female, male, unknown)).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite PAD metadata")
    return result
