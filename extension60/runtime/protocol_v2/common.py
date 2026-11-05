"""Immutable protocol identifiers and small filesystem helpers for v2."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SEEDS = (20261013, 20261014, 20261015, 20261016, 20261017)
ARMS = (
    "full", "no_dsdm", "shared_q", "no_dsdm_shared_q",
    "image_only", "late_fusion", "late_fusion_cap",
)
LABELS = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
EXTERNAL_DOI = "10.17632/g5bj76p2z3.2"
EXTERNAL_URL = "https://data.mendeley.com/datasets/g5bj76p2z3/2"
PAD_SPLIT_SHA256 = "242d081350f5eab9b98b9941d4dc738a22c1eade28657251d97fe8c820067507"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def cloud_or_local(relative_cloud: str, relative_local: str) -> Path:
    cloud = PROJECT / relative_cloud
    return cloud if cloud.exists() else PROJECT / relative_local


def split_dir() -> Path:
    return cloud_or_local(
        "protocol_v1/pad_patient_split_v1",
        "compute_checks/20261014_protocol/pad_patient_split_v1",
    )


def pad_images() -> Path:
    return cloud_or_local(
        "teacher_original/DSMAF/PAD-UFES-20/images",
        "teacher_materials/20261014/original/DSMAF/PAD-UFES-20/images",
    )


def model_code() -> Path:
    return cloud_or_local(
        "gpu_probe/code", "compute_checks/20261014_transfer/gpu_probe/code"
    )


def output_dir() -> Path:
    return PROJECT / "results" / "pad_dermatonet_v2"
