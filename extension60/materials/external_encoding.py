"""Original external metadata encoder, copied without changing its function body."""
import numpy as np
import pandas as pd
from data_v2 import AgeStats,_age_features

def encode_external_metadata(frame: pd.DataFrame, stats: AgeStats) -> np.ndarray:
    """Read exactly age and gender; diagnostic columns are never model inputs."""
    age_z, age_missing = _age_features(frame["age"], stats)
    sex = frame["gender"].fillna("").astype(str).str.strip().str.upper()
    if not sex.isin(("FEMININO", "MASCULINO", "")).all():
        unknown = sorted(set(sex) - {"FEMININO", "MASCULINO", ""})
        raise RuntimeError(f"Unmapped DermatoNet gender values: {unknown}")
    female = (sex == "FEMININO").to_numpy(dtype=np.float32)
    male = (sex == "MASCULINO").to_numpy(dtype=np.float32)
    unknown = (sex == "").to_numpy(dtype=np.float32)
    result = np.column_stack((age_z, age_missing, female, male, unknown)).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite external metadata")
    return result
