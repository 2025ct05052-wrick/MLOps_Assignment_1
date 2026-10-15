"""Dataset loading, validation, and deterministic cleaning."""

from pathlib import Path

import pandas as pd

from src.config import FEATURE_COLUMNS, RAW_COLUMNS, TARGET_COLUMN


class DataContractError(ValueError):
    """Raised when the source dataset does not match its expected schema."""


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the headerless Cleveland data while preserving missing cells."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Raw data file was not found: {source}")
    frame = pd.read_csv(
        source, header=None, names=RAW_COLUMNS, na_values=["?"], skipinitialspace=True
    )
    if frame.empty:
        raise DataContractError("The raw dataset is empty.")
    if frame.shape[1] != len(RAW_COLUMNS):
        raise DataContractError(f"Expected {len(RAW_COLUMNS)} columns, got {frame.shape[1]}.")
    return frame


def clean_heart_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a clean analysis frame and map diagnosis 0-4 to binary disease status."""
    missing = set(RAW_COLUMNS).difference(frame.columns)
    if missing:
        raise DataContractError(f"Missing required columns: {sorted(missing)}")
    cleaned = frame.loc[:, RAW_COLUMNS].copy()
    for column in RAW_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna(subset=["diagnosis"])
    if cleaned.empty:
        raise DataContractError("No rows have a usable diagnosis label.")
    cleaned[TARGET_COLUMN] = (cleaned["diagnosis"] > 0).astype("int8")
    cleaned = cleaned.drop(columns=["diagnosis"]).drop_duplicates().reset_index(drop=True)
    return cleaned.loc[:, [*FEATURE_COLUMNS, TARGET_COLUMN]]


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the 13 validated predictors and binary target."""
    required = set((*FEATURE_COLUMNS, TARGET_COLUMN))
    missing = required.difference(frame.columns)
    if missing:
        raise DataContractError(f"Cannot split data; missing columns: {sorted(missing)}")
    features = frame.loc[:, FEATURE_COLUMNS].copy()
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="raise").astype(int)
    invalid = set(target.unique()).difference({0, 1})
    if invalid:
        raise DataContractError(f"Target must be binary; found: {sorted(invalid)}")
    return features, target


def save_clean_dataset(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write the cleaned dataset as a headered CSV."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination

