"""Shared constants for data preparation, training, and serving."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 2026
TARGET_COLUMN = "target"
RAW_COLUMNS = (
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "diagnosis",
)
FEATURE_COLUMNS = RAW_COLUMNS[:-1]
NUMERIC_FEATURES = ("age", "trestbps", "chol", "thalach", "oldpeak")
CATEGORICAL_FEATURES = ("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal")
DEFAULT_RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_TRACKING_URI = (PROJECT_ROOT / "mlruns").resolve().as_uri()

