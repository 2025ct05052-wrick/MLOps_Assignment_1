from pathlib import Path

import pandas as pd
import pytest

from src.config import RAW_COLUMNS
from src.data import DataContractError, clean_heart_data, load_raw_data, split_features_target

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "processed.cleveland.data"


def test_supplied_cleveland_file_has_expected_shape_and_missing_cells():
    raw = load_raw_data(DATA_PATH)
    assert raw.shape == (303, 14)
    assert raw.columns.tolist() == list(RAW_COLUMNS)
    assert int(raw.isna().sum().sum()) == 6


def test_cleaning_maps_any_positive_diagnosis_to_disease():
    sample = pd.DataFrame(
        [
            [50, 1, 1, 120, 200, 0, 0, 160, 0, 0.0, 1, 0, 3, 0],
            [51, 0, 2, 130, 220, 0, 1, 150, 1, 1.0, 2, 1, 6, 2],
        ],
        columns=RAW_COLUMNS,
    )
    result = clean_heart_data(sample)
    assert result.target.tolist() == [0, 1]
    assert "diagnosis" not in result.columns


def test_missing_predictors_are_preserved_for_pipeline_imputation():
    raw = load_raw_data(DATA_PATH)
    features, target = split_features_target(clean_heart_data(raw))
    assert features.isna().sum().sum() == 6
    assert set(target.unique()) == {0, 1}


def test_cleaning_rejects_missing_required_schema_column():
    with pytest.raises(DataContractError, match="Missing required columns"):
        clean_heart_data(pd.DataFrame({"age": [40]}))

