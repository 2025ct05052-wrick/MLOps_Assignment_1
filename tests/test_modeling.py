import numpy as np

from src.data import clean_heart_data, load_raw_data, split_features_target
from src.modeling import build_pipeline, probability_for_positive_class


def test_pipeline_imputes_encodes_and_predicts_on_raw_feature_frame():
    frame = clean_heart_data(load_raw_data("data/raw/processed.cleveland.data"))
    features, target = split_features_target(frame)
    model = build_pipeline("logistic_regression").fit(features, target)
    prediction = model.predict(features.iloc[:8])
    probability = probability_for_positive_class(model, features.iloc[:8])
    assert prediction.shape == (8,)
    assert set(prediction).issubset({0, 1})
    assert np.all((probability >= 0) & (probability <= 1))

