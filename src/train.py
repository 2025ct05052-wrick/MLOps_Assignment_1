"""Train and compare reproducible heart-disease classifiers."""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_RAW_DATA_PATH,
    DEFAULT_TRACKING_URI,
    PROJECT_ROOT,
    RANDOM_STATE,
)
from src.data import clean_heart_data, load_raw_data, save_clean_dataset, split_features_target
from src.modeling import build_pipeline, evaluate_binary_classifier, parameter_grid

LOGGER = logging.getLogger(__name__)
SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "roc_auc": "roc_auc"}
MODELS = ("logistic_regression", "random_forest")


def save_roc_plot(model, features, target, name: str, path: Path) -> Path:
    """Save a labeled holdout ROC curve for reports and experiment artifacts."""
    figure, axis = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_estimator(model, features, target, name=name.replace("_", " ").title(), ax=axis)
    axis.plot([0, 1], [0, 1], "--", color="#718096", linewidth=1)
    axis.set_title(f"Holdout ROC Curve: {name.replace('_', ' ').title()}")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _setup_mlflow(enabled: bool, tracking_uri: str, experiment_name: str):
    if not enabled:
        return None
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        return mlflow
    except ImportError:
        LOGGER.warning("MLflow is unavailable; continuing without experiment tracking.")
        return None


def _metrics_from_search(search: GridSearchCV) -> dict[str, float]:
    index = search.best_index_
    summary = {}
    for name in SCORING:
        summary[f"cv_{name}_mean"] = float(search.cv_results_[f"mean_test_{name}"][index])
        summary[f"cv_{name}_std"] = float(search.cv_results_[f"std_test_{name}"][index])
    return summary


def run_training(
    data_path: str | Path = DEFAULT_RAW_DATA_PATH,
    output_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = "heart-disease-classification",
    enable_mlflow: bool = True,
    fast: bool = False,
) -> dict:
    """Tune two candidate pipelines, assess a holdout set, and save the winner."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = load_raw_data(data_path)
    clean = clean_heart_data(raw)
    save_clean_dataset(clean, PROJECT_ROOT / "data" / "processed" / "heart_disease_clean.csv")
    features, target = split_features_target(clean)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=RANDOM_STATE
    )
    folds = 3 if fast else 5
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    mlflow = _setup_mlflow(enable_mlflow, tracking_uri, experiment_name)
    results = []

    for model_name in MODELS:
        LOGGER.info("Searching %s over %s stratified folds", model_name, folds)
        search = GridSearchCV(
            build_pipeline(model_name),
            parameter_grid(model_name, fast=fast),
            scoring=SCORING,
            refit="roc_auc",
            cv=splitter,
            # Serial execution avoids fragile worker-process startup on Windows.
            n_jobs=1,
            return_train_score=False,
        )
        search.fit(x_train, y_train)
        cv_metrics = _metrics_from_search(search)
        holdout_metrics = evaluate_binary_classifier(search.best_estimator_, x_test, y_test)
        roc_plot = save_roc_plot(
            search.best_estimator_, x_test, y_test, model_name, output / f"{model_name}_roc.png"
        )
        if mlflow is not None:
            from mlflow.models import ModelSignature
            from mlflow.types import ColSpec, Schema

            with mlflow.start_run(run_name=model_name):
                mlflow.log_params({"model_name": model_name, **search.best_params_, "cv_folds": folds})
                mlflow.log_metrics(cv_metrics)
                mlflow.log_metrics({f"holdout_{k}": v for k, v in holdout_metrics.items()})
                mlflow.log_artifact(str(roc_plot), artifact_path="plots")
                signature = ModelSignature(
                    inputs=Schema([ColSpec("double", name=name) for name in x_train.columns]),
                    outputs=Schema([ColSpec("long")]),
                )
                mlflow.sklearn.log_model(
                    search.best_estimator_,
                    artifact_path="model",
                    signature=signature,
                    input_example=x_train.head(3).astype(float),
                )
        results.append({
            "model_name": model_name,
            "best_params": search.best_params_,
            "cv_metrics": cv_metrics,
            "holdout_metrics": holdout_metrics,
            "estimator": search.best_estimator_,
        })

    winner = max(results, key=lambda row: row["cv_metrics"]["cv_roc_auc_mean"])
    final_pipeline = build_pipeline(winner["model_name"]).set_params(**winner["best_params"])
    final_pipeline.fit(features, target)
    model_path = output / "heart_disease_model.joblib"
    joblib.dump(final_pipeline, model_path)
    try:
        model_path_for_report = model_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        model_path_for_report = str(model_path)

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "source_rows": int(len(raw)),
            "clean_rows": int(len(clean)),
            "feature_count": int(features.shape[1]),
            "training_rows": int(len(x_train)),
            "holdout_rows": int(len(x_test)),
            "positive_rate": float(target.mean()),
            "missing_values_before_pipeline": {k: int(v) for k, v in features.isna().sum().items()},
        },
        "selection_metric": "cv_roc_auc_mean",
        "selected_model": winner["model_name"],
        "selected_params": winner["best_params"],
        "models": [
            {key: value for key, value in row.items() if key != "estimator"}
            for row in results
        ],
        "model_path": model_path_for_report,
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "model_metadata.json").write_text(json.dumps({
        "model_name": winner["model_name"],
        "best_params": winner["best_params"],
        "feature_names": list(features.columns),
        "training_rows": int(len(features)),
        "trained_at_utc": summary["generated_at_utc"],
    }, indent=2), encoding="utf-8")
    pd.DataFrame([
        {"model": row["model_name"], **row["cv_metrics"],
         **{f"holdout_{k}": v for k, v in row["holdout_metrics"].items()}}
        for row in results
    ]).to_csv(output / "model_comparison.csv", index=False)
    LOGGER.info("Selected %s; saved model to %s", winner["model_name"], model_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_RAW_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment-name", default="heart-disease-classification")
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Use a smaller search for CI smoke runs.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run_training(
        data_path=args.data_path, output_dir=args.output_dir, tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name, enable_mlflow=not args.disable_mlflow, fast=args.fast
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

