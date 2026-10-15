#!/usr/bin/env python3
"""Produce repeatable EDA figures and a machine-readable data summary.

The script delegates parsing and cleaning to ``src.data`` so that the same
dataset contract is used during exploratory analysis, training, and serving.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import clean_heart_data, load_raw_data  # noqa: E402

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "eda"
TARGET_COLUMN = "target"
FIGURE_NAMES = {
    "distributions": "feature_distributions.png",
    "correlation": "correlation_heatmap.png",
    "class_balance": "class_balance.png",
    "missingness": "missingness.png",
}


def parse_args() -> argparse.Namespace:
    """Parse EDA command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Create EDA figures and a JSON summary for the heart-disease data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the headerless processed Cleveland source file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for figures and eda_summary.json.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Raster resolution for generated figures.",
    )
    return parser.parse_args()


def setup_plotting() -> None:
    """Configure a restrained, print-friendly visual style."""

    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfe",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#243447",
            "xtick.color": "#334e68",
            "ytick.color": "#334e68",
            "savefig.facecolor": "white",
        }
    )


def as_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric view, treating UCI's ``?`` missing marker as null."""

    normalized = frame.replace("?", np.nan)
    return normalized.apply(pd.to_numeric, errors="coerce")


def validate_clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the documented cleaned-data contract used by this script."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("clean_heart_data must return a pandas DataFrame.")
    if TARGET_COLUMN not in frame.columns:
        raise ValueError("clean_heart_data output must include a 'target' column.")
    if frame.empty:
        raise ValueError("clean_heart_data returned no rows.")

    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if target.isna().any():
        raise ValueError("The cleaned target contains missing or non-numeric values.")
    observed_targets = set(target.astype(int).unique())
    if not observed_targets.issubset({0, 1}):
        raise ValueError(
            "The cleaned target must be binary {0, 1}; "
            f"observed values were {sorted(observed_targets)}."
        )
    return frame.copy()


def save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    """Save a figure consistently and release its resources."""

    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_feature_distributions(frame: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """Create a grid of feature histograms from the cleaned data frame."""

    feature_columns = [column for column in frame.columns if column != TARGET_COLUMN]
    numeric_features = as_numeric(frame[feature_columns]).dropna(axis=1, how="all")
    if numeric_features.empty:
        raise ValueError("No numeric feature columns are available for distribution plots.")

    columns_per_row = 3
    row_count = int(np.ceil(len(numeric_features.columns) / columns_per_row))
    figure, axes = plt.subplots(
        row_count,
        columns_per_row,
        figsize=(15, max(3.2 * row_count, 4.5)),
        constrained_layout=True,
    )
    for axis, column in zip(np.ravel(axes), numeric_features.columns, strict=False):
        values = numeric_features[column].dropna()
        unique_count = values.nunique()
        bins: int | str = "auto" if unique_count > 10 else max(int(unique_count), 1)
        sns.histplot(values, bins=bins, color="#2a6f97", edgecolor="white", ax=axis)
        axis.set_title(column.replace("_", " ").title())
        axis.set_xlabel("")
        axis.set_ylabel("Patients")
    for axis in np.ravel(axes)[len(numeric_features.columns) :]:
        axis.set_visible(False)
    figure.suptitle("Feature distributions in cleaned Cleveland data", fontsize=16, fontweight="bold")
    save_figure(figure, output_path, dpi)


def plot_correlation_heatmap(frame: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """Create a correlation heatmap for numeric predictors and the target."""

    numeric_frame = as_numeric(frame).dropna(axis=1, how="all")
    correlation = numeric_frame.corr(numeric_only=True)
    if correlation.empty:
        raise ValueError("No numeric columns are available for a correlation heatmap.")

    figure, axis = plt.subplots(figsize=(13, 10), constrained_layout=True)
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Pearson correlation", "shrink": 0.85},
        ax=axis,
    )
    axis.set_title("Correlation structure of predictors and heart-disease target", pad=14)
    axis.set_xlabel("")
    axis.set_ylabel("")
    save_figure(figure, output_path, dpi)


def plot_class_balance(frame: pd.DataFrame, output_path: Path, dpi: int) -> dict[str, int]:
    """Plot and return the binary target distribution."""

    counts = (
        pd.to_numeric(frame[TARGET_COLUMN], errors="raise")
        .astype(int)
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )
    labels = ["No heart disease (0)", "Heart disease (1)"]
    colors = ["#4c78a8", "#e76f51"]
    figure, axis = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    bars = axis.bar(labels, counts.values, color=colors, width=0.62)
    total = int(counts.sum())
    for bar, count in zip(bars, counts.values, strict=True):
        percent = (count / total * 100) if total else 0
        axis.annotate(
            f"{count} ({percent:.1f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
            fontweight="bold",
        )
    axis.set_title("Binary target balance after diagnosis recoding")
    axis.set_ylabel("Patients")
    axis.set_xlabel("")
    axis.spines[["top", "right"]].set_visible(False)
    save_figure(figure, output_path, dpi)
    return {str(label): int(count) for label, count in counts.items()}


def plot_missingness(raw_frame: pd.DataFrame, output_path: Path, dpi: int) -> dict[str, int]:
    """Visualize source-data missingness before cleaning or imputation."""

    missing_counts = raw_frame.replace("?", np.nan).isna().sum().sort_values(ascending=False)
    ordered_counts = missing_counts[missing_counts > 0]
    figure, axis = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    if ordered_counts.empty:
        axis.text(
            0.5,
            0.5,
            "No missing values detected in the raw source.",
            ha="center",
            va="center",
            fontsize=13,
        )
        axis.set_axis_off()
    else:
        colors = ["#d1495b" if count else "#b8c2cc" for count in ordered_counts.values]
        bars = axis.barh(ordered_counts.index, ordered_counts.values, color=colors)
        for bar, count in zip(bars, ordered_counts.values, strict=True):
            axis.annotate(
                str(int(count)),
                (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                ha="left",
                va="center",
                xytext=(5, 0),
                textcoords="offset points",
            )
        axis.invert_yaxis()
        axis.set_xlabel("Missing source values")
        axis.set_ylabel("Feature")
        axis.spines[["top", "right"]].set_visible(False)
    axis.set_title("Missingness in raw UCI Cleveland data")
    save_figure(figure, output_path, dpi)
    return {column: int(count) for column, count in missing_counts.items()}


def serializable_descriptives(frame: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    """Return compact numeric descriptive statistics suitable for JSON output."""

    result: dict[str, dict[str, float | int | None]] = {}
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        result[column] = {
            "count": int(values.count()),
            "mean": _optional_float(values.mean()),
            "std": _optional_float(values.std()),
            "min": _optional_float(values.min()),
            "median": _optional_float(values.median()),
            "max": _optional_float(values.max()),
        }
    return result


def _optional_float(value: Any) -> float | None:
    """Convert NumPy/pandas numeric values to JSON-safe floats."""

    if pd.isna(value):
        return None
    return float(value)


def build_summary(
    raw_frame: pd.DataFrame,
    clean_frame: pd.DataFrame,
    data_path: Path,
    class_counts: dict[str, int],
    raw_missingness: dict[str, int],
    output_dir: Path,
) -> dict[str, Any]:
    """Build the JSON-friendly EDA summary object."""

    numeric_clean = as_numeric(clean_frame)
    target_correlation = (
        numeric_clean.corr(numeric_only=True)[TARGET_COLUMN]
        .drop(labels=[TARGET_COLUMN], errors="ignore")
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )
    total = sum(class_counts.values())
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "path": str(data_path.resolve()),
            "raw_rows": int(len(raw_frame)),
            "raw_columns": list(raw_frame.columns),
            "raw_duplicate_rows": int(raw_frame.duplicated().sum()),
            "missing_values_by_column": raw_missingness,
            "total_missing_values": int(sum(raw_missingness.values())),
        },
        "clean_data": {
            "rows": int(len(clean_frame)),
            "columns": list(clean_frame.columns),
            "duplicate_rows": int(clean_frame.duplicated().sum()),
            "target_distribution": class_counts,
            "heart_disease_prevalence": _optional_float(
                class_counts.get("1", 0) / total if total else np.nan
            ),
            "numeric_descriptives": serializable_descriptives(clean_frame),
            "target_correlations": {
                column: _optional_float(value) for column, value in target_correlation.items()
            },
        },
        "artifacts": {
            name: str((output_dir / filename).resolve()) for name, filename in FIGURE_NAMES.items()
        },
    }


def main() -> int:
    """Load data through the project contract and render all EDA outputs."""

    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    data_path = args.data_path.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Raw data file was not found at {data_path}. "
            "Run `python scripts/download_data.py` first or pass --data-path."
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_plotting()

    raw_frame = load_raw_data(data_path)
    if not isinstance(raw_frame, pd.DataFrame):
        raise TypeError("load_raw_data must return a pandas DataFrame.")
    clean_frame = validate_clean_frame(clean_heart_data(raw_frame))

    plot_feature_distributions(clean_frame, output_dir / FIGURE_NAMES["distributions"], args.dpi)
    plot_correlation_heatmap(clean_frame, output_dir / FIGURE_NAMES["correlation"], args.dpi)
    class_counts = plot_class_balance(
        clean_frame, output_dir / FIGURE_NAMES["class_balance"], args.dpi
    )
    raw_missingness = plot_missingness(
        raw_frame, output_dir / FIGURE_NAMES["missingness"], args.dpi
    )
    summary = build_summary(
        raw_frame=raw_frame,
        clean_frame=clean_frame,
        data_path=data_path,
        class_counts=class_counts,
        raw_missingness=raw_missingness,
        output_dir=output_dir,
    )
    summary_path = output_dir / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    logging.info("EDA figures and summary written to %s", output_dir)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        logging.error("EDA failed: %s", error)
        raise SystemExit(1) from error
