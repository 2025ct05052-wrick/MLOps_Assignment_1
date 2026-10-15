# Data acquisition and exploratory analysis

## Dataset and provenance

This project uses the **processed Cleveland** subset of the [UCI Heart Disease
dataset](https://archive.ics.uci.edu/dataset/45/heart+disease). UCI describes it
as 303 patient records with 13 clinical predictors and an angiographic diagnosis.
The original diagnosis is ordinal (`0` through `4`); this project frames the
assignment as a binary risk classification task by mapping `0` to no detected
heart disease and values `1`-`4` to heart disease present.

The raw file is headerless and uses `?` for missing values. In the supplied
Cleveland source, those markers occur in `ca` and `thal`. They are deliberately
retained as missing values for the model preprocessing pipeline to handle, rather
than imputed in a notebook by hand.

The expected column order is:

```text
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak,
slope, ca, thal, target
```

The UCI dataset is available under CC BY 4.0. Cite Jánosi, Steinbrunn,
Pfisterer, and Detrano (1989), *Heart Disease*, UCI Machine Learning Repository,
DOI [10.24432/C52P4X](https://doi.org/10.24432/C52P4X), when using this dataset.

## Acquire the raw data

From the repository root, download the canonical UCI ZIP archive and extract only
the source file used by the project:

```powershell
python scripts/download_data.py
```

The command is idempotent. It writes the archive and
`processed.cleveland.data` to `data/raw/`, prints SHA-256 hashes for both, and
reuses them on later runs. Use `--force` to redownload and re-extract, or use
`--help` to select another output directory, archive URL, member, or timeout.

```powershell
python scripts/download_data.py --force --output-dir data/raw
```

No credentials are required. The URL is intentionally configurable so a course
mirror can be used in an offline or restricted network environment.

## Data contract

The shared data module is the single source of truth for parsing and cleaning:

```python
from src.data import clean_heart_data, load_raw_data

raw = load_raw_data("data/raw/processed.cleveland.data")
clean = clean_heart_data(raw)
```

`load_raw_data(path)` returns a DataFrame using the canonical column names.
`clean_heart_data(frame)` returns the same 13 predictors plus a binary `target`
derived from the original diagnosis (`target > 0`). Keeping this contract shared
between EDA and training prevents target-recoding or column-order drift.

## Generate the EDA package

Run the reproducible plotting script after acquiring the data:

```powershell
python scripts/eda.py
```

By default, the package is written to `artifacts/eda/`. A custom source path or
output location is useful in CI and experiments:

```powershell
python scripts/eda.py `
  --data-path data/raw/processed.cleveland.data `
  --output-dir artifacts/eda `
  --dpi 180
```

The output directory contains:

| File | Purpose |
| --- | --- |
| `feature_distributions.png` | Grid of distributions for all available numeric clinical predictors. |
| `correlation_heatmap.png` | Pearson correlations among numeric predictors and the binary target. |
| `class_balance.png` | Counts and proportions for absent/present heart disease after binary recoding. |
| `missingness.png` | Source-level missing-value counts before any modelling-time imputation. |
| `eda_summary.json` | Reproducible counts, missingness, duplicates, descriptive statistics, target prevalence, correlations, and artifact paths. |

The EDA script uses raw data for the missingness figure, so an imputed downstream
pipeline cannot conceal source quality issues. It uses the cleaned data for target
balance, distributions, and correlations. The summary is designed for CI logs,
reports, and experiment-tracking artifacts; figures should be interpreted as
descriptive rather than as clinical evidence.
