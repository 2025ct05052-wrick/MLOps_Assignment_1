# Testing and reproducibility

The test suite covers the raw-file schema, the binary label transformation,
missing-value handling, inference through the fitted preprocessing pipeline,
the API request and response contract, input validation, and metrics export.

Run the quality checks from the repository root:

    ruff check src app scripts tests
    python -m pytest

The CI workflow installs the same requirements file, runs Ruff and pytest,
executes a fast model-training run, and uploads the resulting model, metrics,
plots, and MLflow tracking directory. A failing lint, test, or training step
fails the workflow.

For a full training run, omit --fast:

    python scripts/train.py --data-path data/raw/processed.cleveland.data --output-dir artifacts

The saved estimator is a single joblib pipeline, so the API applies the same
imputation, scaling, and categorical encoding used during model fitting.

