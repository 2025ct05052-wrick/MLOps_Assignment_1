# Heart Disease MLOps Project

Repository: `MLOps_Assignment_1`

An end-to-end, reproducible binary classification workflow using the processed
Cleveland subset of the UCI Heart Disease dataset. The project covers data
acquisition, exploratory analysis, two-model comparison, MLflow experiment
tracking, a FastAPI scoring service, container and Kubernetes configurations,
CI, and basic production metrics.

## Quick start

Requires Python 3.12. The supplied data file is already under data/raw. To
download it again from UCI, run python scripts/download_data.py.

    python -m venv .venv
    .venv/Scripts/Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt

Generate EDA figures, then train both candidate models:

    python scripts/eda.py
    python scripts/train.py

Training compares Logistic Regression and Random Forest through stratified
cross-validation. ROC-AUC chooses the selected candidate; the holdout metrics
are reported separately. The complete preprocessing and estimator pipeline is
saved to artifacts/heart_disease_model.joblib. MLflow parameters, metrics,
models, and ROC plots are stored locally under mlruns/.

Run the API in a local terminal:

    uvicorn app.main:app --reload

The API serves /health, /predict, and Prometheus metrics at /metrics. Use
Docker Compose to bring up the API, Prometheus, and Grafana together; see
docs/deployment.md.

## Project map

| Path | Purpose |
| --- | --- |
| data/raw | Supplied headerless Cleveland source data |
| scripts/download_data.py | Reproducible UCI ZIP acquisition |
| scripts/eda.py | EDA charts and JSON summary |
| src/data.py | Shared source parsing and target derivation |
| src/modeling.py | Preprocessing and candidate classifiers |
| src/train.py | Tuning, evaluation, MLflow tracking, model packaging |
| app/main.py | FastAPI prediction and metrics endpoints |
| tests | Data, pipeline, and API contract tests |
| .github/workflows/ci.yml | Lint, test, train, and retain workflow artifacts |
| Dockerfile, docker-compose.yml | Container API and observability stack |
| k8s | Kubernetes Deployment and LoadBalancer Service |
| reports | Assignment report and EDA figures |

## Dataset and target

The raw diagnosis is coded from 0 to 4. The binary target is 0 for diagnosis
0 and 1 for diagnoses 1 through 4. The Cleveland source has 303 records. Six
feature values are missing; the training-time sklearn pipeline imputes these
inside each cross-validation fold to avoid preprocessing leakage.

## Reproduce evaluation

    python -m pytest
    ruff check src app scripts tests
    python scripts/train.py --fast

The fast flag uses a smaller search grid for CI. Omit it for the documented
model-selection run. See docs/testing.md, docs/data_and_eda.md, and
docs/deployment.md for details.

## Academic and clinical context

This repository is a course project, not a clinically validated decision
system. It uses a small historical dataset and estimates a dataset label, not
an individual's medical diagnosis. The assignment brief calls for independent
understanding; review the choices and adapt the analysis and demonstration in
your own words before submission.

