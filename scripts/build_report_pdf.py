"""Build a ten-page PDF report from the measured EDA and model outputs."""

import json
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "MLOps_Assignment_Report.pdf"
METRICS = json.loads((ROOT / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
EDA = json.loads((ROOT / "artifacts" / "eda" / "eda_summary.json").read_text(encoding="utf-8"))
VERIFICATION = json.loads((ROOT / "reports" / "verification_status.json").read_text(encoding="utf-8"))

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2F6690")
PALE = colors.HexColor("#EAF1F7")
GRAY = colors.HexColor("#53616F")
INK = colors.HexColor("#263238")
LIGHT = colors.HexColor("#F3F6F9")
GRID = colors.HexColor("#D9D9D9")
WIDTH = 7.0 * inch

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="ReportKicker", parent=styles["Normal"], fontName="Helvetica-Bold",
    fontSize=8, leading=10, textColor=BLUE, spaceAfter=5, uppercase=True,
))
styles.add(ParagraphStyle(
    name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=25, leading=29, textColor=colors.black, alignment=TA_LEFT,
    spaceBefore=22, spaceAfter=10,
))
styles.add(ParagraphStyle(
    name="ReportSubtitle", parent=styles["Normal"], fontName="Helvetica",
    fontSize=14, leading=19, textColor=GRAY, spaceAfter=14,
))
styles.add(ParagraphStyle(
    name="ReportH1", parent=styles["Heading1"], fontName="Helvetica-Bold",
    fontSize=16, leading=19, textColor=colors.black, spaceBefore=3, spaceAfter=8,
))
styles.add(ParagraphStyle(
    name="ReportH2", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=11.5, leading=14, textColor=colors.black, spaceBefore=5, spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="ReportBody", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9.25, leading=13, textColor=INK, spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="ReportSmall", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=8.1, leading=10.2, textColor=GRAY, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="ReportCode", parent=styles["Code"], fontName="Courier",
    fontSize=8, leading=11, textColor=INK, leftIndent=8, rightIndent=8,
    backColor=PALE, borderPadding=7, spaceBefore=3, spaceAfter=8,
))
styles.add(ParagraphStyle(
    name="TableBody", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=8.1, leading=10, textColor=INK, spaceAfter=0,
))
styles.add(ParagraphStyle(
    name="TableHead", parent=styles["BodyText"], fontName="Helvetica-Bold",
    fontSize=8.2, leading=10, textColor=colors.white, spaceAfter=0,
))
styles.add(ParagraphStyle(
    name="Caption", parent=styles["Normal"], fontName="Helvetica-Oblique",
    fontSize=8, leading=10, textColor=GRAY, alignment=TA_CENTER, spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="Centered", parent=styles["Normal"], fontName="Helvetica",
    fontSize=9.1, leading=12, textColor=INK, alignment=TA_CENTER, spaceAfter=5,
))


def p(text: str, style="ReportBody") -> Paragraph:
    return Paragraph(text, styles[style])


def make_table(headers, rows, widths, font_size=None):
    data = [[p(str(value), "TableHead") for value in headers]]
    data.extend([[p(str(value), "TableBody") for value in row] for row in rows])
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.45, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        *[("BACKGROUND", (0, index), (-1, index), LIGHT) for index in range(2, len(data), 2)],
    ]))
    return table


def image(path: Path, width: float, max_height: float | None = None):
    with PILImage.open(path) as source:
        source_width, source_height = source.size
    scale = width / source_width
    if max_height:
        scale = min(scale, max_height / source_height)
    return Image(str(path), width=source_width * scale, height=source_height * scale)


def add_page_title(story, kicker: str, title: str):
    story.extend([p(kicker.upper(), "ReportKicker"), p(title, "ReportH1")])


def on_page(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.setFillColor(GRAY)
    canvas.drawString(document.leftMargin, letter[1] - 0.36 * inch,
                      "MLOPS EXPERIMENTAL LEARNING  |  HEART DISEASE CLASSIFICATION")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(letter[0] - document.rightMargin, 0.35 * inch,
                           f"MLOps Assignment 01  |  {document.page}")
    canvas.restoreState()


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=letter, rightMargin=0.76 * inch,
        leftMargin=0.76 * inch, topMargin=0.58 * inch, bottomMargin=0.56 * inch,
        title="Heart Disease Classification MLOps Assignment Report",
        author="MLOps Assignment",
    )
    story = []

    # Page 1: cover, summary, and architecture.
    story.extend([
        p("MACHINE LEARNING OPERATIONS  |  ASSIGNMENT 01  |  OCTOBER 2026", "ReportKicker"),
        p("Heart Disease Classification", "ReportTitle"),
        p("An end-to-end MLOps workflow for the UCI Cleveland dataset", "ReportSubtitle"),
        make_table(
            ["Course", "Dataset", "Records", "Target"],
            [["MLOps AIMLCZG523", "UCI Heart Disease", "303", "Diagnosis > 0"]],
            [1.6 * inch, 2.0 * inch, 0.9 * inch, 2.5 * inch],
        ),
        Spacer(1, 10),
        p("Executive summary", "ReportH2"),
        p(
            "This project builds a repeatable path from the supplied clinical data to a model-backed API. "
            "The source diagnosis is recoded as absence (0) versus presence (1). One sklearn pipeline "
            "owns missing-value handling, feature scaling, categorical encoding, and classification, so "
            "training and inference share the same transformations. Logistic Regression and Random Forest "
            "are tuned with stratified cross-validation; mean cross-validation ROC-AUC selects the model. "
            "The trained artifact is served by FastAPI, while MLflow records experiments and Prometheus "
            "metrics feed a provisioned Grafana dashboard."
        ),
        p(
            "This report records measured model results and local runtime checks. Docker Compose served "
            "health and prediction requests, Prometheus reported the API target as UP, and Grafana received "
            "metrics. Docker Desktop Kind ran two API replicas, with health, prediction, and metrics "
            "verified through kubectl port-forward. No public cloud endpoint is used.",
            "ReportSmall",
        ),
        image(ROOT / "reports" / "figures" / "architecture.png", WIDTH, 1.8 * inch),
        p("Figure 1. Data, training, serving, and monitoring architecture.", "Caption"),
        PageBreak(),
    ])

    # Page 2: data acquisition.
    add_page_title(story, "01  |  Data acquisition", "Dataset and data contract")
    story.append(p(
        "The project uses the processed Cleveland subset in the UCI Heart Disease archive. The legacy "
        "file has 14 comma-separated fields without a header: 13 predictors and one diagnosis value. "
        "The supplied archive is retained at the workspace root; data/raw/processed.cleveland.data is "
        "the focused input used by the scripts. The download helper fetches the UCI ZIP and extracts "
        "only this named member, printing SHA-256 hashes for provenance."
    ))
    story.append(p(
        "Acquisition was run from the UCI endpoint. The downloaded archive SHA-256 matched the supplied "
        "archive (b17cd273da9ce1caa4710fce80227ea454d4dbf9fcbc8e6a9121672751563adc); the extracted "
        "Cleveland file hash is a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8.",
        "ReportSmall",
    ))
    missing_text = ", ".join(
        f"{name}: {count}" for name, count in EDA["source"]["missing_values_by_column"].items() if count
    )
    story.append(make_table(
        ["Data check", "Observed", "Handling"],
        [
            ["Records", EDA["source"]["raw_rows"], "Keep usable diagnosis rows"],
            ["Predictors", "13 clinical fields", "Preserve source order and names"],
            ["Unknown values", EDA["source"]["total_missing_values"], "Keep missing until model fit"],
            ["Diagnosis", "Ordinal values 0-4", "Binary label: 0 vs 1-4"],
            ["Duplicates", EDA["source"]["raw_duplicate_rows"], "Remove exact duplicate rows"],
        ],
        [1.35 * inch, 1.7 * inch, 3.95 * inch],
    ))
    story.extend([
        Spacer(1, 7),
        p("Target definition", "ReportH2"),
        p(
            "Diagnosis 0 is absence, while values 1, 2, 3, or 4 represent presence. This explicit "
            "transformation is implemented in src/data.py and tested. Predictors are age, sex, chest-pain "
            "type, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG, maximum "
            "heart rate, exercise angina, ST depression, slope, major vessels, and thalassemia category."
        ),
        p(
            f"Six feature values are missing in the source ({missing_text}). These cells remain null "
            "through data splitting. Imputers are fitted inside each training fold so validation and "
            "holdout statistics do not leak into preprocessing."
        ),
        PageBreak(),
    ])

    # Page 3: distributions and class balance.
    add_page_title(story, "02  |  Exploratory analysis", "Feature distributions and class balance")
    story.append(p(
        "The first EDA view summarizes all 13 predictors after target mapping. Continuous measurements "
        "show patient variability, while coded fields such as chest-pain type and thalassemia type are "
        "discrete categories even though they are stored as numbers. This distinction informs the model "
        "pipeline: measurements are scaled; clinical codes are one-hot encoded."
    ))
    story.append(image(ROOT / "artifacts" / "eda" / "feature_distributions.png", 5.55 * inch, 5.9 * inch))
    story.append(p("Figure 2. Histograms for Cleveland clinical features.", "Caption"))
    counts = EDA["clean_data"]["target_distribution"]
    total = sum(counts.values())
    story.append(make_table(
        ["Target", "Meaning", "Patients", "Share"],
        [
            ["0", "No diagnosis", counts.get("0", 0), f"{counts.get('0', 0) / total:.1%}"],
            ["1", "Diagnosis present", counts.get("1", 0), f"{counts.get('1', 0) / total:.1%}"],
        ],
        [0.7 * inch, 2.25 * inch, 1.1 * inch, 1.1 * inch],
    ))
    story.append(Spacer(1, 5))
    story.append(p(
        "The target is moderately imbalanced. Accuracy alone would be incomplete, so precision, recall, "
        "and ROC-AUC are reported. Both candidate classifiers use class weighting."
    ))
    story.append(PageBreak())

    # Page 4: association and missingness figures.
    add_page_title(story, "02  |  Exploratory analysis", "Associations and source missingness")
    story.append(p(
        "The correlation heatmap is descriptive: it shows linear associations among numeric encodings "
        "and the binary target. Ordinal category codes can create correlations that do not represent "
        "true distances between categories, so the figure orients review rather than driving feature selection."
    ))
    story.append(image(ROOT / "artifacts" / "eda" / "correlation_heatmap.png", 5.45 * inch, 4.8 * inch))
    story.append(p("Figure 3. Pearson correlation matrix, including the binary target.", "Caption"))
    side = Table(
        [[
            image(ROOT / "artifacts" / "eda" / "class_balance.png", 3.05 * inch, 1.75 * inch),
            image(ROOT / "artifacts" / "eda" / "missingness.png", 3.05 * inch, 1.75 * inch),
        ]],
        colWidths=[3.2 * inch, 3.2 * inch],
    )
    side.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.extend([
        side,
        p("Figure 4. Class balance (left) and raw missingness (right).", "Caption"),
        p(
            "The missingness chart uses the raw source, so model-time imputation cannot conceal data "
            "quality. Plots and summary counts are regenerated by scripts/eda.py."
        ),
        PageBreak(),
    ])

    # Page 5: preprocessing and candidate models.
    add_page_title(story, "03  |  Feature engineering", "Leakage-safe preprocessing and model selection")
    story.extend([
        p(
            "A ColumnTransformer routes five measurements (age, resting blood pressure, serum cholesterol, "
            "maximum heart rate, and ST depression) through median imputation and StandardScaler. Eight "
            "coded fields pass through most-frequent imputation and one-hot encoding. Unknown categories "
            "at inference are ignored safely. All transformations are inside the classifier pipeline, so "
            "each cross-validation fold learns its own imputation and scaling statistics."
        ),
        make_table(
            ["Candidate", "Role", "Search dimensions"],
            [
                ["Logistic Regression", "Interpretable linear baseline", "C; balanced class weights"],
                ["Random Forest", "Nonlinear tree ensemble", "Tree count, depth, leaf size"],
            ],
            [1.55 * inch, 2.15 * inch, 3.0 * inch],
        ),
        Spacer(1, 8),
        p("Evaluation protocol", "ReportH2"),
        p(
            "A stratified 80/20 split uses a fixed random seed. GridSearchCV tunes each candidate on the "
            "training portion with five stratified folds (three in the CI smoke run). ROC-AUC is the refit "
            "metric. Accuracy, precision, recall, and ROC-AUC means and standard deviations are retained. "
            "The untouched holdout reports all four metrics once. The selected pipeline is refit on all "
            "303 rows for API use."
        ),
        p(
            "The holdout is 61 records and the training portion is 242. These estimates come from a small "
            "historical sample and are not a stable clinical performance guarantee.",
            "ReportSmall",
        ),
        PageBreak(),
    ])

    # Page 6: actual results.
    add_page_title(story, "04  |  Model results", "Cross-validation and holdout performance")
    cv_rows = []
    holdout_rows = []
    for item in METRICS["models"]:
        cv = item["cv_metrics"]
        holdout = item["holdout_metrics"]
        cv_rows.append([
            item["model_name"].replace("_", " ").title(),
            f"{cv['cv_accuracy_mean']:.3f} +/- {cv['cv_accuracy_std']:.3f}",
            f"{cv['cv_precision_mean']:.3f} +/- {cv['cv_precision_std']:.3f}",
            f"{cv['cv_recall_mean']:.3f} +/- {cv['cv_recall_std']:.3f}",
            f"{cv['cv_roc_auc_mean']:.3f} +/- {cv['cv_roc_auc_std']:.3f}",
        ])
        holdout_rows.append([
            item["model_name"].replace("_", " ").title(),
            *(f"{holdout[key]:.3f}" for key in ["accuracy", "precision", "recall", "roc_auc"]),
        ])
    story.extend([
        p("Table 1. Training-set cross-validation scores: mean +/- standard deviation.", "ReportSmall"),
        make_table(
            ["Model", "Accuracy", "Precision", "Recall", "ROC-AUC"],
            cv_rows,
            [1.45 * inch, 1.25 * inch, 1.25 * inch, 1.1 * inch, 1.3 * inch],
        ),
        Spacer(1, 10),
        p("Table 2. Held-out 20% test results.", "ReportSmall"),
        make_table(
            ["Model", "Accuracy", "Precision", "Recall", "ROC-AUC"],
            holdout_rows,
            [1.45 * inch, 1.25 * inch, 1.25 * inch, 1.1 * inch, 1.3 * inch],
        ),
        Spacer(1, 11),
        p("Selection", "ReportH2"),
    ])
    selected = METRICS["selected_model"]
    winner = next(item for item in METRICS["models"] if item["model_name"] == selected)
    other = next(item for item in METRICS["models"] if item["model_name"] != selected)
    margin = winner["cv_metrics"]["cv_roc_auc_mean"] - other["cv_metrics"]["cv_roc_auc_mean"]
    story.append(p(
        f"{selected.replace('_', ' ').title()} is selected because it achieved the highest mean "
        f"cross-validation ROC-AUC ({winner['cv_metrics']['cv_roc_auc_mean']:.3f}). The margin over the "
        f"other candidate is {margin:.4f}, small relative to fold variation, so there is no decisive "
        "separation. The ranking rule was fixed before holdout comparison; the holdout did not select "
        "hyperparameters."
    ))
    story.extend([
        p(
            "These are experimental estimates on a small historical sample. They are not medical thresholds "
            "and should not guide patient care.",
            "ReportSmall",
        ),
        PageBreak(),
    ])

    # Page 7: tracking and packages.
    add_page_title(story, "05  |  Experiment tracking", "MLflow and reproducible artifacts")
    story.extend([
        p(
            "Each candidate run records the model name, selected hyperparameters, cross-validation metrics, "
            "holdout metrics, fitted pipeline, and holdout ROC curve. The default tracking URI is a local "
            "file store under mlruns/. Run mlflow ui from the repository root to inspect the experiment. "
            "metrics.json and model_comparison.csv make results readable without opening the UI. Model "
            "signatures use numeric fields that can represent missing values."
        ),
        make_table(
            ["Artifact", "Contents", "Use"],
            [
                ["heart_disease_model.joblib", "Preprocessor and selected estimator", "API inference"],
                ["model_metadata.json", "Family, parameters, features, time", "Model identity"],
                ["metrics.json", "Data snapshot, selection, CV and holdout", "Reporting and audit"],
                ["model_comparison.csv", "One row per candidate", "Quick comparison"],
                ["*_roc.png", "Holdout ROC curve per model", "Visual assessment"],
                ["MLflow runs", "Parameters, metrics, model and plots", "Experiment history"],
            ],
            [2.0 * inch, 2.75 * inch, 1.65 * inch],
        ),
        Spacer(1, 9),
        p("Reproduction controls", "ReportH2"),
        p(
            "The shared schema parser keeps acquisition, EDA, and training aligned. Fixed random seeds "
            "control splitting, folds, and estimators. The serialized pipeline preserves fitted "
            "preprocessing. requirements.txt and the CI workflow define the environment. The tracking "
            "store is local; a shared server and object store would be needed for team-wide tracking."
        ),
        PageBreak(),
    ])

    # Page 8: API.
    add_page_title(story, "06  |  Model serving", "FastAPI request and response contract")
    story.extend([
        p(
            "The API loads the packaged sklearn pipeline from MODEL_PATH and accepts one JSON object with "
            "the 13 named predictors. Pydantic validates required fields and plausible ranges. ca and "
            "thal may be null; the trained pipeline imputes them. The response returns a 0/1 prediction, "
            "a class label, and the estimated positive-class probability. This probability has not been "
            "calibrated or validated for clinical use."
        ),
        p("Example request with two optional values omitted:", "ReportH2"),
        p(
            '{<br/>  "age": 63, "sex": 1, "cp": 1, "trestbps": 145,<br/>'
            '  "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,<br/>'
            '  "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": null, "thal": null<br/>}',
            "ReportCode",
        ),
        p("POST /predict   |   GET /health   |   GET /metrics", "ReportKicker"),
        make_table(
            ["Endpoint", "Purpose", "Expected behavior"],
            [
                ["/predict", "One prediction", "Prediction, label, confidence, version"],
                ["/health", "Readiness", "503 when model artifact is unavailable"],
                ["/metrics", "Monitoring", "Prometheus request, latency, class counters"],
            ],
            [1.15 * inch, 1.55 * inch, 3.7 * inch],
        ),
        Spacer(1, 8),
        p("Privacy and logging", "ReportH2"),
        p(
            "Application logs capture method, route, status, and elapsed time, but omit request bodies. "
            "The local Python process and Docker Compose smoke checks returned /health status ok and valid "
            "/predict responses for samples/predict.json. The Docker container also returned HTTP 200 from "
            "/metrics. Kubernetes health, prediction, and metrics endpoints were verified separately through "
            "kubectl port-forward."
        ),
        PageBreak(),
    ])

    # Page 9: CI and verification.
    add_page_title(story, "07  |  CI and quality", "Automated checks and local verification")
    story.extend([
        p(
            "GitHub Actions installs the requirements file, runs Ruff, executes pytest, trains both models "
            "in a smaller CI search mode, and uploads the model, metrics, plots, and MLflow store. A lint, "
            "test, or training failure fails the workflow."
        ),
        make_table(
            ["Check", "Coverage", "Observed status"],
            [
                ["Ruff", "Imports and common correctness checks", VERIFICATION["ruff"]],
                ["Pytest", "Data, recoding, pipeline, API contract", VERIFICATION["pytest"]],
                ["Full training", "Five-fold search, metrics and model", VERIFICATION["training"]],
                ["API smoke", "Health and sample prediction", VERIFICATION["api"]],
                ["Remote workflow", "Workflow artifact retained per run", VERIFICATION.get("github_ci", "Not run")],
            ],
            [1.35 * inch, 2.6 * inch, 2.5 * inch],
        ),
        Spacer(1, 8),
        p("Reproduction commands", "ReportH2"),
        p("ruff check src app scripts tests", "ReportCode"),
        p("python -m pytest", "ReportCode"),
        p("python scripts/train.py --fast", "ReportCode"),
        p(
            f"Current GitHub Actions status: {VERIFICATION.get('github_ci', 'Not run')}. "
            "The recorded passing workflow run is documented in "
            "docs/screenshots/github-actions-ci-passed.png.",
            "ReportSmall",
        ),
        PageBreak(),
    ])

    # Page 10: deployment and references.
    add_page_title(story, "08  |  Deployment and operations", "Container, Kubernetes, and monitoring")
    story.extend([
        p(
            "The Docker image packages the API and serialized model. Docker Compose starts the API with "
            "Prometheus and Grafana. The API exports request totals, latency histograms, and prediction "
            "counts. A provisioned dashboard shows route request rate, p95 latency, prediction counts, "
            "and 5xx ratio. Kubernetes manifests specify two replicas, health probes, resource bounds, "
            "scrape annotations, and a LoadBalancer Service."
        ),
        make_table(
            ["Evidence item", "Prepared configuration", "Remaining runtime proof"],
            [
                ["API container", "Dockerfile; health, prediction, and metrics requests", "Passed locally"],
                ["Observability", "Prometheus target UP; Grafana dashboard", "Passed locally"],
                ["Kubernetes", "Two ready replicas; endpoints via port-forward", "Passed locally"],
                ["Public deployment", "No cloud-specific resource is assumed", "Not performed; no public URL"],
                ["Video and screenshots", "Four evidence screenshots in docs/screenshots", "Short demo video remains"],
            ],
            [1.25 * inch, 2.55 * inch, 2.65 * inch],
        ),
        Spacer(1, 7),
        p("Conclusion and limitations", "ReportH2"),
        p(
            "Docker Compose and the local Docker Desktop Kind cluster have both been exercised. The "
            "Kubernetes Deployment reached two ready replicas, and /health, /predict, and /metrics were "
            "verified through port-forwarding. Prometheus, Grafana, Kubernetes, and GitHub Actions "
            "screenshots are included in the repository. "
            f"The current GitHub Actions result is {VERIFICATION.get('github_ci', 'not yet run')}. No public "
            "cloud endpoint is claimed; a short demo video and the course submission remain for the student. "
            "The dataset has 303 historical records and is not suitable for diagnosis or health decisions."
        ),
        p("References", "ReportH2"),
        p(
            "UCI Machine Learning Repository. Heart Disease dataset, DOI 10.24432/C52P4X. "
            '<link href="https://archive.ics.uci.edu/dataset/45/heart+disease" color="#2F6690">'
            "archive.ics.uci.edu/dataset/45/heart+disease</link>",
            "ReportSmall",
        ),
        p(
            "MLflow. Tracking API documentation. "
            '<link href="https://mlflow.org/docs/latest/tracking/tracking-api/" color="#2F6690">'
            "mlflow.org/docs/latest/tracking/tracking-api</link>",
            "ReportSmall",
        ),
        p(
            "scikit-learn. ColumnTransformer documentation. "
            '<link href="https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html" '
            'color="#2F6690">scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html</link>',
            "ReportSmall",
        ),
    ])

    document.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return OUTPUT


if __name__ == "__main__":
    print(build())

