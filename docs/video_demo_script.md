# Optional live screen-recording guide

The submission video at `reports/MLOps_Assignment_Demo.mp4` is a silent 3:20 visual walkthrough assembled from the project's actual screenshots and current metrics. It is not a capture of live PowerShell execution. Use this longer guide only if you want to create a separate live, narrated recording.

Record one clear 4–5 minute demonstration using real output from this project. Keep the recording continuous, speak in your own words, and use the sample request in `samples/predict.json`; do not show real patient data or invent results.

## Prepare before recording

1. Open PowerShell in the project folder and activate the project environment if it exists:

   ```powershell
   Set-Location "C:\Users\wrick\OneDrive - wilp.bits-pilani.ac.in\MLOPS_Assignment_1"
   if (Test-Path .\.venv\Scripts\Activate.ps1) { . .\.venv\Scripts\Activate.ps1 }
   ```

2. Make sure the full model and its MLflow runs are present. If you need to regenerate them, run these commands before building the Docker image:

   ```powershell
   python scripts/download_data.py
   python scripts/eda.py
   python scripts/train.py
   ```

   Let training finish and show its final JSON output. Use the full training command, without `--fast`; `--fast` is the smaller CI smoke run. If training takes too long for the video, finish it before pressing Record, then show the actual output files and MLflow runs during the recording.

3. Start the container stack after training, so the image contains the current model:

   ```powershell
   docker compose up --build -d
   docker compose ps
   ```

   Confirm the API is healthy. Then send sample requests so Prometheus and Grafana have recent data:

   ```powershell
   $body = Get-Content .\samples\predict.json -Raw
   Invoke-RestMethod http://localhost:8000/health
   1..10 | ForEach-Object {
     Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict `
       -ContentType "application/json" -Body $body | Out-Null
   }
   ```

   Prometheus scrapes every 15 seconds, so allow at least 15 seconds after the requests before showing the dashboard.

4. In a second PowerShell window at the project root, start MLflow and leave that window open:

   ```powershell
   mlflow ui --backend-store-uri .\mlruns --host 127.0.0.1 --port 5000
   ```

5. Check the Kubernetes cluster before recording. Do not start a port-forward yet; Docker Compose is using local port 8000.

   ```powershell
   kubectl get nodes
   kubectl rollout status deployment/heart-disease-api --timeout=180s
   kubectl get pods,services -o wide
   ```

6. Open these browser pages in separate tabs so you can switch to them quickly:

   - MLflow: `http://localhost:5000`
   - Prometheus targets: `http://localhost:9090/targets`
   - Grafana: `http://localhost:3000`
   - GitHub repository Actions: `https://github.com/2025ct05052-wrick/MLOps_Assignment_1/actions`

   In Grafana, open **Dashboards → MLOps Assignment → Heart Disease API Overview**. If the login page appears, use the local demo credentials configured in `docker-compose.yml`. Turn on Do Not Disturb, enlarge terminal text, and close unrelated windows. Never show a GitHub token or other secret.

## What to record

### 0:00–0:20 — Introduce the project

**Open:** the repository README and the project tree in Explorer or your editor.

**Point to:** `data/raw`, `src`, `scripts`, `app`, `monitoring`, `k8s`, and `.github/workflows/ci.yml`.

**Say:** “This project takes the Cleveland heart-disease dataset through analysis and model training, packages the selected pipeline behind an API, and adds monitoring, Kubernetes deployment files, and automated CI.” Mention that it is an educational dataset and not a medical diagnostic tool.

### 0:20–0:55 — Show data acquisition and EDA

**Open:** the project-root PowerShell window. If not already run, enter:

```powershell
python scripts/download_data.py
python scripts/eda.py
Get-ChildItem .\artifacts\eda
```

**Highlight:** the download script’s SHA-256 output, then the generated EDA files. Open `artifacts/eda/class_balance.png` and `artifacts/eda/missingness.png`.

**Say:** “The source has 303 rows and 13 predictors. The target is converted to a binary label; six source feature values are missing and are handled by the training pipeline.” Do not imply that the historical dataset represents current patients.

### 0:55–1:40 — Show model training and MLflow

**Open:** PowerShell at the project root and run the full training command if it is not already complete:

```powershell
python scripts/train.py
```

Wait for the command to finish. Point to the selected model and metrics in the final output. Then show the saved summary:

```powershell
Get-Content .\artifacts\metrics.json
```

**Switch to:** `http://localhost:5000`. Click the `heart-disease-classification` experiment, select the Logistic Regression and Random Forest runs, and compare their cross-validation ROC-AUC values and logged artifacts. Point to the ROC plot if time allows.

**Say:** “The candidate is selected using mean cross-validation ROC-AUC, while holdout scores are reported separately. The preprocessing is part of the pipeline, so imputation and encoding are fitted within each training fold.” Avoid presenting the model score as clinical accuracy.

### 1:40–2:30 — Demonstrate the API

**Open:** the API PowerShell window. Run:

```powershell
docker compose ps
Invoke-RestMethod http://localhost:8000/health
$body = Get-Content .\samples\predict.json -Raw
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict `
  -ContentType "application/json" -Body $body
Invoke-WebRequest http://localhost:8000/metrics -UseBasicParsing |
  Select-Object StatusCode
```

**Highlight:** the healthy API, the prediction response fields (`prediction`, `label`, `confidence`, and `model_version`), and HTTP status 200 for `/metrics`.

**Say:** “The API validates the 13 input fields, returns one model prediction and confidence, and exposes operational metrics separately.” Use only the supplied sample JSON.

### 2:30–3:15 — Show monitoring

**Switch to:** Prometheus at `http://localhost:9090/targets`.

**Point to:** the `heart-disease-api` scrape target and its green **UP** state.

**Switch to:** the Grafana dashboard. Set the time range to **Last 6 hours** if needed and refresh.

**Point to:** request rate by route, p95 latency, and predictions by class. The server-error panel may show **No data** when no server errors occurred; explain that this is expected rather than generating an artificial failure.

### 3:15–4:10 — Verify Kubernetes

The Docker Compose API and Kubernetes port-forward both use local port 8000. After finishing the dashboard segment, stop Compose before forwarding:

```powershell
docker compose down
kubectl get nodes
kubectl get pods,services -o wide
kubectl rollout status deployment/heart-disease-api --timeout=180s
```

**Highlight:** the node is `Ready`, both `heart-disease-api` pods are `1/1 Running`, and the service exposes port 80.

In this same PowerShell window, start the forward and leave it running:

```powershell
kubectl port-forward service/heart-disease-api 8000:80
```

Open a second PowerShell window in the project folder and run:

```powershell
Invoke-RestMethod http://localhost:8000/health
$body = Get-Content .\samples\predict.json -Raw
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict `
  -ContentType "application/json" -Body $body
Invoke-WebRequest http://localhost:8000/metrics -UseBasicParsing |
  Select-Object StatusCode
```

**Say:** “The local Kubernetes deployment has two replicas and health probes. I am forwarding the service to localhost and verifying the same health, prediction, and metrics endpoints.”

### 4:10–4:45 — Finish with CI and deployment configuration

**Open:** the repository’s **Actions** tab and select the latest completed **MLOps CI** run for `main`.

**Point to:** the green **Success** status, the commit, and the green `quality-training` job. Then open `k8s/deployment.yaml` and briefly highlight `replicas: 2`, the readiness/liveness probes, and resource limits. If there is time, show `k8s/service.yaml` and its `LoadBalancer` type.

**Close with:** “This demonstrates the path from reproducible training to API serving, monitoring, local Kubernetes, and automated quality checks. The included deployment is local; this project does not claim a public cloud endpoint.”

## Final recording check

- Keep the recording between 3 and 5 minutes; trim only long waits, not failed output or evidence needed to understand the run.
- Make command output readable and keep the cursor near the result you are explaining.
- Use real outputs from this project. Do not paste placeholder logs or hide an error that changes the result.
- Save the MP4 with your course submission files. The still-image evidence is already stored separately under `docs/screenshots/`.
- After the video, press Ctrl+C in the port-forward window. Leave the Kubernetes resources alone unless you have a separate cleanup instruction.
