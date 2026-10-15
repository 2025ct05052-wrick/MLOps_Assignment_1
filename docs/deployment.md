# Serving and monitoring

## Train before serving

Use Python 3.12, install the project requirements, and train from the repository root:

    python -m venv .venv
    .venv/Scripts/Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python scripts/train.py

Training writes artifacts/heart_disease_model.joblib. It contains the fitted preprocessing
steps and classifier as one sklearn pipeline. The API needs this file to answer predictions.

## Local Docker and dashboards

Start Docker Desktop, then run:

    docker compose up --build -d
    Invoke-RestMethod http://localhost:8000/health
    python scripts/smoke_test_api.py

Send one sample record. The JSON keys are the 13 predictors from the Cleveland dataset. The
optional null fields ca and thal are imputed by the fitted pipeline.

    $body = @{
      age=63; sex=1; cp=1; trestbps=145; chol=233; fbs=1; restecg=2
      thalach=150; exang=0; oldpeak=2.3; slope=3; ca=0; thal=6
    } | ConvertTo-Json
    Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType 'application/json' -Body $body
    Invoke-WebRequest http://localhost:8000/metrics

The response includes the binary prediction, human-readable label, probability of disease,
and model version. Logs contain route, HTTP status, and latency, but no patient values.
Prometheus scrapes the API every 15 seconds. Grafana is available at localhost:3000 with
the demo credentials admin/admin. Its provisioned dashboard shows route request rate, p95
latency, prediction counts, and server-error ratio. Save screenshots under docs/screenshots.

Stop the stack with docker compose down. For a smoke deployment, confirm /health reports
model_loaded true, /predict returns a response, Prometheus shows the heart-disease-api
target as UP, and the Grafana panels receive data after requests.

## Kubernetes deployment

The k8s manifests define two API replicas, HTTP readiness and liveness probes, resource
requests and limits, Prometheus scrape annotations, and a LoadBalancer Service. Build and
load the image into the local cluster, then apply the manifests:

    docker build -t heart-disease-api:latest .
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    kubectl rollout status deployment/heart-disease-api
    kubectl get service heart-disease-api

For Minikube, run minikube service heart-disease-api --url. A cloud LoadBalancer receives
an address after the cloud provider provisions it. The manifest assumes the image is
already available to the cluster; configure a remote registry and image pull secret for
a managed cloud cluster.

## Runtime evidence

The local Docker Compose run was verified: the API health check passed, sample predictions
were returned, the Prometheus target was UP, and the provisioned Grafana dashboard received
metrics.

The Kubernetes deployment was also verified on Docker Desktop's local Kind cluster. The
Deployment rolled out with two ready replicas. Using
`kubectl port-forward service/heart-disease-api 8000:80`, `/health` returned `ok`, `/predict`
returned a result, and `/metrics` returned HTTP 200. Screenshots of the actual local runs
are stored under `docs/screenshots/`.

These are local demonstrations; no public endpoint or cloud deployment is claimed.

