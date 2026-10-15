# Short pipeline recording outline

Target length: 3 to 5 minutes.

1. Show the project structure and explain that the raw Cleveland subset is
   versioned separately from the cleaning and preprocessing code.
2. Run the acquisition command or show its printed SHA-256 values, then show
   the EDA output folder and class-balance and missingness figures.
3. Start model training. Show the two candidate runs in the MLflow UI and the
   metrics and ROC artifacts logged for each.
4. Start the API stack with Docker Compose. Show the healthy model status and
   submit the sample JSON request to /predict.
5. Show Prometheus scraping the API and open the provisioned Grafana overview.
6. Finish by showing the GitHub Actions workflow and Kubernetes manifests.

Record the real command output and local dashboards after running the project.
Do not present sample or placeholder output as deployment evidence. Store the
recording alongside the submission, and save screenshots under docs/screenshots.

