# Deployment evidence

These screenshots were captured from the actual local Docker Desktop demo:

- `prometheus-target-up.png` — Prometheus reports the API scrape target as UP.
- `grafana-heart-disease-dashboard.png` — provisioned dashboard with observed API metrics.
- `kubernetes-verification.png` — two ready API pods and successful `/health`, `/predict`,
  and `/metrics` checks through `kubectl port-forward`.

The first GitHub Actions run for commit `258ec97` passed. Capture its completed run
page as `github-actions-ci-passed.png` before final submission.

