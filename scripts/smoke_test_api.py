"""Check that a running local API is ready and accepts the sample record."""

import argparse
import json
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--payload",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "samples" / "predict.json",
    )
    args = parser.parse_args()
    health = requests.get(f"{args.base_url.rstrip('/')}/health", timeout=5)
    health.raise_for_status()
    if not health.json().get("model_loaded"):
        raise RuntimeError(f"API is not ready: {health.text}")

    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    response = requests.post(
        f"{args.base_url.rstrip('/')}/predict", json=payload, timeout=10
    )
    response.raise_for_status()
    result = response.json()
    required = {"prediction", "label", "confidence", "model_version"}
    if not required.issubset(result):
        raise RuntimeError(f"Prediction response is missing fields: {result}")
    print(json.dumps({"health": health.json(), "prediction": result}, indent=2))


if __name__ == "__main__":
    main()

