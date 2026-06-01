#!/usr/bin/env bash
# Optional: set MODEL_JOBLIB_URL in Render (or locally) to download the pipeline at build time.
set -euo pipefail

ARTIFACT="artifacts/fraud_pipeline.joblib"

if [ -f "$ARTIFACT" ] && [ -s "$ARTIFACT" ]; then
  echo "Model artifact already present: $ARTIFACT"
  exit 0
fi

if [ -z "${MODEL_JOBLIB_URL:-}" ]; then
  echo "No MODEL_JOBLIB_URL set; expecting Git LFS to provide $ARTIFACT"
  exit 0
fi

mkdir -p artifacts
echo "Downloading model from MODEL_JOBLIB_URL..."
curl -fsSL "$MODEL_JOBLIB_URL" -o "$ARTIFACT"
echo "Saved $ARTIFACT ($(wc -c < "$ARTIFACT") bytes)"
