import sys
from pathlib import Path

from fastapi import APIRouter, Request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from predict_utils import predict_one

from ..schemas import ClaimInput, PredictionOutput

router = APIRouter()


@router.post("/predict", response_model=PredictionOutput)
def predict_claim(claim: ClaimInput, request: Request):
    assets = request.app.state.assets
    out = predict_one(assets, claim.model_dump())

    threshold = float(assets.metadata["best_threshold"])
    probability = out["probability"]
    label = "Fraud" if probability >= threshold else "Not Fraud"

    return PredictionOutput(
        label=label,
        probability=probability,
        confidence=out["confidence"],
        threshold_used=threshold,
    )
