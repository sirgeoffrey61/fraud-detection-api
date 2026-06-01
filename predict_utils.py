from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd


@dataclass
class PredictionAssets:
    model: Any
    metadata: Dict[str, Any]


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def get_artifact_paths() -> Tuple[Path, Path]:
    root = get_project_root()
    return root / "artifacts" / "fraud_pipeline.joblib", root / "artifacts" / "model_metadata.json"


def load_assets() -> PredictionAssets:
    model_path, meta_path = get_artifact_paths()
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Missing model artifacts. Expected files:\n"
            f"- {model_path}\n"
            f"- {meta_path}\n"
            "Run the notebook export cell first."
        )

    model = joblib.load(model_path)
    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return PredictionAssets(model=model, metadata=metadata)


def _base_defaults() -> Dict[str, Any]:
    return {
        "Claim_Amount": 1200000.0,
        "Patient_Age": 45,
        "Patient_Gender": "Female",
        "Patient_City": "CityA",
        "Patient_State": "CA",
        "Hospital_ID": 1001,
        "Provider_Type": "Specialist Office",
        "Provider_Specialty": "Cardiology",
        "Provider_City": "CityB",
        "Provider_State": "CA",
        "Diagnosis_Code": "D100",
        "Procedure_Code": 10001,
        "Number_of_Procedures": 2,
        "Admission_Type": "Elective",
        "Discharge_Type": "Home",
        "Length_of_Stay_Days": 60,
        "Service_Type": "Laboratory",
        "Deductible_Amount": 1200.0,
        "CoPay_Amount": 200.0,
        "Number_of_Previous_Claims_Patient": 1,
        "Number_of_Previous_Claims_Provider": 4,
        "Provider_Patient_Distance_Miles": 650.0,
        "Claim_Submitted_Late": 1,
        "Claim_Service_Gap_Days": 12,
        "Days_To_Policy_Expiry": 50,
        "Claim_Month": 6,
        "Claim_DayOfWeek": 2,
        "Amount_Per_Procedure": 600000.0,
        "PrevClaim_PatientToProvider_Ratio": 0.2,
    }


def prepare_input_row(raw_inputs: Dict[str, Any], input_columns: list[str]) -> pd.DataFrame:
    row = _base_defaults()
    row.update(raw_inputs)

    # Keep derived fields coherent with user values.
    procedures = max(float(row.get("Number_of_Procedures", 1)), 1.0)
    row["Amount_Per_Procedure"] = float(row.get("Claim_Amount", 0.0)) / procedures
    provider_prev = float(row.get("Number_of_Previous_Claims_Provider", 0.0))
    patient_prev = float(row.get("Number_of_Previous_Claims_Patient", 0.0))
    row["PrevClaim_PatientToProvider_Ratio"] = patient_prev / (provider_prev + 1.0)

    df = pd.DataFrame([row]).reindex(columns=input_columns)
    return df


def predict_one(assets: PredictionAssets, raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
    input_columns = assets.metadata["input_columns"]
    model = assets.model

    input_df = prepare_input_row(raw_inputs, input_columns)
    pred = int(model.predict(input_df)[0])

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(input_df)[0, 1])
    else:
        prob = float(pred)

    label = "Fraud" if pred == 1 else "Not Fraud"
    if prob >= 0.75:
        confidence = "High"
    elif prob >= 0.45:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "label": label,
        "probability": prob,
        "confidence": confidence,
        "raw_prediction": pred,
    }
