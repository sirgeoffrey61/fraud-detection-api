from __future__ import annotations

import gradio as gr

from predict_utils import load_assets, predict_one


assets = load_assets()


def run_prediction(
    claim_amount,
    patient_age,
    patient_gender,
    provider_type,
    provider_specialty,
    service_type,
    admission_type,
    distance_miles,
    submitted_late,
):
    raw = {
        "Claim_Amount": float(claim_amount),
        "Patient_Age": int(patient_age),
        "Patient_Gender": str(patient_gender),
        "Provider_Type": str(provider_type),
        "Provider_Specialty": str(provider_specialty),
        "Service_Type": str(service_type),
        "Admission_Type": str(admission_type),
        "Provider_Patient_Distance_Miles": float(distance_miles),
        "Claim_Submitted_Late": int(submitted_late),
    }
    out = predict_one(assets, raw)
    return out["label"], f"{out['probability']:.2%}", out["confidence"]


demo = gr.Interface(
    fn=run_prediction,
    inputs=[
        gr.Number(label="Claim Amount", value=1200000),
        gr.Slider(0, 120, value=45, step=1, label="Patient Age"),
        gr.Dropdown(["Female", "Male", "Other"], value="Female", label="Patient Gender"),
        gr.Dropdown(["Specialist Office", "Hospital", "Clinic", "Urgent Care", "Laboratory", "Pharmacy"], value="Specialist Office", label="Provider Type"),
        gr.Dropdown(["Cardiology", "Orthopedics", "General Practice", "Neurology", "Pediatrics", "Oncology", "Psychiatry", "Physical Therapy"], value="Cardiology", label="Provider Specialty"),
        gr.Dropdown(["Laboratory", "Emergency Room", "Ambulance", "Outpatient", "Pharmacy", "Inpatient"], value="Laboratory", label="Service Type"),
        gr.Dropdown(["Elective", "Urgent", "Emergency", "Trauma", "Newborn"], value="Elective", label="Admission Type"),
        gr.Number(label="Provider-Patient Distance (Miles)", value=650),
        gr.Radio([0, 1], value=1, label="Claim Submitted Late"),
    ],
    outputs=[
        gr.Textbox(label="Predicted Class"),
        gr.Textbox(label="Fraud Probability"),
        gr.Textbox(label="Confidence"),
    ],
    title="Fraud Predictive Analysis - Gradio Demo",
    description="Interactive fraud prediction using the same trained pipeline as Streamlit.",
)


if __name__ == "__main__":
    demo.launch()
