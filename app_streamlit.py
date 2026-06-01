from __future__ import annotations

import streamlit as st

from predict_utils import load_assets, predict_one


st.set_page_config(page_title="Fraud Predictive Analysis", layout="wide")
st.title("Fraud Predictive Analysis - Streamlit")


@st.cache_resource(show_spinner=False)
def get_assets():
    return load_assets()


try:
    assets = get_assets()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()


metrics = assets.metadata.get("metrics", {})
col_a, col_b, col_c = st.columns(3)
col_a.metric("Best Model", str(assets.metadata.get("best_model", "N/A")))
col_b.metric("Test F1", f"{float(metrics.get('Test_F1', 0.0)):.4f}")
col_c.metric("Test ROC-AUC", f"{float(metrics.get('Test_ROC_AUC', 0.0)):.4f}")

st.markdown("### Manual Claim Input")

with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)
    claim_amount = c1.number_input("Claim Amount", min_value=0.0, value=1200000.0, step=1000.0)
    patient_age = c2.number_input("Patient Age", min_value=0, max_value=120, value=45, step=1)
    distance_miles = c3.number_input("Provider-Patient Distance (Miles)", min_value=0.0, value=650.0, step=1.0)

    c4, c5, c6 = st.columns(3)
    patient_gender = c4.selectbox("Patient Gender", ["Female", "Male", "Other"])
    provider_type = c5.selectbox("Provider Type", ["Specialist Office", "Hospital", "Clinic", "Urgent Care", "Laboratory", "Pharmacy"])
    provider_specialty = c6.selectbox(
        "Provider Specialty",
        ["Cardiology", "Orthopedics", "General Practice", "Neurology", "Pediatrics", "Oncology", "Psychiatry", "Physical Therapy"],
    )

    c7, c8, c9 = st.columns(3)
    service_type = c7.selectbox("Service Type", ["Laboratory", "Emergency Room", "Ambulance", "Outpatient", "Pharmacy", "Inpatient"])
    admission_type = c8.selectbox("Admission Type", ["Elective", "Urgent", "Emergency", "Trauma", "Newborn"])
    submitted_late = c9.selectbox("Claim Submitted Late", [0, 1], index=1)

    submitted = st.form_submit_button("Predict Fraud Risk")

if submitted:
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

    st.markdown("### Prediction Result")
    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted Class", out["label"])
    r2.metric("Fraud Probability", f"{out['probability']:.2%}")
    r3.metric("Confidence", out["confidence"])

st.markdown("### Fraud Trigger Guidance")
st.write(
    "- Higher risk often appears with `Specialist Office` provider types and certain specialties.\n"
    "- Long provider-patient distance can increase risk.\n"
    "- `Elective/Urgent` admissions and selected service types may have elevated fraud rates.\n"
    "- Use probability + business threshold for final review decisions."
)
