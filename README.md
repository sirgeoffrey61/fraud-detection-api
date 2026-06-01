# Fraud Detection API

Healthcare claims fraud detection served as a FastAPI REST API, backed by a scikit-learn Random Forest model and packaged with Docker.

## What This Project Does

This project scores synthetic healthcare insurance claims for fraud risk using a trained Random Forest classifier. A FastAPI service exposes predictions over HTTP, with model artifacts loaded at startup from `artifacts/`. The pipeline was tuned on imbalanced data: the default 0.5 decision threshold yields F1 0.667, while precision–recall analysis selects 0.2725 as the operating point, improving F1 to 0.713 by trading some precision for recall on rare fraud cases. The same backend powers optional Streamlit and Gradio UIs, and the API is containerized for deployment.

## Project Structure

```
App/
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── routers/
│       ├── __init__.py
│       ├── health.py
│       └── predict.py
├── artifacts/
│   ├── fraud_pipeline.joblib
│   └── model_metadata.json
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── app_streamlit.py
├── app_gradio.py
├── predict_utils.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── test_predict.json
```

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest |
| Default threshold F1 | 0.667 |
| Tuned threshold | 0.2725 |
| Tuned F1 | 0.713 |
| AUC-ROC | 0.876 |

## Quick Start (Docker)

```powershell
git clone <repository-url>
cd App
docker compose up --build
```

```powershell
curl.exe http://127.0.0.1:8000/
```

```powershell
curl.exe http://127.0.0.1:8000/health
```

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@test_predict.json"
```

The `docker-compose.yml` service mounts `./artifacts:/app/artifacts`, so you can replace `fraud_pipeline.joblib` and `model_metadata.json` on the host without rebuilding the image.

## Quick Start (Local)

```powershell
cd App
pip install -r requirements.txt
uvicorn api.main:app --reload
```

```powershell
curl.exe http://127.0.0.1:8000/
```

```powershell
curl.exe http://127.0.0.1:8000/health
```

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@test_predict.json"
```

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health ping |
| GET | `/health` | Model name and tuned threshold |
| POST | `/predict` | Fraud prediction with confidence score |

## Running Tests

```powershell
cd App
pytest tests/ -v
```

Expected: `3 passed`

## Tech Stack

| Layer | Technology |
|-------|------------|
| ML model | scikit-learn RandomForest |
| API | FastAPI + Pydantic |
| Container | Docker + docker-compose |
| UI | Streamlit + Gradio |
| Threshold tuning | Precision-Recall curve, F1 maximization |
