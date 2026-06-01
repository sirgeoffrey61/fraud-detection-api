import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict_utils import load_assets

from .routers import health, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.assets = load_assets()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(predict.router)
app.include_router(health.router)


@app.get("/")
def root():
    return {"status": "ok", "service": "fraud-detection-api"}
