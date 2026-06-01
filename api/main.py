import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict_utils import load_assets

from .routers import health, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.assets = load_assets()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sirgeoffrey61.github.io"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(health.router)


@app.get("/")
def root():
    return {"status": "ok", "service": "fraud-detection-api"}
