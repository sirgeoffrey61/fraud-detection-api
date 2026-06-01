from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request):
    assets = request.app.state.assets
    return {
        "status": "ok",
        "model": assets.metadata["best_model"],
        "best_threshold": assets.metadata["best_threshold"],
    }
