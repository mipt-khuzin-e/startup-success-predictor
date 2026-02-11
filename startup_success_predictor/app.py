"""FastAPI application for startup success prediction."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from startup_success_predictor.config import get_settings

logger = logging.getLogger(__name__)


class StartupFeatures(BaseModel):
    """Input features for prediction."""

    features: dict[str, float] = Field(
        ...,
        description="Dictionary of numerical features",
        examples=[{"feature_1": 0.5, "feature_2": 1.2}],
    )
    categorical: dict[str, int] = Field(
        default_factory=dict,
        description="Dictionary of categorical features (encoded as integers)",
        examples=[{"category_1": 0, "category_2": 1}],
    )


class PredictionResponse(BaseModel):
    """Prediction response."""

    success: bool = Field(..., description="Predicted success (True/False)")
    probability: float = Field(
        ...,
        description="Probability of success",
        ge=0.0,
        le=1.0,
    )


def load_onnx_model(model_path: Path) -> ort.InferenceSession:
    """Load ONNX model for CPU inference."""
    logger.info("Loading ONNX model from: %s", model_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    logger.info("Model loaded successfully!")
    return session


def run_inference(
    session: ort.InferenceSession, input_data: StartupFeatures
) -> PredictionResponse:
    """Run ONNX inference on a single sample and return prediction."""
    all_features = {**input_data.features, **input_data.categorical}
    feature_values = list(all_features.values())
    input_array = np.array([feature_values], dtype=np.float32)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logits = session.run([output_name], {input_name: input_array})[0]

    probability = float(1 / (1 + np.exp(-logits[0][0])))
    success = probability > 0.5

    return PredictionResponse(success=success, probability=probability)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model on startup, clean up on shutdown."""
    model_path = get_settings().models_dir / "classifier.onnx"

    if not model_path.exists():
        logger.warning("Model not found at %s", model_path)
        logger.warning(
            "API will start but predictions will fail until model is available."
        )
    else:
        app.state.onnx_session = load_onnx_model(model_path)

    yield


app = FastAPI(
    title="Startup Success Predictor API",
    description="Predict startup success using WGAN-GP augmented classifier",
    version="0.1.0",
    lifespan=lifespan,
)


def _get_session(request: Request) -> ort.InferenceSession:
    """Get ONNX session from app state or raise 503."""
    session: ort.InferenceSession | None = getattr(
        request.app.state, "onnx_session", None
    )
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return session


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Startup Success Predictor API",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    """Health check endpoint."""
    session = getattr(request.app.state, "onnx_session", None)
    model_status = "loaded" if session is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: StartupFeatures, request: Request) -> PredictionResponse:
    """Predict startup success probability."""
    session = _get_session(request)
    return run_inference(session, input_data)


@app.post("/predict_batch")
async def predict_batch(
    input_data: list[StartupFeatures], request: Request
) -> list[PredictionResponse]:
    """Predict startup success for multiple samples."""
    session = _get_session(request)
    return [run_inference(session, sample) for sample in input_data]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
