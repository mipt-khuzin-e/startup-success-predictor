"""FastAPI application for startup success prediction."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from startup_success_predictor.config import get_settings

# Initialize FastAPI app
app = FastAPI(
    title="Startup Success Predictor API",
    description="Predict startup success using WGAN-GP augmented classifier",
    version="0.1.0",
)

# Global model session
onnx_session: ort.InferenceSession | None = None


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
    """
    Load ONNX model.

    Args:
        model_path: Path to ONNX model

    Returns:
        ONNX Runtime inference session
    """
    print(f"Loading ONNX model from: {model_path}")
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    print("Model loaded successfully!")
    return session


@app.on_event("startup")
async def startup_event() -> None:
    """Load model on startup."""
    global onnx_session

    settings = get_settings()
    model_path = settings.models_dir / "classifier.onnx"

    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}")
        print("API will start but predictions will fail until model is available.")
    else:
        onnx_session = load_onnx_model(model_path)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Startup Success Predictor API",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    model_status = "loaded" if onnx_session is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: StartupFeatures) -> PredictionResponse:
    """
    Predict startup success.

    Args:
        input_data: Input features

    Returns:
        Prediction response with success flag and probability
    """
    if onnx_session is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs.",
        )

    try:
        # Combine features and categorical
        all_features = {**input_data.features, **input_data.categorical}

        # Convert to numpy array (assuming features are in correct order)
        # In production, you would need to ensure feature order matches training
        feature_values = list(all_features.values())
        input_array = np.array([feature_values], dtype=np.float32)

        # Run inference
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name

        logits = onnx_session.run([output_name], {input_name: input_array})[0]

        # Apply sigmoid to get probability
        probability = float(1 / (1 + np.exp(-logits[0][0])))

        # Determine success
        success = probability > 0.5

        return PredictionResponse(
            success=success,
            probability=probability,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e!s}",
        ) from e


@app.post("/predict_batch")
async def predict_batch(
    input_data: list[StartupFeatures],
) -> list[PredictionResponse]:
    """
    Predict startup success for multiple samples.

    Args:
        input_data: List of input features

    Returns:
        List of prediction responses
    """
    if onnx_session is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs.",
        )

    try:
        results = []
        for sample in input_data:
            result = await predict(sample)
            results.append(result)

        return results

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {e!s}",
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
