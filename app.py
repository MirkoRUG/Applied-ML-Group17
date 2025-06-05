from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import os

from project_name.models.catboost_model import get_model

# Initialize app
app = FastAPI(
    title="Movie Score Prediction API",
    description="Predict movie scores using a trained CatBoost regression model.",
    version="1.0.0"
)

# Model path
MODEL_PATH = "models/catboost_movie_model.cbm"

# Load model at startup
try:
    if os.path.exists(MODEL_PATH):
        predictor = get_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
        print("Please run 'python train_model.py' to train and save a model first.")
        predictor = None
except Exception as e:
    print(f"Failed to load model: {e}")
    predictor = None

# Input and Output Schemas
class MovieFeatures(BaseModel):
    name: str = Field(..., example="Inception")
    rating: str = Field(..., example="PG-13")
    genre: str = Field(..., example="Sci-Fi")
    year: Optional[int] = Field(None, example=2010)
    released: str = Field(..., example="July 16, 2010 (United States)")
    director: str = Field(..., example="Christopher Nolan")
    writer: str = Field(..., example="Jonathan Nolan")
    star: str = Field(..., example="Leonardo DiCaprio")
    country: str = Field(..., example="USA")
    budget: Optional[float] = Field(None, example=160000000.0)
    company: str = Field(..., example="Warner Bros.")
    runtime: Optional[float] = Field(None, example=148.0)

class PredictionResponse(BaseModel):
    predicted_score: float

# Health check endpoint
@app.get("/health")
def health_check():
    """Check if the API and model are working."""
    if predictor is None:
        return {"status": "unhealthy", "message": "Model not loaded"}
    return {"status": "healthy", "message": "API and model are working"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse, summary="Predict movie score")
def predict(features: MovieFeatures):
    """Predict movie score based on input features."""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train and save a model first using 'python train_model.py'"
        )

    try:
        # Convert input to dictionary
        input_data = features.model_dump()

        # Make prediction using the loaded model
        prediction = predictor.predict(input_data)

        return PredictionResponse(predicted_score=round(float(prediction), 2))

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
