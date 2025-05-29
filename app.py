from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
from catboost import CatBoostRegressor

# Initialize app
app = FastAPI(
    title="Movie Score Prediction API",
    description="Predict movie scores using a trained CatBoost regression model.",
    version="1.0.0"
)

# Define categorical columns
categorical = ["name", "rating", "genre", "released", "director", "writer",
               "star", "country", "company"]

# Load and prepare data, then train model
try:
    data = pd.read_csv("data.csv").dropna(subset=["score", "votes", "gross"])
    X = data.drop(columns=["score"])
    y = data["score"]
    X[categorical] = X[categorical].astype(str)

    expected_columns = list(X.columns)

    model = CatBoostRegressor(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        loss_function='RMSE',
        verbose=0
    )
    model.fit(X, y, cat_features=categorical)

except Exception as e:
    raise RuntimeError(f"Failed to initialize model: {e}")

# Input and Output Schemas
class MovieFeatures(BaseModel):
    name: str = Field(..., example="Inception")
    rating: str = Field(..., example="PG-13")
    genre: str = Field(..., example="Sci-Fi")
    released: str = Field(..., example="2010")
    director: str = Field(..., example="Christopher Nolan")
    writer: str = Field(..., example="Jonathan Nolan")
    star: str = Field(..., example="Leonardo DiCaprio")
    country: str = Field(..., example="USA")
    company: str = Field(..., example="Warner Bros.")
    runtime: Optional[float] = Field(None, example=148.0)
    votes: Optional[float] = Field(None, example=2000000.0)
    gross: Optional[float] = Field(None, example=829895144.0)

class PredictionResponse(BaseModel):
    predicted_score: float

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse, summary="Predict movie score")
def predict(features: MovieFeatures):
    try:
        input_df = pd.DataFrame([features.dict()])
        input_df[categorical] = input_df[categorical].astype(str)

        # Add missing columns with default values
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns to match training data
        input_df = input_df[expected_columns]

        prediction = model.predict(input_df)
        return PredictionResponse(predicted_score=round(float(prediction[0]), 2))

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
