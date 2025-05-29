from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from catboost import CatBoostRegressor

app = FastAPI()

# standard model input schema definition
class MovieFeatures(BaseModel):
    name: str
    rating: str
    genre: str
    released: str
    director: str
    writer: str
    star: str
    country: str
    company: str
    runtime: Optional[float] = None
    votes: Optional[float] = None
    gross: Optional[float] = None


data = pd.read_csv("data.csv").dropna(subset=["score", "votes", "gross"])
X = data.drop(columns=["score"])
y = data["score"]

categorical = ["name", "rating", "genre", "released", "director", "writer",
               "star", "country", "company"]

X[categorical] = X[categorical].astype(str)

model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, loss_function='RMSE', verbose=0)
model.fit(X, y, cat_features=categorical)

# reordering input in the correct order
expected_columns = list(X.columns)
#post crud to make the model predict
@app.post("/predict")
def predict(features: MovieFeatures):
    input_df = pd.DataFrame([features.dict()])
    input_df[categorical] = input_df[categorical].astype(str)

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0  

    # need to reorder columns to match the training data's order
    input_df = input_df[expected_columns]

    prediction = model.predict(input_df)
    return {"predicted_score": prediction[0]}
