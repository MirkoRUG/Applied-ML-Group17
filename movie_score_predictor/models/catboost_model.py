
import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from typing import Union, List, Tuple, Optional
import joblib
from pathlib import Path

from ..data.preprocessing import (
    prepare_features_and_target, 
    get_categorical_features,
    clean_and_prepare_data
)


class MovieScorePredictor:
    
    def __init__(self, model_params: Optional[dict] = None):
        self.model = None
        self.categorical_features = get_categorical_features()
        self.feature_columns = None
        
        # default params
        self.model_params = {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'loss_function': 'RMSE',
            'verbose': False,
            'random_seed': 42,
            'early_stopping_rounds': 50
        }
        
        if model_params:
            self.model_params.update(model_params)
    
    def train(self, df: pd.DataFrame, target_col: str = 'score') -> dict:
        X, y = prepare_features_and_target(df, target_col)
        
        self.feature_columns = list(X.columns)
        
        # get cat feature indices
        cat_features_indices = []
        for i, col in enumerate(X.columns):
            if col in self.categorical_features:
                cat_features_indices.append(i)
        
        self.model = CatBoostRegressor(**self.model_params)
        
        self.model.fit(
            X, y,
            cat_features=cat_features_indices,
            eval_set=(X, y),
            use_best_model=True
        )
        
        # calc metrics
        train_predictions = self.model.predict(X)
        train_rmse = np.sqrt(np.mean((y - train_predictions) ** 2))
        train_mae = np.mean(np.abs(y - train_predictions))
        
        metrics = {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_)),
            'n_features': len(X.columns),
            'n_samples': len(X)
        }
        
        return metrics
    
    def predict(self, data: Union[pd.DataFrame, dict]) -> Union[float, np.ndarray]:
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        data_processed = clean_and_prepare_data(data, drop_columns=['votes', 'gross'])
        
        # ensure all columns present
        for col in self.feature_columns:
            if col not in data_processed.columns:
                if col in self.categorical_features:
                    data_processed[col] = 'Unknown'
                else:
                    data_processed[col] = 0
        
        data_processed = data_processed[self.feature_columns]
        
        predictions = self.model.predict(data_processed)
        
        if len(predictions) == 1:
            return float(predictions[0])
        
        return predictions
    
    def save_model(self, model_path: str) -> None:
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if model_path.endswith('.cbm'):
            self.model.save_model(model_path)

            metadata_path = model_path.replace('.cbm', '_metadata.pkl')
            metadata = {
                'feature_columns': self.feature_columns,
                'categorical_features': self.categorical_features,
                'model_params': self.model_params
            }
            joblib.dump(metadata, metadata_path)
            print(f"Model saved to {model_path}")
            print(f"Metadata saved to {metadata_path}")
        else:
            model_data = {
                'model': self.model,
                'feature_columns': self.feature_columns,
                'categorical_features': self.categorical_features,
                'model_params': self.model_params
            }
            joblib.dump(model_data, model_path)
            print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if model_path.endswith('.cbm'):
            self.model = CatBoostRegressor()
            self.model.load_model(model_path)

            metadata_path = model_path.replace('.cbm', '_metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.feature_columns = metadata['feature_columns']
                self.categorical_features = metadata['categorical_features']
                self.model_params = metadata['model_params']
            else:
                self.feature_columns = None
                self.categorical_features = get_categorical_features()
                self.model_params = {}
        else:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.categorical_features = model_data['categorical_features']
            self.model_params = model_data['model_params']

        print(f"Model loaded from {model_path}")


def get_model(model_path: str = "models/catboost_movie_model.cbm") -> MovieScorePredictor:
    predictor = MovieScorePredictor()
    predictor.load_model(model_path)
    return predictor


def make_prediction(data: Union[pd.DataFrame, dict],
                   model_path: str = "models/catboost_movie_model.cbm") -> Union[float, np.ndarray]:
    predictor = get_model(model_path)
    return predictor.predict(data)


def train_and_save_model(data_path: str = "data.csv",
                        model_path: str = "models/catboost_movie_model.cbm",
                        model_params: Optional[dict] = None) -> dict:
    df = pd.read_csv(data_path)
    
    predictor = MovieScorePredictor(model_params)
    
    metrics = predictor.train(df)
    
    predictor.save_model(model_path)
    
    return metrics
