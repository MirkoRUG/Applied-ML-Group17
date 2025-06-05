"""
CatBoost model implementation for movie score prediction.
Includes model training, loading, and prediction functions.
"""

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
    """
    CatBoost-based movie score prediction model.
    """
    
    def __init__(self, model_params: Optional[dict] = None):
        """
        Initialize the MovieScorePredictor.
        
        Args:
            model_params: Dictionary of CatBoost parameters
        """
        self.model = None
        self.categorical_features = get_categorical_features()
        self.feature_columns = None
        
        # Default model parameters
        self.model_params = {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'loss_function': 'RMSE',
            'verbose': False,
            'random_seed': 42,
            'early_stopping_rounds': 50
        }
        
        # Update with custom parameters if provided
        if model_params:
            self.model_params.update(model_params)
    
    def train(self, df: pd.DataFrame, target_col: str = 'score') -> dict:
        """
        Train the CatBoost model on the provided dataset.
        
        Args:
            df: Training DataFrame
            target_col: Name of target column
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare features and target
        X, y = prepare_features_and_target(df, target_col)
        
        # Store feature columns for later use
        self.feature_columns = list(X.columns)
        
        # Get categorical feature indices
        cat_features_indices = []
        for i, col in enumerate(X.columns):
            if col in self.categorical_features:
                cat_features_indices.append(i)
        
        # Initialize and train model
        self.model = CatBoostRegressor(**self.model_params)
        
        # Train the model
        self.model.fit(
            X, y,
            cat_features=cat_features_indices,
            eval_set=(X, y),
            use_best_model=True
        )
        
        # Calculate training metrics
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
        """
        Make predictions using the trained model.
        
        Args:
            data: Input data (DataFrame or dictionary)
            
        Returns:
            Predicted score(s)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        # Convert input to DataFrame if it's a dictionary
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Prepare the data using the same preprocessing pipeline
        data_processed = clean_and_prepare_data(data, drop_columns=['votes', 'gross'])
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in data_processed.columns:
                # Add missing columns with default values
                if col in self.categorical_features:
                    data_processed[col] = 'Unknown'
                else:
                    data_processed[col] = 0
        
        # Reorder columns to match training data
        data_processed = data_processed[self.feature_columns]
        
        # Make predictions
        predictions = self.model.predict(data_processed)
        
        # Return single value if input was a single sample
        if len(predictions) == 1:
            return float(predictions[0])
        
        return predictions
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save CatBoost model in native .cbm format
        if model_path.endswith('.cbm'):
            self.model.save_model(model_path)

            # Save metadata separately
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
            # Fallback to pickle format for compatibility
            model_data = {
                'model': self.model,
                'feature_columns': self.feature_columns,
                'categorical_features': self.categorical_features,
                'model_params': self.model_params
            }
            joblib.dump(model_data, model_path)
            print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.

        Args:
            model_path: Path to the saved model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if model_path.endswith('.cbm'):
            # Load CatBoost model from native format
            self.model = CatBoostRegressor()
            self.model.load_model(model_path)

            # Load metadata separately
            metadata_path = model_path.replace('.cbm', '_metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.feature_columns = metadata['feature_columns']
                self.categorical_features = metadata['categorical_features']
                self.model_params = metadata['model_params']
            else:
                # Fallback defaults if metadata not found
                self.feature_columns = None
                self.categorical_features = get_categorical_features()
                self.model_params = {}
        else:
            # Load from pickle format
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.categorical_features = model_data['categorical_features']
            self.model_params = model_data['model_params']

        print(f"Model loaded from {model_path}")


def get_model(model_path: str = "models/catboost_movie_model.cbm") -> MovieScorePredictor:
    """
    Load and return a trained CatBoost model.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded MovieScorePredictor instance
    """
    predictor = MovieScorePredictor()
    predictor.load_model(model_path)
    return predictor


def make_prediction(data: Union[pd.DataFrame, dict],
                   model_path: str = "models/catboost_movie_model.cbm") -> Union[float, np.ndarray]:
    """
    Make predictions using a saved model.
    
    Args:
        data: Input data for prediction
        model_path: Path to the saved model file
        
    Returns:
        Predicted score(s)
    """
    predictor = get_model(model_path)
    return predictor.predict(data)


def train_and_save_model(data_path: str = "data.csv",
                        model_path: str = "models/catboost_movie_model.cbm",
                        model_params: Optional[dict] = None) -> dict:
    """
    Train a new model and save it to disk.
    
    Args:
        data_path: Path to the training data CSV file
        model_path: Path to save the trained model
        model_params: Custom model parameters
        
    Returns:
        Training metrics
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Initialize predictor
    predictor = MovieScorePredictor(model_params)
    
    # Train model
    metrics = predictor.train(df)
    
    # Save model
    predictor.save_model(model_path)
    
    return metrics
