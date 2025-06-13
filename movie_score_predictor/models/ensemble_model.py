import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from ..data.preprocessing import prepare_features_and_target, \
    get_categorical_features, clean_and_prepare_data


class MovieScoreEnsemblePredictor:
    """Basic ensemble model for movie score prediction"""

    def __init__(self, n_models=3, model_params=None):
        self.n_models = n_models
        self.models = []
        self.cat_features = get_categorical_features()
        self.feature_cols = None

        # Basic model settings
        self.base_params = {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'loss_function': 'RMSE',
            'verbose': False,
            'early_stopping_rounds': 50
        }

        if model_params:
            self.base_params.update(model_params)

    def _create_models(self):
        models = []
        for i in range(self.n_models):
            params = self.base_params.copy()
            params['random_seed'] = 42 + i
            # Vary parameters slightly
            params['learning_rate'] *= 0.9 + 0.2 * (i % 3)
            params['depth'] += i % 2 + 1
            models.append(params)
        return models

    def train(self, df, target_col='score'):
        X, y = prepare_features_and_target(df, target_col)
        self.feature_cols = list(X.columns)

        # Get categorical feature indices
        cat_indices = [i for i, col in enumerate(X.columns)
                       if col in self.cat_features]

        self.models = []
        model_configs = self._create_models()

        for params in model_configs:
            model = CatBoostRegressor(**params)
            model.fit(X, y, cat_features=cat_indices)
            self.models.append(model)

        # Calculate training error
        preds = np.array([model.predict(X) for model in self.models])
        avg_preds = np.mean(preds, axis=0)
        rmse = np.sqrt(np.mean((y - avg_preds) ** 2))
        return {'rmse': rmse}

    def predict(self, data, return_uncertainty=False):
        if not self.models or not self.feature_cols:
            return None

        if isinstance(data, dict):
            data = pd.DataFrame([data])

        data = clean_and_prepare_data(data)

        # Ensure all columns are present
        for col in self.feature_cols:
            if col not in data.columns:
                data[col] = 'Unknown' if col in self.cat_features else 0

        data = data[self.feature_cols]

        preds = np.array([model.predict(data) for model in self.models])
        mean_preds = np.mean(preds, axis=0)

        if return_uncertainty:
            std = np.std(preds, axis=0)
            if len(mean_preds) == 1:
                return float(mean_preds[0]), float(std[0])
            return mean_preds, std

        if len(mean_preds) == 1:
            return float(mean_preds[0])
        return mean_preds
