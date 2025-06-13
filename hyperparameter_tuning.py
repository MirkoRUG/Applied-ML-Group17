import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor
import optuna
import warnings
warnings.filterwarnings('ignore')

from movie_score_predictor.data.preprocessing import prepare_features_and_target, get_categorical_features


def objective(trial):
    # suggest hyperparams
    params = {
        'iterations': trial.suggest_int('iterations', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 1),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'verbose': False,
        'random_seed': 42
    }
    
    # train model with suggested params
    model = CatBoostRegressor(**params)
    
    # get cat feature indices
    cat_indices = [i for i, col in enumerate(X_train.columns) if col in categorical_features]
    
    # fit and predict
    model.fit(X_train, y_train, cat_features=cat_indices, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
    predictions = model.predict(X_val)
    
    # return rmse as objective
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    return rmse


def plot_optimization_history(study):
    # plot optimization progress
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # objective value over trials
    values = [trial.value for trial in study.trials if trial.value is not None]
    ax1.plot(values, 'b-', alpha=0.7)
    ax1.set_xlabel('trial')
    ax1.set_ylabel('rmse')
    ax1.set_title('optimization progress')
    ax1.grid(True, alpha=0.3)
    
    # best value progression
    best_values = []
    best_so_far = float('inf')
    for val in values:
        if val < best_so_far:
            best_so_far = val
        best_values.append(best_so_far)
    
    ax2.plot(best_values, 'r-', linewidth=2)
    ax2.set_xlabel('trial')
    ax2.set_ylabel('best rmse')
    ax2.set_title('best score over time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_tuning_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def evaluate_final_model(best_params):
    # train final model with best params
    final_model = CatBoostRegressor(**best_params)
    cat_indices = [i for i, col in enumerate(X_train.columns) if col in categorical_features]
    
    final_model.fit(X_train, y_train, cat_features=cat_indices, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
    
    # test predictions
    test_pred = final_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"\nfinal test results:")
    print(f"rmse: {test_rmse:.4f}")
    print(f"mae: {test_mae:.4f}")
    
    return final_model


if __name__ == "__main__":
    print("loading data...")
    df = pd.read_csv("data.csv")
    
    # prepare data
    X, y = prepare_features_and_target(df, 'score')
    categorical_features = get_categorical_features()
    
    # split data
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    print(f"train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    
    # baseline model
    print("\ntraining baseline model...")
    baseline_params = {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 6,
        'verbose': False,
        'random_seed': 42
    }
    
    baseline_model = CatBoostRegressor(**baseline_params)
    cat_indices = [i for i, col in enumerate(X_train.columns) if col in categorical_features]
    baseline_model.fit(X_train, y_train, cat_features=cat_indices, verbose=False)
    
    baseline_pred = baseline_model.predict(X_val)
    baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))
    print(f"baseline validation rmse: {baseline_rmse:.4f}")
    
    # hyperparameter optimization
    print("\nstarting hyperparameter optimization...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    
    print(f"\nbest trial:")
    print(f"rmse: {study.best_value:.4f}")
    print(f"params: {study.best_params}")
    
    # improvement calculation
    improvement = ((baseline_rmse - study.best_value) / baseline_rmse) * 100
    print(f"\nimprovement: {improvement:.2f}%")
    
    # plot results
    plot_optimization_history(study)
    
    # final evaluation
    final_model = evaluate_final_model(study.best_params)
    
    print(f"\ntuning complete! results saved to hyperparameter_tuning_results.png")