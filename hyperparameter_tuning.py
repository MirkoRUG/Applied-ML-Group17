"""
Lightweight hyperparameter tuning for CatBoost movie score prediction model.
Uses Optuna for Bayesian optimization to find optimal hyperparameters.
Outputs best parameters without retraining the model.
"""

import pandas as pd
import numpy as np
import optuna
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings

from movie_score_predictor.models.catboost_model import MovieScorePredictor
from movie_score_predictor.data.preprocessing import prepare_features_and_target, get_categorical_features

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_hyperparameters(data_path="data.csv", n_trials=100, cv_folds=5, random_seed=42):
    """
    Run hyperparameter optimization for CatBoost model.

    Args:
        data_path: Path to training data CSV
        n_trials: Number of optimization trials
        cv_folds: Number of cross-validation folds
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with optimization results
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    X, y = prepare_features_and_target(df, "score")

    # Get categorical features
    categorical_features = get_categorical_features()
    cat_features_indices = [i for i, col in enumerate(X.columns) if col in categorical_features]

    print(f"Dataset: {len(X)} samples, {len(X.columns)} features")
    print(f"Categorical features: {len(cat_features_indices)}")

    # Initialize cross-validation
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    def define_search_space(trial):
        """Define hyperparameter search space for CatBoost."""
        return {
            # Core hyperparameters (compatible with train_model.py arguments)
            'iterations': trial.suggest_int('iterations', 500, 2000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),

            # Additional optimization parameters
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
            'random_strength': trial.suggest_float('random_strength', 0.1, 10, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),

            # Fixed parameters
            'loss_function': 'RMSE',
            'random_seed': random_seed,
            'verbose': False,
            'early_stopping_rounds': 50
        }

    def objective(trial):
        """Objective function for Optuna optimization."""
        params = define_search_space(trial)

        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            try:
                # Create and train model
                predictor = MovieScorePredictor(model_params=params)
                predictor.model.fit(
                    X_train, y_train,
                    cat_features=cat_features_indices,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                    verbose=False
                )

                # Calculate validation RMSE
                y_pred = predictor.model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                cv_scores.append(rmse)

            except Exception as e:
                print(f"Trial {trial.number} failed on fold {fold}: {e}")
                return float('inf')

        mean_cv_score = np.mean(cv_scores)

        # Progress update
        if trial.number % 10 == 0:
            print(f"Trial {trial.number}: CV RMSE = {mean_cv_score:.4f}")

        return mean_cv_score

    print(f"\nStarting hyperparameter optimization with {n_trials} trials...")

    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=random_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nOptimization completed!")
    print(f"Best CV RMSE: {study.best_value:.4f}")

    return {
        'best_params': study.best_params,
        'best_cv_score': study.best_value,
        'n_trials': len(study.trials),
        'study': study
    }

def save_results(results, output_file="hyperparameter_results.json"):
    """Save optimization results to JSON file."""
    # Prepare clean results for JSON serialization
    clean_results = {
        'timestamp': datetime.now().isoformat(),
        'best_cv_score': results['best_cv_score'],
        'best_params': results['best_params'],
        'n_trials': results['n_trials'],
        'optimization_history': [
            {
                'trial': trial.number,
                'value': trial.value if trial.value is not None else float('inf'),
                'params': trial.params
            }
            for trial in results['study'].trials
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(clean_results, f, indent=2)

    print(f"Results saved to {output_file}")


def create_optimization_plot(study, output_file="optimization_progress.png"):
    """Create visualization of optimization progress."""
    plt.figure(figsize=(12, 8))

    # Get trial data
    trials = study.trials
    trial_numbers = [t.number for t in trials if t.value is not None]
    trial_values = [t.value for t in trials if t.value is not None]

    if not trial_values:
        print("No valid trials to plot")
        return

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Optimization progress
    ax1.plot(trial_numbers, trial_values, 'b-', alpha=0.6, linewidth=1)
    ax1.scatter(trial_numbers, trial_values, c='blue', alpha=0.6, s=20)

    # Add best value line
    best_values = []
    current_best = float('inf')
    for val in trial_values:
        if val < current_best:
            current_best = val
        best_values.append(current_best)

    ax1.plot(trial_numbers, best_values, 'r-', linewidth=2, label=f'Best: {min(trial_values):.4f}')
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('CV RMSE')
    ax1.set_title('Hyperparameter Optimization Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Parameter importance (if available)
    try:
        importance = optuna.importance.get_param_importances(study)
        if importance:
            params = list(importance.keys())[:8]  # Top 8 parameters
            values = [importance[p] for p in params]

            ax2.barh(params, values)
            ax2.set_xlabel('Importance')
            ax2.set_title('Parameter Importance')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Parameter importance not available',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Parameter Importance')
    except:
        ax2.text(0.5, 0.5, 'Parameter importance calculation failed',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Parameter Importance')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Optimization plot saved to {output_file}")


def print_results(results):
    """Print formatted results to console."""
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION RESULTS")
    print("="*60)

    print(f"Best CV RMSE: {results['best_cv_score']:.4f}")
    print(f"Number of trials: {results['n_trials']}")

    print(f"\nBest Hyperparameters:")
    print("-" * 30)

    # Separate core parameters (compatible with train_model.py)
    core_params = ['iterations', 'learning_rate', 'depth']
    other_params = [k for k in results['best_params'].keys() if k not in core_params]

    print("Core parameters (for train_model.py):")
    for param in core_params:
        if param in results['best_params']:
            value = results['best_params'][param]
            if isinstance(value, float):
                print(f"  --{param} {value:.4f}")
            else:
                print(f"  --{param} {value}")

    print(f"\nAdditional optimized parameters:")
    for param in other_params:
        value = results['best_params'][param]
        if isinstance(value, float):
            print(f"  {param}: {value:.4f}")
        else:
            print(f"  {param}: {value}")

    print(f"\nTo use these parameters with train_model.py:")
    core_args = []
    for param in core_params:
        if param in results['best_params']:
            value = results['best_params'][param]
            if isinstance(value, float):
                core_args.append(f"--{param} {value:.4f}")
            else:
                core_args.append(f"--{param} {value}")

    if core_args:
        print(f"python train_model.py {' '.join(core_args)}")

    print("="*60)


def main():
    """Main function to run hyperparameter optimization."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Hyperparameter optimization for CatBoost movie score prediction'
    )
    parser.add_argument('--data_path', type=str, default='data.csv',
                       help='Path to training data CSV')
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of optimization trials')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--output_json', type=str, default='hyperparameter_results.json',
                       help='Output file for results JSON')
    parser.add_argument('--output_plot', type=str, default='optimization_progress.png',
                       help='Output file for optimization plot')

    args = parser.parse_args()

    # Run optimization
    results = optimize_hyperparameters(
        data_path=args.data_path,
        n_trials=args.n_trials,
        cv_folds=args.cv_folds
    )

    # Print results to console
    print_results(results)

    # Save results to JSON
    save_results(results, args.output_json)

    # Create visualization
    create_optimization_plot(results['study'], args.output_plot)


if __name__ == "__main__":
    main()
