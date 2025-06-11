# Hyperparameter Tuning for CatBoost Model

A lightweight hyperparameter optimization tool for the CatBoost movie score prediction model using Optuna.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run hyperparameter optimization:**
   ```bash
   python hyperparameter_tuning.py --n_trials 100
   ```

3. **Use the optimized parameters:**
   The script will output the best parameters and show you how to use them with `train_model.py`:
   ```bash
   python train_model.py --learning_rate 0.0847 --depth 8 --iterations 1200
   ```

## What It Does

- **Finds optimal hyperparameters** using Bayesian optimization (Optuna)
- **Uses cross-validation** for robust parameter evaluation
- **Outputs best parameters** compatible with existing `train_model.py` script
- **Creates visualization** showing optimization progress
- **Saves results** to JSON file for later reference

## Output Files

- `hyperparameter_results.json` - Best parameters and optimization history
- `optimization_progress.png` - Visualization of optimization progress

## Command Line Options

```bash
python hyperparameter_tuning.py [OPTIONS]

Options:
  --data_path TEXT        Path to training data CSV [default: data.csv]
  --n_trials INTEGER      Number of optimization trials [default: 100]
  --cv_folds INTEGER      Number of cross-validation folds [default: 5]
  --output_json TEXT      Output JSON file [default: hyperparameter_results.json]
  --output_plot TEXT      Output plot file [default: optimization_progress.png]
```

## Example Output

```
Best CV RMSE: 0.5847
Number of trials: 100

Best Hyperparameters:
Core parameters (for train_model.py):
  --iterations 1200
  --learning_rate 0.0847
  --depth 8

To use these parameters with train_model.py:
python train_model.py --iterations 1200 --learning_rate 0.0847 --depth 8
```

## Integration with Existing Workflow

This tool is designed to work with your existing training workflow:

1. Run `hyperparameter_tuning.py` to find optimal parameters
2. Use the suggested command to retrain your model with `train_model.py`
3. The optimized model will be saved as usual to `models/catboost_movie_model.cbm`

## Optimized Parameters

The tool optimizes these CatBoost hyperparameters:

- **Core parameters** (compatible with `train_model.py`):
  - `iterations` (500-2000)
  - `learning_rate` (0.01-0.3)
  - `depth` (4-10)

- **Additional parameters** (for advanced optimization):
  - `l2_leaf_reg`, `random_strength`, `bagging_temperature`
  - `border_count`, `min_data_in_leaf`
  - `subsample`, `colsample_bylevel`

## Performance Tips

- **Quick test**: 50 trials (~30 minutes)
- **Standard**: 100 trials (~1 hour)
- **Thorough**: 200+ trials (~2+ hours)

The optimization uses early stopping to skip unpromising parameter combinations, making it efficient even with many trials.
