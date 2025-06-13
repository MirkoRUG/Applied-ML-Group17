# Movie Score Predictor

Predicts movie scores using CatBoost machine learning models. Takes input like director, genre, runtime, and budget to estimate IMDb-style ratings.

## Quick Start

**Docker (recommended):**
```bash
docker compose up --develop
```
Open `http://localhost:8501/`

**Local setup:**
```bash
# Option 1: Using conda (includes all dependencies)
conda env create -f environment.yml
conda activate movieapi
streamlit run main.py

# Option 2: Using pip (Python package manager)
pip install -r requirements.txt
streamlit run main.py
```

## What's Included

- `main.py` - Fast predictions using pre-trained model
- `streamlit_ensemble_app.py` - Ensemble predictions with uncertainty estimates  
- `train_model.py` - Train new models from scratch
- `hyperparameter_tuning.py` - Optimize model parameters
- `movie_score_predictor/` - Core ML pipeline and preprocessing

## Training Models

```bash
# Train with default settings
python train_model.py --data_path data.csv

# Optimize hyperparameters (takes ~30 minutes)
python hyperparameter_tuning.py
```

## Two Apps Available

**main.py**: Quick predictions, loads instantly
**streamlit_ensemble_app.py**: Slower startup but includes confidence intervals

Both apps provide web interfaces for entering movie details and getting score predictions.