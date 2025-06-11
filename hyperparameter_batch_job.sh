#!/bin/bash
#SBATCH --job-name=catboost_hyperparam_tuning
#SBATCH --time=03:00:00                    # 3 hours time limit
#SBATCH --cpus-per-task=24                 # 24 CPU cores for parallel processing
#SBATCH --mem=24G                          # 24GB RAM
#SBATCH --partition=cpu                    # Use CPU partition (adjust if needed)
#SBATCH --output=hyperparam_tuning_%j.out  # Output file with job ID
#SBATCH --error=hyperparam_tuning_%j.err   # Error file with job ID
#SBATCH --mail-type=BEGIN,END,FAIL         # Email notifications (optional)
#SBATCH --mail-user=your_email@university.edu  # Replace with your email

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start Time: $(date)"
echo "=========================================="

# Set project directory
PROJECT_DIR="/scratch/s5200954/Applied ML hyperparam tuning/Applied-ML-Group17"
echo "Project Directory: $PROJECT_DIR"

# Change to project directory
cd "$PROJECT_DIR" || {
    echo "ERROR: Cannot change to project directory: $PROJECT_DIR"
    echo "Please check if the path exists and is accessible"
    exit 1
}

echo "Current working directory: $(pwd)"
echo "Directory contents:"
ls -la

# Load necessary modules (adjust based on your cluster's available modules)
echo "=========================================="
echo "Loading required modules..."
echo "=========================================="

# Try different Python module names (clusters vary)
if module avail python 2>/dev/null | grep -q python; then
    # Try common Python module names
    if module avail python/3.9 2>/dev/null | grep -q python/3.9; then
        module load python/3.9
        echo "Loaded python/3.9"
    elif module avail python/3.8 2>/dev/null | grep -q python/3.8; then
        module load python/3.8
        echo "Loaded python/3.8"
    elif module avail Python 2>/dev/null | grep -q Python; then
        module load Python
        echo "Loaded Python module"
    else
        echo "Loading default python module"
        module load python
    fi
else
    echo "No Python module found, trying to use system Python"
fi

# Load GCC if available (often needed for package compilation)
if module avail gcc 2>/dev/null | grep -q gcc; then
    module load gcc
    echo "Loaded GCC module"
fi

# Show loaded modules
echo "Loaded modules:"
module list

# Check Python availability
echo "=========================================="
echo "Checking Python environment..."
echo "=========================================="

python3 --version || {
    echo "ERROR: Python3 not available"
    echo "Trying python..."
    python --version || {
        echo "ERROR: No Python interpreter found"
        exit 1
    }
    # Use python instead of python3
    PYTHON_CMD="python"
}

# Set Python command
PYTHON_CMD=${PYTHON_CMD:-"python3"}
echo "Using Python command: $PYTHON_CMD"

# Check if pip is available
$PYTHON_CMD -m pip --version || {
    echo "ERROR: pip not available"
    echo "Trying to install pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    $PYTHON_CMD get-pip.py --user
    rm get-pip.py
}

# Set up Python path for user installations
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.8/site-packages:$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH"

# Install required packages
echo "=========================================="
echo "Installing Python dependencies..."
echo "=========================================="

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found in $PROJECT_DIR"
    echo "Creating minimal requirements.txt..."
    cat > requirements.txt << EOF
pandas>=2.0.0
catboost>=1.2.0
numpy>=1.24.0
scikit-learn>=1.4.0
optuna>=3.6.0
matplotlib>=3.8.0
EOF
fi

echo "Installing packages from requirements.txt..."
$PYTHON_CMD -m pip install --user -r requirements.txt || {
    echo "ERROR: Failed to install some packages"
    echo "Trying to install core packages individually..."
    
    # Install core packages one by one
    for package in "pandas" "catboost" "numpy" "scikit-learn" "optuna" "matplotlib"; do
        echo "Installing $package..."
        $PYTHON_CMD -m pip install --user "$package" || echo "Warning: Failed to install $package"
    done
}

# Verify key packages are installed
echo "=========================================="
echo "Verifying package installations..."
echo "=========================================="

$PYTHON_CMD -c "
import sys
print(f'Python version: {sys.version}')

packages = ['pandas', 'catboost', 'numpy', 'sklearn', 'optuna', 'matplotlib']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg} - OK')
    except ImportError as e:
        print(f'✗ {pkg} - FAILED: {e}')
        sys.exit(1)

print('All required packages are available!')
"

# Check if hyperparameter_tuning.py exists
if [ ! -f "hyperparameter_tuning.py" ]; then
    echo "ERROR: hyperparameter_tuning.py not found in $PROJECT_DIR"
    echo "Available Python files:"
    ls -la *.py
    exit 1
fi

# Check if data.csv exists
if [ ! -f "data.csv" ]; then
    echo "ERROR: data.csv not found in $PROJECT_DIR"
    echo "Available data files:"
    ls -la *.csv
    exit 1
fi

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=========================================="
echo "Starting hyperparameter optimization..."
echo "=========================================="
echo "Configuration:"
echo "- Trials: 200"
echo "- CV Folds: 5"
echo "- CPU Cores: $SLURM_CPUS_PER_TASK"
echo "- Start Time: $(date)"
echo "=========================================="

# Run hyperparameter tuning with error handling
$PYTHON_CMD hyperparameter_tuning.py \
    --n_trials 200 \
    --cv_folds 5 \
    --data_path data.csv \
    --output_json hyperparameter_results_cluster.json \
    --output_plot optimization_progress_cluster.png || {
    
    echo "=========================================="
    echo "ERROR: Hyperparameter tuning failed!"
    echo "=========================================="
    echo "Attempting diagnosis..."
    
    # Try with fewer trials for debugging
    echo "Trying with reduced parameters for debugging..."
    $PYTHON_CMD hyperparameter_tuning.py \
        --n_trials 5 \
        --cv_folds 3 \
        --data_path data.csv \
        --output_json debug_results.json \
        --output_plot debug_plot.png || {
        
        echo "Even debug run failed. Checking basic functionality..."
        $PYTHON_CMD -c "
from movie_score_predictor.models.catboost_model import MovieScorePredictor
from movie_score_predictor.data.preprocessing import prepare_features_and_target
import pandas as pd
print('Basic imports successful')

df = pd.read_csv('data.csv')
print(f'Data loaded: {len(df)} rows')

X, y = prepare_features_and_target(df, 'score')
print(f'Features prepared: {X.shape}')
print('Basic functionality test passed')
"
    }
    
    exit 1
}

echo "=========================================="
echo "Hyperparameter optimization completed!"
echo "=========================================="
echo "End Time: $(date)"

# Display results summary
echo "Generated files:"
ls -la hyperparameter_results_cluster.json optimization_progress_cluster.png 2>/dev/null || echo "Output files not found"

# Show brief results if JSON file exists
if [ -f "hyperparameter_results_cluster.json" ]; then
    echo "=========================================="
    echo "Results Summary:"
    echo "=========================================="
    $PYTHON_CMD -c "
import json
try:
    with open('hyperparameter_results_cluster.json', 'r') as f:
        results = json.load(f)
    print(f'Best CV RMSE: {results[\"best_cv_score\"]:.4f}')
    print(f'Number of trials: {results[\"n_trials\"]}')
    print('Best parameters:')
    for param, value in results['best_params'].items():
        if param in ['iterations', 'learning_rate', 'depth']:
            print(f'  {param}: {value}')
except Exception as e:
    print(f'Could not parse results: {e}')
"
fi

echo "=========================================="
echo "Job completed successfully!"
echo "=========================================="

# Copy results to a backup location (optional)
BACKUP_DIR="$HOME/hyperparam_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp hyperparameter_results_cluster.json optimization_progress_cluster.png "$BACKUP_DIR/" 2>/dev/null
echo "Results backed up to: $BACKUP_DIR"

echo "To use the optimized parameters with train_model.py:"
if [ -f "hyperparameter_results_cluster.json" ]; then
    $PYTHON_CMD -c "
import json
try:
    with open('hyperparameter_results_cluster.json', 'r') as f:
        results = json.load(f)
    params = results['best_params']
    core_params = []
    for param in ['iterations', 'learning_rate', 'depth']:
        if param in params:
            value = params[param]
            if isinstance(value, float):
                core_params.append(f'--{param} {value:.4f}')
            else:
                core_params.append(f'--{param} {value}')
    if core_params:
        print(f'python train_model.py {\" \".join(core_params)}')
except:
    pass
"
fi
