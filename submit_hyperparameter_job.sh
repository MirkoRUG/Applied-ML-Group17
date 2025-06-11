#!/bin/bash
# Quick submission script for hyperparameter tuning job

echo "=========================================="
echo "CatBoost Hyperparameter Tuning Job Submission"
echo "=========================================="

# Check if we're in the right directory
PROJECT_DIR="/scratch/s5200954/Applied ML hyperparam tuning/Applied-ML-Group17"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found: $PROJECT_DIR"
    echo "Please check the path and try again."
    exit 1
fi

cd "$PROJECT_DIR"

# Check required files
echo "Checking required files..."
if [ ! -f "hyperparameter_tuning.py" ]; then
    echo "ERROR: hyperparameter_tuning.py not found"
    exit 1
fi

if [ ! -f "data.csv" ]; then
    echo "ERROR: data.csv not found"
    exit 1
fi

if [ ! -f "hyperparameter_batch_job.sh" ]; then
    echo "ERROR: hyperparameter_batch_job.sh not found"
    echo "Please ensure the batch script is in the project directory"
    exit 1
fi

echo "✓ All required files found"

# Make batch script executable
chmod +x hyperparameter_batch_job.sh

# Submit the job
echo "Submitting job to SLURM..."
JOB_ID=$(sbatch hyperparameter_batch_job.sh | grep -o '[0-9]*')

if [ -n "$JOB_ID" ]; then
    echo "✓ Job submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo ""
    echo "Monitor your job with:"
    echo "  squeue -u $USER"
    echo "  squeue -j $JOB_ID"
    echo ""
    echo "View output in real-time:"
    echo "  tail -f hyperparam_tuning_${JOB_ID}.out"
    echo ""
    echo "Check for errors:"
    echo "  tail -f hyperparam_tuning_${JOB_ID}.err"
    echo ""
    echo "Cancel job if needed:"
    echo "  scancel $JOB_ID"
else
    echo "ERROR: Job submission failed"
    echo "Check your SLURM configuration and try again"
    exit 1
fi

echo "=========================================="
echo "Expected runtime: 30-60 minutes for 200 trials"
echo "Results will be saved as:"
echo "  - hyperparameter_results_cluster.json"
echo "  - optimization_progress_cluster.png"
echo "=========================================="
