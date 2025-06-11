# University Cluster Hyperparameter Tuning Guide

This guide explains how to run the CatBoost hyperparameter optimization on your university cluster.

## Quick Start

1. **Upload files to cluster:**
   ```bash
   # Copy the batch scripts to your project directory
   scp hyperparameter_batch_job.sh submit_hyperparameter_job.sh your_username@cluster:/scratch/s5200954/Applied\ ML\ hyperparam\ tuning/Applied-ML-Group17/
   ```

2. **Connect to cluster and submit job:**
   ```bash
   ssh your_username@cluster
   cd "/scratch/s5200954/Applied ML hyperparam tuning/Applied-ML-Group17"
   chmod +x submit_hyperparameter_job.sh
   ./submit_hyperparameter_job.sh
   ```

3. **Monitor the job:**
   ```bash
   squeue -u your_username
   tail -f hyperparam_tuning_JOBID.out
   ```

## Files Included

- **`hyperparameter_batch_job.sh`** - Main SLURM batch script
- **`submit_hyperparameter_job.sh`** - Convenient submission wrapper
- **`CLUSTER_USAGE.md`** - This guide

## Job Configuration

The batch script is configured with:
- **CPUs:** 24 cores
- **Memory:** 24GB RAM
- **Time limit:** 3 hours
- **Trials:** 200 (adjustable)
- **Cross-validation:** 5-fold

## Customization

### Adjust Number of Trials

Edit `hyperparameter_batch_job.sh` line 120:
```bash
# Change from 200 to your desired number
--n_trials 300 \
```

### Modify Resource Requirements

Edit the SLURM parameters in `hyperparameter_batch_job.sh`:
```bash
#SBATCH --cpus-per-task=32     # More CPUs for faster processing
#SBATCH --mem=32G              # More memory if needed
#SBATCH --time=04:00:00        # Longer time limit
```

### Change Email Notifications

Update line 9 in `hyperparameter_batch_job.sh`:
```bash
#SBATCH --mail-user=your_actual_email@university.edu
```

## Monitoring Your Job

### Check Job Status
```bash
squeue -u your_username
```

### View Real-time Output
```bash
tail -f hyperparam_tuning_JOBID.out
```

### Check for Errors
```bash
tail -f hyperparam_tuning_JOBID.err
```

### Cancel Job if Needed
```bash
scancel JOBID
```

## Expected Timeline

| Trials | Estimated Time | Quality |
|--------|---------------|---------|
| 50     | 15-20 min     | Basic optimization |
| 100    | 25-35 min     | Good optimization |
| 200    | 45-60 min     | Recommended |
| 300    | 60-90 min     | Thorough optimization |
| 500    | 90-120 min    | Extensive search |

## Output Files

After completion, you'll find:

1. **`hyperparameter_results_cluster.json`** - Best parameters and optimization history
2. **`optimization_progress_cluster.png`** - Visualization of optimization progress
3. **`hyperparam_tuning_JOBID.out`** - Job output log
4. **`hyperparam_tuning_JOBID.err`** - Error log (should be empty if successful)

## Using the Results

The job will output the optimal parameters at the end:
```bash
python train_model.py --iterations 1200 --learning_rate 0.0847 --depth 8
```

Copy this command and run it on your local machine or cluster to train the optimized model.

## Troubleshooting

### Job Fails to Start
- Check if you're in the correct directory
- Verify `data.csv` and `hyperparameter_tuning.py` exist
- Ensure you have sufficient cluster quota

### Python Module Issues
The script tries multiple Python module names:
- `python/3.9`
- `python/3.8` 
- `Python`
- `python`

If none work, contact your cluster support for the correct module name.

### Memory Issues
If you get out-of-memory errors:
- Reduce `--cv_folds` from 5 to 3
- Reduce `--n_trials` 
- Increase `#SBATCH --mem=32G`

### Time Limit Exceeded
If the job times out:
- Increase `#SBATCH --time=04:00:00`
- Reduce number of trials
- Use fewer CV folds

### Package Installation Fails
The script installs packages to `~/.local/`. If this fails:
- Check cluster documentation for Python package installation
- Contact cluster support
- Try loading a different Python module

## Advanced Usage

### Running Multiple Configurations

You can submit multiple jobs with different parameters:

```bash
# Job 1: Quick optimization
sed 's/--n_trials 200/--n_trials 100/' hyperparameter_batch_job.sh > quick_job.sh
sbatch quick_job.sh

# Job 2: Thorough optimization  
sed 's/--n_trials 200/--n_trials 500/' hyperparameter_batch_job.sh > thorough_job.sh
sbatch thorough_job.sh
```

### Comparing Results

After running multiple jobs, compare results:
```bash
python -c "
import json
files = ['hyperparameter_results_cluster.json', 'quick_results.json', 'thorough_results.json']
for f in files:
    try:
        with open(f) as file:
            data = json.load(file)
        print(f'{f}: CV RMSE = {data[\"best_cv_score\"]:.4f}')
    except:
        pass
"
```

## Support

If you encounter issues:
1. Check the error log: `hyperparam_tuning_JOBID.err`
2. Review cluster documentation
3. Contact your university's cluster support team
4. Ensure your cluster account has sufficient resources/quota

## Performance Tips

- **Start small:** Test with 20-50 trials first
- **Use appropriate resources:** Don't over-request CPUs/memory
- **Monitor progress:** Check output logs periodically
- **Plan timing:** Submit during off-peak hours for faster queue times
