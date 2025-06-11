#!/bin/bash
# Script to check cluster configuration and help configure the batch job

echo "=========================================="
echo "Cluster Configuration Checker"
echo "=========================================="

echo "1. Checking available partitions..."
echo "Available partitions:"
sinfo -s 2>/dev/null || {
    echo "sinfo command not available, trying alternative..."
    scontrol show partition 2>/dev/null | grep PartitionName | head -5
}

echo ""
echo "2. Checking default partition..."
scontrol show config 2>/dev/null | grep DefaultPartition || echo "Could not determine default partition"

echo ""
echo "3. Checking available nodes and resources..."
echo "Node information:"
sinfo -N -l 2>/dev/null | head -10 || echo "Could not get node information"

echo ""
echo "4. Checking your current limits..."
echo "Your account limits:"
sacctmgr show user $USER -s 2>/dev/null || echo "Could not get account limits"

echo ""
echo "5. Checking queue status..."
echo "Current queue:"
squeue -u $USER 2>/dev/null || echo "No jobs in queue"

echo ""
echo "=========================================="
echo "Recommendations:"
echo "=========================================="

# Try to determine the best partition
echo "Checking for common partition names..."
PARTITIONS=$(sinfo -h -o "%P" 2>/dev/null | tr -d '*' | sort -u)

if [ -n "$PARTITIONS" ]; then
    echo "Available partitions found:"
    for partition in $PARTITIONS; do
        echo "  - $partition"
    done
    
    # Suggest the best partition
    if echo "$PARTITIONS" | grep -q "regular"; then
        SUGGESTED="regular"
    elif echo "$PARTITIONS" | grep -q "compute"; then
        SUGGESTED="compute"
    elif echo "$PARTITIONS" | grep -q "cpu"; then
        SUGGESTED="cpu"
    elif echo "$PARTITIONS" | grep -q "normal"; then
        SUGGESTED="normal"
    else
        SUGGESTED=$(echo "$PARTITIONS" | head -1)
    fi
    
    echo ""
    echo "Suggested partition: $SUGGESTED"
    echo ""
    echo "To use this partition, add this line to hyperparameter_batch_job.sh:"
    echo "#SBATCH --partition=$SUGGESTED"
    
else
    echo "Could not determine available partitions."
    echo "Try running without specifying a partition (use default)."
fi

echo ""
echo "=========================================="
echo "Resource Recommendations:"
echo "=========================================="

# Check available resources
MAX_CPUS=$(sinfo -h -o "%c" 2>/dev/null | sort -n | tail -1)
MAX_MEM=$(sinfo -h -o "%m" 2>/dev/null | sort -n | tail -1)

if [ -n "$MAX_CPUS" ]; then
    echo "Maximum CPUs per node: $MAX_CPUS"
    if [ "$MAX_CPUS" -lt 24 ]; then
        echo "Recommendation: Reduce --cpus-per-task to $MAX_CPUS or less"
    fi
fi

if [ -n "$MAX_MEM" ]; then
    echo "Maximum memory per node: ${MAX_MEM}MB"
    MAX_MEM_GB=$((MAX_MEM / 1024))
    if [ "$MAX_MEM_GB" -lt 24 ]; then
        echo "Recommendation: Reduce --mem to ${MAX_MEM_GB}G or less"
    fi
fi

echo ""
echo "=========================================="
echo "Quick Fix Commands:"
echo "=========================================="

echo "Option 1: Try without partition (use default):"
echo "  # The current script should work now"
echo ""

if [ -n "$SUGGESTED" ]; then
    echo "Option 2: Add specific partition:"
    echo "  sed -i '/# Note: Partition line removed/a #SBATCH --partition=$SUGGESTED' hyperparameter_batch_job.sh"
    echo ""
fi

echo "Option 3: Reduce resources if needed:"
echo "  sed -i 's/--cpus-per-task=24/--cpus-per-task=16/' hyperparameter_batch_job.sh"
echo "  sed -i 's/--mem=24G/--mem=16G/' hyperparameter_batch_job.sh"
echo ""

echo "Option 4: Shorter test run:"
echo "  sed -i 's/--n_trials 200/--n_trials 50/' hyperparameter_batch_job.sh"
echo "  sed -i 's/--time=03:00:00/--time=01:00:00/' hyperparameter_batch_job.sh"

echo ""
echo "After making changes, try submitting again:"
echo "  ./submit_hyperparameter_job.sh"
