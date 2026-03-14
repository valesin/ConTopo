#!/bin/bash

# Configuration
BASE_SWEEP=""
META_SWEEP=""
DRY_RUN=false

# Help message
usage() {
    echo "Usage: $0 --base-sweep <base_sweep_name> --meta-sweep <meta_sweep_name> [options]"
    echo ""
    echo "Options:"
    echo "  --base-sweep    Name of the base model training sweep (e.g., training_small)"
    echo "  --meta-sweep    Name of the metalearning sweep (e.g., metalearning)"
    echo "  --dry-run       Print commands without executing them"
    echo "  -h, --help      Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-sweep)
            BASE_SWEEP="$2"
            shift 2
            ;;
        --meta-sweep)
            META_SWEEP="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$BASE_SWEEP" || -z "$META_SWEEP" ]]; then
    echo "Error: --base-sweep and --meta-sweep are required."
    usage
fi

# Function to run a command
run_cmd() {
    local step_desc=$1
    local cmd=$2

    echo "--------------------------------------------------------------------------------"
    echo "Step: $step_desc"
    echo "Running: $cmd"
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would execute command above."
    else
        eval "$cmd"
        local exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "Error: Step '$step_desc' failed with exit code $exit_code."
            exit $exit_code
        fi
    fi
    echo "Step completed successfully."
}

echo "Starting ConTopo Pipeline Execution"
echo "Base Sweep: $BASE_SWEEP"
echo "Meta Sweep: $META_SWEEP"
echo "Dry Run:    $DRY_RUN"

# Phase A: Base Model Population
run_cmd "Train base models" "uv run scripts/01_train_models.py --multirun +sweeps=$BASE_SWEEP"
run_cmd "Cache inference data" "uv run scripts/02_cache_inference.py --multirun +sweeps=$BASE_SWEEP"
run_cmd "Compute topographic profiles" "uv run scripts/03_compute_profiles.py --multirun +sweeps=$BASE_SWEEP"

# Phase B: Metalearning (Adapters)
run_cmd "Train adapters" "uv run scripts/05_train_adapters.py --multirun +sweeps=$META_SWEEP"

echo "--------------------------------------------------------------------------------"
echo "ConTopo Pipeline Execution Finished Successfully!"
