#!/bin/bash
# filepath: /home/dshteyma/experiment_code4/ANLP/scripts/run_parallel.sh

# Set default values
MAX_NEW_TOKENS=2048
OUTPUT_DIR="/mnt/beegfs/mixed-tier/work/dshteyma/output4"
CONTEXT_WINDOWS="[-1, 3, 6]"
NUM_PARALLEL_RUNS=1  # Default to 1
DATASET_NAME="math-algebra"  # Default dataset
MODEL_NAME="Qwen3"  # Default model

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --max_new_tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --context_windows)
      CONTEXT_WINDOWS="$2"
      shift 2
      ;;
    --num_parallel_runs)
      NUM_PARALLEL_RUNS="$2"
      shift 2
      ;;
    --dataset_name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Run parallel instances using SLURM
for i in $(seq 0 $((NUM_PARALLEL_RUNS-1))); do
  # Calculate the starting index for this run
  start_idx=$((i+26))
  
  # Create the sample_indices list for this run
  sample_indices="[${start_idx},$((start_idx+1)),$((start_idx+2)),$((start_idx+3))]"
  # sample_indices="[${start_idx},$((start_idx+1)),$((start_idx+2)),$((start_idx+3))]"
  
  echo "Starting run $i with sample_indices $sample_indices for dataset $DATASET_NAME using model $MODEL_NAME"
  
  # Launch each job with srun on separate GPUs
  srun --partition=g48 \
       --nodes=1 \
       --ntasks=1 \
       --cpus-per-task=100 \
       --gres=gpu:8 \
       --constraint=ampere \
       --job-name="run_${i}" \
       --output="${OUTPUT_DIR}/slurm_${MODEL_NAME}_${DATASET_NAME}_run${i}_%j.out" \
       python main.py \
         --max_new_tokens "$MAX_NEW_TOKENS" \
         --output_dir "$OUTPUT_DIR" \
         --num_samples_per_task 1 \
         --sample_indices "$sample_indices" \
         --dataset_name "$DATASET_NAME" \
         --model_name "$MODEL_NAME" \
         --context_windows "$CONTEXT_WINDOWS" &
  
  # Small delay to avoid submission race conditions
  sleep 5
done

# Wait for all background processes to complete
wait

echo "Finished All SLURM jobs submitted"