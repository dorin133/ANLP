#!/bin/bash
#SBATCH -p g48
#SBATCH --job-name=CoT_hypothesis
#SBATCH --qos=high
#SBATCH --nodes=1          # Number of nodes
#SBATCH --ntasks=1         # Number of tasks
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --array=1-1        # Array range
#SBATCH --output=/dev/null   # Discard standard output (because we write to the log.txt file)
#SBATCH --constraint=ampere  # GPU type

# Get the current date and time
current_time=$(date +"%d-%m_%H-%M-%S")
# OUTPUT_DIR="./output_runs/training_outputs_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${current_time}"
OUTPUT_DIR="./output_runs/training_outputs_job_check"

mkdir_is_exists() {
    if [ -d "$1" ]; then
        echo "Directory '$1' already exists."
    else
        mkdir -p "$1"
        echo "Directory '$1' created."
    fi
}
# Create the output directory and the sub folder for the experiment code if it doesn't exist
mkdir_is_exists $OUTPUT_DIR
mkdir_is_exists $OUTPUT_DIR/experiment_code
mkdir_is_exists $OUTPUT_DIR/logs

# activate the conda environment
source /mnt/beegfs/mixed-tier/work/dshteyma/miniforge3/bin/activate workenv

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Print start message
echo "Starting math reasoning evaluation script..."

# Copy the .py files used to run main.py and requirements to the logs directory
git log -n 1 > $OUTPUT_DIR/commit.txt
pip freeze > $OUTPUT_DIR/pip_freeze.txt
echo $0 $ARGS $current_time > $OUTPUT_DIR/cmd.txt
cp -r ./main.py $OUTPUT_DIR/experiment_code
cp -r ./hypothesis_functions.py $OUTPUT_DIR/experiment_code

# Run the Python script
python --output_dir $OUTPUT_DIR/logs --num_samples_per_task 10 --overwrite_output_dir main.py

# Print completion message
echo "Evaluation complete! Results saved to $OUTPUT_DIR."