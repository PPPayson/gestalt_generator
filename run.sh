#!/bin/bash
#SBATCH --job-name=gestalt_vlm
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

module load ollama/0.21.0-llj6

export OLLAMA_MODELS=/oscar/data/shared/ollama_models
PORT=$(( 11434 + ${SLURM_JOB_ID:-0} % 2000 ))
export OLLAMA_HOST="127.0.0.1:${PORT}"
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL="$WORKERS"
export OLLAMA_KEEP_ALIVE=15m

ollama serve &
sleep 5

module load miniforge3/25.3.0-3
source /oscar/runtime/software/x86_64_v3/miniforge3-25.3.0-3-a6hhdjzejtacz63sugjqnvgosfqz63ul/etc/profile.d/conda.sh
conda activate ../ollama_venv

python ollama_eval.py \
    --model llama3.2-vision:11b

python ollama_eval.py \
    --model llama3.2-vision:90b

