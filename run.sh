#!/bin/bash
#SBATCH -J ollama_eval
#SBATCH -N 1-1
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH -p gpu-he
#SBATCH -C h100


module load ollama/0.21.0-llj6


export OLLAMA_MODELS=/oscar/data/shared/ollama_models

PORT=$((11434 + ${SLURM_JOB_ID:-0} % 2000))
export OLLAMA_HOST="127.0.0.1:${PORT}"

export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_KEEP_ALIVE=15m


# Start Ollama server
ollama serve &
OLLAMA_PID=$!

# Always stop the Ollama server when the job exits
trap 'kill "$OLLAMA_PID" 2>/dev/null' EXIT


# Wait until Ollama is actually ready
echo "Waiting for Ollama server on ${OLLAMA_HOST}..."

until curl -s "http://${OLLAMA_HOST}/api/version" > /dev/null; do

    # Detect if ollama serve crashed during startup
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "ERROR: Ollama server exited during startup."
        exit 1
    fi

    sleep 1
done

echo "Ollama server ready."


# Load Python environment
module load miniforge3/25.3.0-3

source /oscar/runtime/software/x86_64_v3/miniforge3-25.3.0-3-a6hhdjzejtacz63sugjqnvgosfqz63ul/etc/profile.d/conda.sh

conda activate ../ollama_venv


# Models to evaluate
MODELS=(
    "llama3.2-vision:11b"
    "llama3.2-vision:90b"
    "qwen3.6:35b"
    "qwen3.5:35b"
    "qwen3.5:122b"
    "gemma4:26b"
    "gemma4:31b"
    "mistral-medium-3.5:128b"
    "qwen3-vl:32b"
    "qwen2.5vl:32b"
    "qwen2.5vl:72b"
    "llama4:16x17b"
    "llama4:128x17b"
)


# Run evaluation for every model
for MODEL in "${MODELS[@]}"; do

    echo
    echo "========================================"
    echo "Running model: $MODEL"
    echo "========================================"

    if python ollama_eval.py \
        --model "$MODEL" \
        --thinking 
    then
        echo "Completed successfully: $MODEL"
    else
        EXIT_CODE=$?
        echo "ERROR: $MODEL failed with exit code $EXIT_CODE"
        echo "Continuing to next model..."
    fi

done


echo
echo "========================================"
echo "Evaluation sweep finished."
echo "========================================"