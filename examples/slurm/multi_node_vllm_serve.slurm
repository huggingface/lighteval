#! /bin/bash

#SBATCH --job-name=EVALUATE_Llama-3.2-1B-Instruct
#SBATCH --account=brb@h100
#SBATCH --output=evaluation.log
#SBATCH --error=evaluation.log
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:4
#SBATCH --hint=nomultithread
#SBATCH --constraint=h100
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --parsable

set -e
set -x

module purge
module load miniforge/24.9.0
conda activate $WORKDIR/lighteval-h100

function find_available_port {
    printf $(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
}

PORT=$(find_available_port)
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER=${NODELIST[0]}
MASTER_IP=$(hostname --ip-address)

export HF_HOME=$WORKDIR/HF_HOME
export MODEL_DIRECTORY=$WORKDIR/HuggingFace_Models/meta-llama/Llama-3.2-1B-Instruct
export TASK_FILE=$WORKDIR/community_tasks/french_eval.py
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

function set_VLLM_HOST_IP {
    export VLLM_HOST_IP=$(hostname --ip-address);
}
export -f set_VLLM_HOST_IP;

srun -N1 -n1 -c $(( SLURM_CPUS_PER_TASK/2 )) -w $MASTER bash -c "set_VLLM_HOST_IP; ray start --head --port=$PORT --block" &
sleep 5

if [[ $SLURM_NNODES -gt 1 ]]; then
    srun -N $(( SLURM_NNODES-1 )) --ntasks-per-node=1 -c $(( SLURM_CPUS_PER_TASK/2 )) -x $MASTER bash -c "set_VLLM_HOST_IP; ray start --address=$MASTER_IP:$PORT --block" &
    sleep 5
fi

set_VLLM_HOST_IP

MODEL_NAME="Llama-3.2-1B-Instruct"
SERVER_PORT=$(find_available_port)
export OPENAI_API_KEY="I-love-vllm-serve"
export OPENAI_BASE_URL="http://localhost:$SERVER_PORT/v1"

vllm serve $MODEL_DIRECTORY \
    --served-model-name $MODEL_NAME \
    --api-key $OPENAI_API_KEY \
    --enforce-eager \
    --port $SERVER_PORT \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.8 \
    --disable-custom-all-reduce \
    1>vllm.stdout 2>vllm.stderr &


ATTEMPT=0
DELAY=5
MAX_ATTEMPTS=60
until curl -s -o /dev/null -w "%{http_code}" $OPENAI_BASE_URL/models -H "Authorization: Bearer $OPENAI_API_KEY" | grep -E "^2[0-9]{2}$"; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "$ATTEMPT attempts"
    if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
        echo "Failed: the server did not respond any of the $MAX_ATTEMPTS requests."
        exit 1
    fi
    echo "vllm serve is not ready yet"
    sleep $DELAY
done

export TOKENIZER_PATH="$MODEL_DIRECTORY"

TASK_ARGS="community|gpqa-fr|0|0,community|ifeval-fr|0|0"

lighteval endpoint openai "$MODEL_NAME" "$TASK_ARGS" --custom-tasks $TASK_FILE
