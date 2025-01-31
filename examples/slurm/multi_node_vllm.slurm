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

MODEL_ARGS="pretrained=$MODEL_DIRECTORY,gpu_memory_utilisation=0.5,trust_remote_code=False,dtype=bfloat16,max_model_length=8192,tensor_parallel_size=8"
TASK_ARGS="community|gpqa-fr|0|0,community|ifeval-fr|0|0"

srun -N1 -n1 -c $(( SLURM_CPUS_PER_TASK/2 )) --overlap -w $MASTER lighteval vllm "$MODEL_ARGS" "$TASK_ARGS" --custom-tasks $TASK_FILE
