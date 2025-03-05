#!/bin/bash
#SBATCH --job-name=-multilingual-eval-inceptionai/jais-family-6p7b-ar
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=11
#SBATCH --gpus=1
#SBATCH --output=/fsx/hynek_kydlicek/training-logs/multilingual-lighteval/eval-logs/inceptionai/jais-family-6p7b/eval-%A.out
#SBATCH --error=/fsx/hynek_kydlicek/training-logs/multilingual-lighteval/eval-logs/inceptionai/jais-family-6p7b/eval-%A.out
#SBATCH --requeue

###########################################
# [BEGINING] ADAPT TO YOUR ENVIRONMENT
source /admin/home/hynek_kydlicek/.bashrc
source /fsx/hynek_kydlicek/miniconda3/etc/profile.d/conda.sh
conda activate /fsx/hynek_kydlicek/miniconda3/envs/lighteval-multilang


export HF_HOME=/fsx/hynek_kydlicek/.cache/huggingface
export HF_DATASETS_OFFLINE=0
export OPENAI_API_KEY=None

# [END] ADAPT TO YOUR ENVIRONMENT
###########################################


set -x -e
echo "START TIME: $(date)"
echo python3 version = `python3 --version`

# SLURM stuff
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export TMPDIR=/scratch/hynek_kydlicek
mkdir -p $TMPDIR
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_DEVICE_MAX_CONNECTIONS="1"
export OMP_NUM_THREADS=11

# VLLM
export VLLM_WORKER_MULTIPROC_METHOD=spawn

module load cuda/12.1

echo go $COUNT_NODE
echo $HOSTNAMES

launch_args="accelerate\
    --model_args vllm,pretrained=inceptionai/jais-family-6p7b,dtype=bfloat16,max_model_length=4096,pairwise_tokenization=True,tensor_parallel_size=1,gpu_memory_utilisation=0.8 \
    --custom_task lighteval.tasks.multilingual.tasks \
    --tasks 'lighteval|exams_ara_mcf_native|5|1,lighteval|belebele_arb_Arab_mcf_native|5|1,lighteval|soqal_ara_mcf_native|5|1,lighteval|mmlu_ara_mcf_native|5|1,lighteval|alghafa_arc_ara_mcf_native:easy|5|1,lighteval|mlmm_hellaswag_ara_mcf_native|5|1,lighteval|alghafa_piqa_ara_mcf_native|5|1,lighteval|alghafa_race_ara_mcf_native|5|1,lighteval|alghafa_sciqa_ara_mcf_native|5|1,lighteval|xcodah_ara_mcf_native|5|1,lighteval|xcsqa_ara_mcf_native|5|1,lighteval|xnli2.0_ara_mcf_native|5|1,lighteval|xstory_cloze_ara_mcf_native|5|1' \
     \
    --max_samples '1000' \
    --dataset_loading_processes 1 \
    --save_details \
    --output_dir 's3://multilingual-evals/models_comparison_new_format'" 

# sleep $((RANDOM % 60))
srun -u bash -c "python3 -u -m lighteval ${launch_args}"
                