#!/bin/bash
# Test script for mAIME tasks using vLLM backend

# What I did: 

# first time running:
# Install lighteval dependencies that aren't in the container
# pip install --target=/shared_silo/scratch/mbarrett/python_packages \  colorlog pytablewriter typer termcolor rich aenum nltk scikit-learn \  sacrebleu rouge_score sentencepiece protobuf pycountry fsspec httpx \  latex2sympy2_extended langcodes inspect-ai GitPython pydantic hf-xet

# interactive node
#srun --partition=amd-tw-verification --time=1:00:00 --gpus-per-node=1 --pty \
#  apptainer exec --rocm \
#  -B /shared_silo/scratch:/shared_silo/scratch:rw \
#  -B /usr/share/libdrm:/usr/share/libdrm:ro \
#  -B /dev/kfd:/dev/kfd \
#  -B /dev/dri:/dev/dri \
#  /shared_silo/scratch/containers/rocm_vllm_rocm7.0.0_vllm_0.11.2_20251210.sif \
#  bash

# Unset the conflicting variable
unset ROCR_VISIBLE_DEVICES

# Set up environment to find lighteval
export PYTHONPATH="/home/mbarrett@amd.com/lighteval/src:/shared_silo/scratch/mbarrett/python_packages:$PYTHONPATH"


export HSA_OVERRIDE_GFX_VERSION=9.4.2  # For MI300/MI325X GPUs

echo ""

# Run evaluation using accelerate backend  
# Umaime25:da which includes both pass@1 (k=1,n=1) and avg@1 (n=1) metrics
# maime25_avg does not work with accelerate backend because the temperature is set to 0 which does not work for sampling 


#python -m lighteval accelerate \
#    "model_name=distilgpt2,batch_size=1,generation_parameters={temperature:0.8}" \
#    "maime25_avg:da" \
#    --max-samples 2 \
#    --output-dir "./results_maime_single" \
#    --custom-tasks "src/lighteval/tasks/multilingual/tasks/maime.py" \
#    --save-details

#python -m lighteval accelerate \
#    "model_name=distilgpt2,batch_size=1,generation_parameters={temperature:0.8}" \
#    "maime25_avg:da" \
#    --max-samples 2 \
#    --output-dir "./results_maime_single" \
#    --custom-tasks "src/lighteval/tasks/multilingual/tasks/maime.py" \
#    --save-details

python -m lighteval vllm \
    "model_name=distilgpt2,generation_parameters={temperature:0.8}" \
    "maime25_avg:da" \
    --max-samples 5 \
    --output-dir "./results_maime_avg64" \
    --custom-tasks "/home/mbarrett@amd.com/lighteval/src/lighteval/tasks/multilingual/tasks/maime.py" \
    --save-details

echo ""
echo "Evaluation complete! Check ./results_maime_single/ for results"
echo ""
echo "Metrics computed:"
echo "  - pass@k:k=1&n=1 (standard pass@1 accuracy)"
echo "  - avg@n:n=1 (average score with 1 sample per problem)"
echo ""
echo "NOTE: For pass@k with multiple samples (n>1), use maime25_avg:da or maime25_gpassk:da"
echo "      with temperature>0. vLLM backend supports sampling natively."
