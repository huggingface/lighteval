#!/bin/bash
# Test script for mAIME tasks

cd /Users/maria/Repos/lighteval

echo "Testing mAIME tasks with distilgpt2..."
echo ""

# Run single-shot evaluation (greedy decoding)
# Note: pass@k evaluation with multiple samples requires the transformers backend
# to automatically enable do_sample=True when temperature > 0, but this is not
# currently working in lighteval without modifying core code.
# 
# For pass@k evaluation, you would need to either:
# 1. Use the vllm backend instead of accelerate
# 2. Use an API endpoint backend (openai, etc.)
# 3. Modify lighteval's transformers_model.py to set do_sample=True when temp>0
#
# For now, running single-shot evaluation with greedy decoding:
# Using maime25:da which includes both pass@1 (k=1,n=1) and avg@1 (n=1) metrics
lighteval accelerate \
    "model_name=distilgpt2,batch_size=1" \
    "maime25:da" \
    --max-samples 5 \
    --output-dir "./results_maime_single" \
    --load-tasks-multilingual \
    --save-details

echo ""
echo "Evaluation complete! Check ./results_maime_single/ for results"
echo ""
echo "Metrics computed:"
echo "  - pass@k:k=1&n=1 (standard pass@1 accuracy)"
echo "  - avg@n:n=1 (average score with 1 sample per problem)"
echo ""
echo "NOTE: maime25_avg:da (n=64) and maime25_gpassk:da (n=48) require temperature>0"
echo "      which needs lighteval core modification for transformers backend."
