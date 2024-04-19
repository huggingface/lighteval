# Nanotron tests guide
## How it works:
First select some tasks and then use the model to generate reference scores and save them in reference_task_scores_nanotron.py file, it has been done, but if you want to add a new task, you need to re-run it.

After that, each time a test need to be conducted, the evaluation will be run and the results are compared to the previous reference score.

## To run nanotron test:
```
pytest tests/test_main_nanotron.py -sv
```

## Choose your own tasks for evaluation:
Modify the **tasks.tasks** in config file(lighteval/tests/config/lighteval_config_override_custom.yaml) to set the tasks.
Example:
```
tasks:
   custom_tasks: null
   dataset_loading_processes: 1
   max_samples: 10
   multichoice_continuations_start_space: null
   no_multichoice_continuations_start_space: null
   num_fewshot_seeds: null
   tasks: lighteval|anli:r1|0|0,lighteval|blimp:adjunct_island|0|0,...
```

## Randomized results
Please make sure to set **for_inference** to true. This will load model with a fixed output layer norm implementation. It's set to true by default for training
```
model:
  ddp_bucket_cap_mb: 25
  dtype: float64
  init_method:
    std: 0.02
  make_vocab_size_divisible_by: 1
  model_config:
    bos_token_id: 1
    eos_token_id: 2
    hidden_act: silu
    hidden_size: 512
    initializer_range: 0.02
    intermediate_size: 2048
    is_llama_config: true
    max_position_embeddings: 2048
    num_attention_heads: 16
    num_hidden_layers: 16
    num_key_value_heads: 16
    pad_token_id: null
    pretraining_tp: 1
    rms_norm_eps: 1.0e-05
    rope_scaling: null
    tie_word_embeddings: true
    use_cache: true
    vocab_size: 50272
    for_inference: true
```
