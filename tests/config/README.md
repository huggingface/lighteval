# Nanotron tests guide
## How it works: 
First select some tasks and then use the model to generate reference scores and save them in reference_task_scores_nanotron.py file, it has been done, but if you want to add a new task, you need to re-run it.

After that, each time a test need to be conducted, the evaluation will be run and the results are compared to the previous reference score.

## To run nanotron test:   
```
pytest tests/test_main_nanotron.py -sv
```

## Choose your own tasks for evaluation:
Modify the tasks.tasks in config file(lighteval/tests/config/lighteval_config_override_custom.yaml) to set the tasks.   
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
