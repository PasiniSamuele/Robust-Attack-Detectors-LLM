#!/bin/bash

test_synth_results_root="experiments/task_detect_sqli_extended/template_create_function_readable/prompt_parameters_empty"
test_synth_results_root_sap="new_experiments_sap_sqli/task_detect_sqli_extended/template_create_function_readable/prompt_parameters_empty"

# Empty the runs.txt file
> runs_sqli.txt

# Find directories starting with "run_0" in test_synth_results_root and write them to runs.txt
find $test_synth_results_root -type d -name "run_0*" -printf "%p\n" >> runs_sqli.txt

# Find directories starting with "run_0" in test_synth_results_root_sap and write them to runs.txt
find $test_synth_results_root_sap -type d -name "run_0*" -printf "%p\n" >> runs_sqli.txt