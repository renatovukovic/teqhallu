#!/bin/bash

# Run from this script's directory so the module imports work from any cwd.
cd "$(dirname "$0")" || exit 1

# These names are defined in inference/hallucination_detection_configs_dialhalu.py.
config_names=(
	"dialhalu_test_reasoning_baseline_gemini_flash_config"
	"dialhalu_test_world_knowledge_baseline_gemini_flash_config"
	"dialhalu_test_task_oriented_baseline_gemini_flash_config"
	"dialhalu_test_chit_chat_baseline_gemini_flash_config"
	"dialhalu_test_reasoning_sqlite_pipeline_plus_direct_value_examples_gemini_flash_config"
	"dialhalu_test_world_knowledge_sqlite_pipeline_plus_direct_value_examples_gemini_flash_config"
	"dialhalu_test_task_oriented_sqlite_pipeline_plus_direct_value_examples_gemini_flash_config"
	"dialhalu_test_chit_chat_sqlite_pipeline_plus_direct_value_examples_gemini_flash_config"
)

for config_name in "${config_names[@]}"
do
	uv run python -u -m evaluation.evaluate_dialhalu \
		--config_name "$config_name"
done
