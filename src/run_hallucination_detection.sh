#!/bin/bash

# Run from this script's directory so the module imports work from any cwd.
cd "$(dirname "$0")" || exit 1

# These names are defined in inference/hallucination_detection_configs.py.
config_names=(
	"ragtruth_test_qa_baseline_gemini_flash_config"
	"ragtruth_test_summary_baseline_gemini_flash_config"
	"ragtruth_test_data2text_baseline_gemini_flash_config"
	"ragtruth_test_qa_sqlite_pipeline_plus_direct_value_examples_gemini_flash_config"
	"ragtruth_test_summary_sqlite_pipeline_plus_direct_value_examples_gemini_flash_config"
	"ragtruth_test_data2text_sqlite_pipeline_plus_direct_value_examples_gemini_flash_config"
)

for config_name in "${config_names[@]}"
do
	uv run python -u -m inference.hallucination_detection_inference \
		--config_name "$config_name"
done