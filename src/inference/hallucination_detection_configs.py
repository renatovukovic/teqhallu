# coding=utf-8
#
# Copyright 2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Renato Vukovic (renato.vukovic@hhu.de)
#
# This code was generated with the help of AI writing assistants
# including GitHub Copilot, ChatGPT, Bing Chat.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from inference.hallucination_detection_config_class import hallucination_detection_config
from inference.hallucination_detection_seed_configs import *
from inference.hallucination_detection_configs_dialhalu import *


def get_config(config_name):
	#return config based on string name
	if config_name not in globals():
		raise ValueError(f"Config name {config_name} not found")
	return globals()[config_name]


#### RAGTruth prediction configs ####

#baseline prediction gemini flash
ragtruth_test_qa_baseline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="baseline"
)

ragtruth_test_summary_baseline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="baseline"
)

ragtruth_test_data2text_baseline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="baseline"
)

#baseline less strict prompt
ragtruth_test_qa_baseline_less_strict_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="baseline_not_too_strict"
)

ragtruth_test_summary_baseline_less_strict_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="baseline_not_too_strict"
)

ragtruth_test_data2text_baseline_less_strict_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="baseline_not_too_strict"
)

###sqlite hallucination prediction prompt
ragtruth_test_qa_sqlite_pipeline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="sqlite_hallucination"
)

ragtruth_test_summary_sqlite_pipeline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="sqlite_hallucination"
)

ragtruth_test_data2text_sqlite_pipeline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="sqlite_hallucination"
)

### with column value examples
ragtruth_test_qa_sqlite_pipeline_value_examples_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="sqlite_hallucination",
	return_column_values=True,
)

ragtruth_test_summary_sqlite_pipeline_value_examples_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="sqlite_hallucination",
	return_column_values=True,
)

ragtruth_test_data2text_sqlite_pipeline_value_examples_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="sqlite_hallucination",
	return_column_values=True,
)

### with column value examples combined with direct prediction in the last step prompt
ragtruth_test_qa_sqlite_pipeline_plus_direct_value_examples_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct",
	return_column_values=True,
)

ragtruth_test_summary_sqlite_pipeline_plus_direct_value_examples_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct",
	return_column_values=True,
)

ragtruth_test_data2text_sqlite_pipeline_plus_direct_value_examples_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct",
	return_column_values=True,
)

#select for IDs using all column values for existing ontology for ID set comparison approach
ragtruth_test_qa_sqlite_pipeline_id_select_queries_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="sqlite_id_select",
	return_column_values=True,
)

ragtruth_test_summary_sqlite_pipeline_id_select_queries_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="sqlite_id_select",
	return_column_values=True,
)

ragtruth_test_data2text_sqlite_pipeline_id_select_queries_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="sqlite_id_select",
	return_column_values=True,
)


###two step sql hallucination prediction prompt
ragtruth_test_qa_twostep_sql_pipeline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="two_step_sql"
)

ragtruth_test_summary_twostep_sql_pipeline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="two_step_sql"
)

ragtruth_test_data2text_twostep_sql_pipeline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="two_step_sql"
)

###two step subset hallucination prediction prompt
ragtruth_test_qa_twostep_subset_pipeline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="two_step_subset"
)

ragtruth_test_summary_twostep_subset_pipeline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="two_step_subset"
)

ragtruth_test_data2text_twostep_subset_pipeline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="two_step_subset"
)


### multistep atomic claim hallucination prediction prompt
ragtruth_test_qa_multistep_atomic_claim_pipeline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="multistep_atomic_claim"
)

ragtruth_test_summary_multistep_atomic_claim_pipeline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="multistep_atomic_claim"
)

ragtruth_test_data2text_multistep_atomic_claim_pipeline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="multistep_atomic_claim"
)


### json key-value hallucination prediction prompt
ragtruth_test_qa_json_keyvalue_pipeline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="json_keyvalue"
)

ragtruth_test_summary_json_keyvalue_pipeline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="json_keyvalue"
)

ragtruth_test_data2text_json_keyvalue_pipeline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="json_keyvalue"
)


#model "meta/llama-3.3-70b-instruct-maas"

#baseline prediction llama
ragtruth_test_qa_baseline_llama_config = hallucination_detection_config(
	model_name = "meta/llama-3.3-70b-instruct-maas",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="baseline"
)

ragtruth_test_summary_baseline_llama_config = hallucination_detection_config(
	model_name = "meta/llama-3.3-70b-instruct-maas",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="baseline"
)

ragtruth_test_data2text_baseline_llama_config = hallucination_detection_config(
	model_name = "meta/llama-3.3-70b-instruct-maas",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="baseline"
)

#baseline less strict prompt llama
ragtruth_test_qa_baseline_less_strict_llama_config = hallucination_detection_config(
	model_name = "meta/llama-3.3-70b-instruct-maas",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="baseline_not_too_strict"
)

ragtruth_test_summary_baseline_less_strict_llama_config = hallucination_detection_config(
	model_name = "meta/llama-3.3-70b-instruct-maas",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="baseline_not_too_strict"
)

ragtruth_test_data2text_baseline_less_strict_llama_config = hallucination_detection_config(
	model_name = "meta/llama-3.3-70b-instruct-maas",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="baseline_not_too_strict"
)

### with column value examples
ragtruth_test_qa_sqlite_pipeline_value_examples_llama_config = hallucination_detection_config(
	model_name = "meta/llama-3.3-70b-instruct-maas",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="sqlite_hallucination",
	return_column_values=True,
)

ragtruth_test_summary_sqlite_pipeline_value_examples_llama_config = hallucination_detection_config(
	model_name = "meta/llama-3.3-70b-instruct-maas",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="sqlite_hallucination",
	return_column_values=True,
)

ragtruth_test_data2text_sqlite_pipeline_value_examples_llama_config = hallucination_detection_config(
	model_name = "meta/llama-3.3-70b-instruct-maas",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="sqlite_hallucination",
	return_column_values=True,
)

### with column value examples combined with direct prediction in the last step prompt
ragtruth_test_qa_sqlite_pipeline_plus_direct_value_examples_llama_config = hallucination_detection_config(
	model_name = "meta/llama-3.3-70b-instruct-maas",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct",
	return_column_values=True,
)

ragtruth_test_summary_sqlite_pipeline_plus_direct_value_examples_llama_config = hallucination_detection_config(
	model_name = "meta/llama-3.3-70b-instruct-maas",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct",
	return_column_values=True,
)

ragtruth_test_data2text_sqlite_pipeline_plus_direct_value_examples_llama_config = hallucination_detection_config(
	model_name = "meta/llama-3.3-70b-instruct-maas",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct",
	return_column_values=True,
)


#model "qwen/qwen3-235b-a22b-instruct-2507-maas"

#baseline prediction qwen
ragtruth_test_qa_baseline_qwen_config = hallucination_detection_config(
	model_name = "qwen/qwen3-235b-a22b-instruct-2507-maas",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="baseline"
)

ragtruth_test_summary_baseline_qwen_config = hallucination_detection_config(
	model_name = "qwen/qwen3-235b-a22b-instruct-2507-maas",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="baseline"
)

ragtruth_test_data2text_baseline_qwen_config = hallucination_detection_config(
	model_name = "qwen/qwen3-235b-a22b-instruct-2507-maas",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="baseline"
)

#baseline less strict prompt qwen
ragtruth_test_qa_baseline_less_strict_qwen_config = hallucination_detection_config(
	model_name = "qwen/qwen3-235b-a22b-instruct-2507-maas",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="baseline_not_too_strict"
)

ragtruth_test_summary_baseline_less_strict_qwen_config = hallucination_detection_config(
	model_name = "qwen/qwen3-235b-a22b-instruct-2507-maas",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="baseline_not_too_strict"
)

ragtruth_test_data2text_baseline_less_strict_qwen_config = hallucination_detection_config(
	model_name = "qwen/qwen3-235b-a22b-instruct-2507-maas",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="baseline_not_too_strict"
)

### with column value examples
ragtruth_test_qa_sqlite_pipeline_value_examples_qwen_config = hallucination_detection_config(
	model_name = "qwen/qwen3-235b-a22b-instruct-2507-maas",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="sqlite_hallucination",
	return_column_values=True,
)

ragtruth_test_summary_sqlite_pipeline_value_examples_qwen_config = hallucination_detection_config(
	model_name = "qwen/qwen3-235b-a22b-instruct-2507-maas",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="sqlite_hallucination",
	return_column_values=True,
)

ragtruth_test_data2text_sqlite_pipeline_value_examples_qwen_config = hallucination_detection_config(
	model_name = "qwen/qwen3-235b-a22b-instruct-2507-maas",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="sqlite_hallucination",
	return_column_values=True,
)

### with column value examples combined with direct prediction in the last step prompt
ragtruth_test_qa_sqlite_pipeline_plus_direct_value_examples_qwen_config = hallucination_detection_config(
	model_name = "qwen/qwen3-235b-a22b-instruct-2507-maas",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct",
	return_column_values=True,
)

ragtruth_test_summary_sqlite_pipeline_plus_direct_value_examples_qwen_config = hallucination_detection_config(
	model_name = "qwen/qwen3-235b-a22b-instruct-2507-maas",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct",
	return_column_values=True,
)

ragtruth_test_data2text_sqlite_pipeline_plus_direct_value_examples_qwen_config = hallucination_detection_config(
	model_name = "qwen/qwen3-235b-a22b-instruct-2507-maas",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct",
	return_column_values=True,
)


#model "openai/gpt-oss-20b-maas"

#baseline prediction gpt-oss
ragtruth_test_qa_baseline_gpt_oss_config = hallucination_detection_config(
	model_name = "openai/gpt-oss-20b-maas",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="baseline"
)

ragtruth_test_summary_baseline_gpt_oss_config = hallucination_detection_config(
	model_name = "openai/gpt-oss-20b-maas",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="baseline"
)

ragtruth_test_data2text_baseline_gpt_oss_config = hallucination_detection_config(
	model_name = "openai/gpt-oss-20b-maas",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="baseline"
)

#baseline less strict prompt gpt-oss
ragtruth_test_qa_baseline_less_strict_gpt_oss_config = hallucination_detection_config(
	model_name = "openai/gpt-oss-20b-maas",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="baseline_not_too_strict"
)

ragtruth_test_summary_baseline_less_strict_gpt_oss_config = hallucination_detection_config(
	model_name = "openai/gpt-oss-20b-maas",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="baseline_not_too_strict"
)

ragtruth_test_data2text_baseline_less_strict_gpt_oss_config = hallucination_detection_config(
	model_name = "openai/gpt-oss-20b-maas",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="baseline_not_too_strict"
)

### with column value examples
ragtruth_test_qa_sqlite_pipeline_value_examples_gpt_oss_config = hallucination_detection_config(
	model_name = "openai/gpt-oss-20b-maas",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="sqlite_hallucination",
	return_column_values=True,
)

ragtruth_test_summary_sqlite_pipeline_value_examples_gpt_oss_config = hallucination_detection_config(
	model_name = "openai/gpt-oss-20b-maas",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="sqlite_hallucination",
	return_column_values=True,
)

ragtruth_test_data2text_sqlite_pipeline_value_examples_gpt_oss_config = hallucination_detection_config(
	model_name = "openai/gpt-oss-20b-maas",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="sqlite_hallucination",
	return_column_values=True,
)

### with column value examples combined with direct prediction in the last step prompt
ragtruth_test_qa_sqlite_pipeline_plus_direct_value_examples_gpt_oss_config = hallucination_detection_config(
	model_name = "openai/gpt-oss-20b-maas",
	dataset = "RAGTruth",
	tasktype = "QA",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct",
	return_column_values=True,
)

ragtruth_test_summary_sqlite_pipeline_plus_direct_value_examples_gpt_oss_config = hallucination_detection_config(
	model_name = "openai/gpt-oss-20b-maas",
	dataset = "RAGTruth",
	tasktype = "Summary",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct",
	return_column_values=True,
)

ragtruth_test_data2text_sqlite_pipeline_plus_direct_value_examples_gpt_oss_config = hallucination_detection_config(
	model_name = "openai/gpt-oss-20b-maas",
	dataset = "RAGTruth",
	tasktype = "Data2txt",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct",
	return_column_values=True,
)


