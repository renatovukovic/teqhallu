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


def get_config(config_name):
	#return config based on string name
	if config_name not in globals():
		raise ValueError(f"Config name {config_name} not found")
	return globals()[config_name]



#sqlite pipeline from two models used in DiaHalu: Gemini-1.5-Pro and GPT-4 only on TOD
dialhalu_test_task_oriented_sqlite_pipeline_plus_direct_value_examples_gemini_15pro_config = hallucination_detection_config(
	model_name = "gemini1.5-pro",
	dataset = "DiaHalu",
	tasktype = "Task-oriented Style",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct_dialhalu",
	return_column_values=True,
)

dialhalu_test_task_oriented_sqlite_pipeline_plus_direct_value_examples_gpt4_config = hallucination_detection_config(
	model_name = "gpt-4-0613",
	dataset = "DiaHalu",
	tasktype = "Task-oriented Style",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct_dialhalu",
	return_column_values=True,
)

dialhalu_test_task_oriented_baseline_gemini_15pro_config = hallucination_detection_config(
	model_name = "gemini1.5-pro",
	dataset = "DiaHalu",
	tasktype = "Task-oriented Style",
	splits = ["test"],
	prompt_name="baseline_dialhalu"
)

dialhalu_test_task_oriented_baseline_gpt4_config = hallucination_detection_config(
	model_name = "gpt-4-0613",
	dataset = "DiaHalu",
	tasktype = "Task-oriented Style",
	splits = ["test"],
	prompt_name="baseline_dialhalu"
)

#### DiaHalu prediction configs ####

## model "gemini2.5-flash" ##

#baseline prediction gemini flash
dialhalu_test_reasoning_baseline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "DiaHalu",
	tasktype = "Reasoning",
	splits = ["test"],
	prompt_name="baseline_dialhalu"
)

dialhalu_test_world_knowledge_baseline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "DiaHalu",
	tasktype = "World Knowledge",
	splits = ["test"],
	prompt_name="baseline_dialhalu"
)

dialhalu_test_task_oriented_baseline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "DiaHalu",
	tasktype = "Task-oriented Style",
	splits = ["test"],
	prompt_name="baseline_dialhalu"
)

dialhalu_test_chit_chat_baseline_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "DiaHalu",
	tasktype = "Chit-Chat",
	splits = ["test"],
	prompt_name="baseline_dialhalu"
)

### with column value examples combined with direct prediction in the last step prompt
dialhalu_test_reasoning_sqlite_pipeline_plus_direct_value_examples_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "DiaHalu",
	tasktype = "Reasoning",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct_dialhalu",
	return_column_values=True,
)

dialhalu_test_world_knowledge_sqlite_pipeline_plus_direct_value_examples_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "DiaHalu",
	tasktype = "World Knowledge",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct_dialhalu",
	return_column_values=True,
)

dialhalu_test_task_oriented_sqlite_pipeline_plus_direct_value_examples_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "DiaHalu",
	tasktype = "Task-oriented Style",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct_dialhalu",
	return_column_values=True,
)

dialhalu_test_chit_chat_sqlite_pipeline_plus_direct_value_examples_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "DiaHalu",
	tasktype = "Chit-Chat",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct_dialhalu",
	return_column_values=True,
)


#also try inference on TOD portion of diahalu using a DB built on multiwoz
dialhalu_test_task_oriented_sqlite_pipeline_plus_direct_value_examples_with_mwoz_ontology_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "DiaHalu",
	tasktype = "Task-oriented Style",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct_dialhalu",
	return_column_values=True,
)

#try the base prompt from diahalu dataset

dialhalu_test_reasoning_base_from_diahalu_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "DiaHalu",
	tasktype = "Reasoning",
	splits = ["test"],
	prompt_name="base_from_diahalu"
)

dialhalu_test_world_knowledge_base_from_diahalu_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "DiaHalu",
	tasktype = "World Knowledge",
	splits = ["test"],
	prompt_name="base_from_diahalu"
)

dialhalu_test_task_oriented_base_from_diahalu_gemini_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "DiaHalu",
	tasktype = "Task-oriented Style",
	splits = ["test"],
	prompt_name="base_from_diahalu"
)

dialhalu_test_chit_chat_base_from_diahalu_flash_config = hallucination_detection_config(
	model_name = "gemini2.5-flash",
	dataset = "DiaHalu",
	tasktype = "Chit-Chat",
	splits = ["test"],
	prompt_name="base_from_diahalu"
)


dialhalu_test_task_oriented_base_from_diahalu_gemini_15pro_config = hallucination_detection_config(
	model_name = "gemini1.5-pro",
	dataset = "DiaHalu",
	tasktype = "Task-oriented Style",
	splits = ["test"],
	prompt_name="base_from_diahalu"
)

dialhalu_test_task_oriented_base_from_diahalu_gpt4_config = hallucination_detection_config(
	model_name = "gpt-4-0613",
	dataset = "DiaHalu",
	tasktype = "Task-oriented Style",
	splits = ["test"],
	prompt_name="base_from_diahalu"
)


## model "meta/llama-3.3-70b-instruct-maas" ##

#baseline prediction llama
dialhalu_test_reasoning_baseline_llama_config = hallucination_detection_config(
	model_name = "meta/llama-3.3-70b-instruct-maas",
	dataset = "DiaHalu",
	tasktype = "Reasoning",
	splits = ["test"],
	prompt_name="baseline_dialhalu"
)

dialhalu_test_world_knowledge_baseline_llama_config = hallucination_detection_config(
	model_name = "meta/llama-3.3-70b-instruct-maas",
	dataset = "DiaHalu",
	tasktype = "World Knowledge",
	splits = ["test"],
	prompt_name="baseline_dialhalu"
)

dialhalu_test_task_oriented_baseline_llama_config = hallucination_detection_config(
	model_name = "meta/llama-3.3-70b-instruct-maas",
	dataset = "DiaHalu",
	tasktype = "Task-oriented Style",
	splits = ["test"],
	prompt_name="baseline_dialhalu"
)

dialhalu_test_chit_chat_baseline_llama_config = hallucination_detection_config(
	model_name = "meta/llama-3.3-70b-instruct-maas",
	dataset = "DiaHalu",
	tasktype = "Chit-Chat",
	splits = ["test"],
	prompt_name="baseline_dialhalu"
)

### with column value examples combined with direct prediction in the last step prompt
dialhalu_test_reasoning_sqlite_pipeline_plus_direct_value_examples_llama_config = hallucination_detection_config(
	model_name = "meta/llama-3.3-70b-instruct-maas",
	dataset = "DiaHalu",
	tasktype = "Reasoning",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct_dialhalu",
	return_column_values=True,
)

dialhalu_test_world_knowledge_sqlite_pipeline_plus_direct_value_examples_llama_config = hallucination_detection_config(
	model_name = "meta/llama-3.3-70b-instruct-maas",
	dataset = "DiaHalu",
	tasktype = "World Knowledge",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct_dialhalu",
	return_column_values=True,
)

dialhalu_test_task_oriented_sqlite_pipeline_plus_direct_value_examples_llama_config = hallucination_detection_config(
	model_name = "meta/llama-3.3-70b-instruct-maas",
	dataset = "DiaHalu",
	tasktype = "Task-oriented Style",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct_dialhalu",
	return_column_values=True,
)

dialhalu_test_chit_chat_sqlite_pipeline_plus_direct_value_examples_llama_config = hallucination_detection_config(
	model_name = "meta/llama-3.3-70b-instruct-maas",
	dataset = "DiaHalu",
	tasktype = "Chit-Chat",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct_dialhalu",
	return_column_values=True,
)


## model "qwen/qwen3-235b-a22b-instruct-2507-maas" ##

#baseline prediction qwen
dialhalu_test_reasoning_baseline_qwen_config = hallucination_detection_config(
	model_name = "qwen/qwen3-235b-a22b-instruct-2507-maas",
	dataset = "DiaHalu",
	tasktype = "Reasoning",
	splits = ["test"],
	prompt_name="baseline_dialhalu"
)

dialhalu_test_world_knowledge_baseline_qwen_config = hallucination_detection_config(
	model_name = "qwen/qwen3-235b-a22b-instruct-2507-maas",
	dataset = "DiaHalu",
	tasktype = "World Knowledge",
	splits = ["test"],
	prompt_name="baseline_dialhalu"
)

dialhalu_test_task_oriented_baseline_qwen_config = hallucination_detection_config(
	model_name = "qwen/qwen3-235b-a22b-instruct-2507-maas",
	dataset = "DiaHalu",
	tasktype = "Task-oriented Style",
	splits = ["test"],
	prompt_name="baseline_dialhalu"
)

dialhalu_test_chit_chat_baseline_qwen_config = hallucination_detection_config(
	model_name = "qwen/qwen3-235b-a22b-instruct-2507-maas",
	dataset = "DiaHalu",
	tasktype = "Chit-Chat",
	splits = ["test"],
	prompt_name="baseline_dialhalu"
)

### with column value examples combined with direct prediction in the last step prompt
dialhalu_test_reasoning_sqlite_pipeline_plus_direct_value_examples_qwen_config = hallucination_detection_config(
	model_name = "qwen/qwen3-235b-a22b-instruct-2507-maas",
	dataset = "DiaHalu",
	tasktype = "Reasoning",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct_dialhalu",
	return_column_values=True,
)

dialhalu_test_world_knowledge_sqlite_pipeline_plus_direct_value_examples_qwen_config = hallucination_detection_config(
	model_name = "qwen/qwen3-235b-a22b-instruct-2507-maas",
	dataset = "DiaHalu",
	tasktype = "World Knowledge",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct_dialhalu",
	return_column_values=True,
)

dialhalu_test_task_oriented_sqlite_pipeline_plus_direct_value_examples_qwen_config = hallucination_detection_config(
	model_name = "qwen/qwen3-235b-a22b-instruct-2507-maas",
	dataset = "DiaHalu",
	tasktype = "Task-oriented Style",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct_dialhalu",
	return_column_values=True,
)

dialhalu_test_chit_chat_sqlite_pipeline_plus_direct_value_examples_qwen_config = hallucination_detection_config(
	model_name = "qwen/qwen3-235b-a22b-instruct-2507-maas",
	dataset = "DiaHalu",
	tasktype = "Chit-Chat",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct_dialhalu",
	return_column_values=True,
)


## model "openai/gpt-oss-20b-maas" ##

#baseline prediction gpt-oss
dialhalu_test_reasoning_baseline_gpt_oss_config = hallucination_detection_config(
	model_name = "openai/gpt-oss-20b-maas",
	dataset = "DiaHalu",
	tasktype = "Reasoning",
	splits = ["test"],
	prompt_name="baseline_dialhalu"
)

dialhalu_test_world_knowledge_baseline_gpt_oss_config = hallucination_detection_config(
	model_name = "openai/gpt-oss-20b-maas",
	dataset = "DiaHalu",
	tasktype = "World Knowledge",
	splits = ["test"],
	prompt_name="baseline_dialhalu"
)

dialhalu_test_task_oriented_baseline_gpt_oss_config = hallucination_detection_config(
	model_name = "openai/gpt-oss-20b-maas",
	dataset = "DiaHalu",
	tasktype = "Task-oriented Style",
	splits = ["test"],
	prompt_name="baseline_dialhalu"
)

dialhalu_test_chit_chat_baseline_gpt_oss_config = hallucination_detection_config(
	model_name = "openai/gpt-oss-20b-maas",
	dataset = "DiaHalu",
	tasktype = "Chit-Chat",
	splits = ["test"],
	prompt_name="baseline_dialhalu"
)

### with column value examples combined with direct prediction in the last step prompt
dialhalu_test_reasoning_sqlite_pipeline_plus_direct_value_examples_gpt_oss_config = hallucination_detection_config(
	model_name = "openai/gpt-oss-20b-maas",
	dataset = "DiaHalu",
	tasktype = "Reasoning",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct_dialhalu",
	return_column_values=True,
)

dialhalu_test_world_knowledge_sqlite_pipeline_plus_direct_value_examples_gpt_oss_config = hallucination_detection_config(
	model_name = "openai/gpt-oss-20b-maas",
	dataset = "DiaHalu",
	tasktype = "World Knowledge",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct_dialhalu",
	return_column_values=True,
)

dialhalu_test_task_oriented_sqlite_pipeline_plus_direct_value_examples_gpt_oss_config = hallucination_detection_config(
	model_name = "openai/gpt-oss-20b-maas",
	dataset = "DiaHalu",
	tasktype = "Task-oriented Style",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct_dialhalu",
	return_column_values=True,
)

dialhalu_test_chit_chat_sqlite_pipeline_plus_direct_value_examples_gpt_oss_config = hallucination_detection_config(
	model_name = "openai/gpt-oss-20b-maas",
	dataset = "DiaHalu",
	tasktype = "Chit-Chat",
	splits = ["test"],
	prompt_name="sqlite_hallucination_plus_direct_dialhalu",
	return_column_values=True,
)
