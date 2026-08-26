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

import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import os
from langchain_google_vertexai import ChatVertexAI, HarmCategory, HarmBlockThreshold
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
import langchain_core
import torch
import pprint
import openai
from openai import OpenAI
from dotenv import load_dotenv
from google.auth import default, transport
from google.auth.transport.requests import Request

torch.serialization.add_safe_globals([langchain_core.messages.human.HumanMessage])
torch.serialization.add_safe_globals([langchain_core.messages.ai.AIMessage])


def require_environment_variable(name: str) -> str:
	"""Return a required environment variable or raise a helpful error."""
	value = os.getenv(name)
	if not value:
		raise RuntimeError(f"Required environment variable is not set: {name}")
	return value


#local imports
from data_processing.ragtruth_processing import load_jsonl, combine_sources_and_responses, group_by_task_type
from inference.hallucination_detection_configs import get_config
from inference.text_to_sql_hallucination_detection import execute_text_to_sql_prompt_dict, load_prompt_steps, parse_twostep_prompt_file, execute_two_step_prompt_dict, load_id_select_prompt_steps, execute_text_to_sql_id_select_prompt_dict, execute_text_to_sql_prompt_dict_with_openai_api, completion_with_retries
from inference.multistep_structured_hallucination_detection import (
	load_multistep_prompt_steps,
	execute_atomic_claim_prompt_dict,
	execute_json_keyvalue_prompt_dict,
	execute_atomic_claim_prompt_dict_with_openai_api,
	execute_json_keyvalue_prompt_dict_with_openai_api,
)





def setup_logger(name: str = "logger") -> logging.Logger:
    """
    Set up and return a logger that logs to both console and a file
    inside a 'logs' directory.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)

        # File handler
        log_filename = os.path.join("logs", f"{name}.log")
        fh = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--config_name",
		type=str,
		default="baseline_QA_inference",
		help="The name of the hallucination detection config"
	)
        

	args = parser.parse_args()
	load_dotenv(Path(".env"))

	config = get_config(args.config_name)
        
	logger = setup_logger(f"{args.config_name}_inference_logger")
        
	dataset = []
    
	if config.dataset == "RAGTruth":
		logger.info(f"Load RAGTruth Data, {config.tasktype} task")
		ragtruth_dataset_path = "../Data/ragtruth_data/"
		response_path = Path(f"{ragtruth_dataset_path}/response.jsonl")
		source_path = Path(f"{ragtruth_dataset_path}/source_info.jsonl")
		response_list = load_jsonl(response_path)
		source_list = load_jsonl(source_path)
                
		responses_with_source_info = combine_sources_and_responses(response_list, source_list)
                
		dataset = [data_entry for data_entry in group_by_task_type(responses_with_source_info)[config.tasktype] if data_entry["split"] in config.splits]
	
	elif config.dataset == "DiaHalu":
		logger.info(f"Load DiaHalu Data, {config.tasktype} task")
		diahalu_path = Path("../Data/DiaHalu-main/DiaHalu_Bench.jsonl")
		diahalu_list = load_jsonl(diahalu_path)
		dataset = [entry for entry in diahalu_list if entry["domain"] == config.tasktype]
		
                
	
	if "gemini" in config.model_name:
		vertex_project_id = require_environment_variable("VERTEX_AI_PROJECT_ID")
		if config.seed:
			llm = ChatVertexAI(
				model=config.model_name,
				temperature=1.0,           # sample
				max_tokens=None,           # unlimited tokens (or specify a limit)
				max_retries=6,             # retry on transient errors
				stop=None,                 # optional stop sequences
				model_kwargs={"seed": config.seed},   # reproducible randomness
				project=vertex_project_id,
			)
		else:
			llm = ChatVertexAI(
				model=config.model_name,
				temperature=0,             # deterministic responses
				max_tokens=None,           # unlimited tokens (or specify a limit)
				max_retries=6,             # retry on transient errors
				stop=None,                 # optional stop sequences
				project=vertex_project_id,
			)

	elif "gpt-4" in config.model_name:
		client = OpenAI(
			organization=require_environment_variable("OPENAI_ORGANIZATION"),
			project=require_environment_variable("OPENAI_PROJECT_ID"),
			api_key=require_environment_variable("OPENAI_PROJECT_API_KEY"),
		)
		MODEL_ID = config.model_name
		logger.info("Use OpenAI GPT Model")

	else: # use open model via OpenAI API

		# --- 1. Configuration ---
		project_id = require_environment_variable("VERTEX_AI_PROJECT_ID")
		if "llama" in args.config_name:
			#LOCATION = "us-east5"
			LOCATION = "us-central1"            # Replace with the region where the model is available
		elif "qwen" in args.config_name:
			LOCATION = "us-south1"
		else:
			LOCATION = "us-central1"
		# This is the MaaS Model ID
		MODEL_ID = config.model_name
		# You must first enable this model in Vertex AI Model Garden and accept its EULA.

		# --- 2. Authentication and Endpoint Setup ---
		# Obtain Application Default Credentials (ADC) token
		credentials, _ = default()
		auth_request = transport.requests.Request()
		credentials.refresh(auth_request)
		gcp_token = credentials.token

		# Construct the Vertex AI MaaS endpoint URL for the OpenAI library
		vertex_ai_endpoint_url = (
			f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/"
			f"projects/{project_id}/locations/{LOCATION}/endpoints/openapi"
		)

		# Initialize the OpenAI client pointing to the Vertex AI MaaS endpoint
		client = openai.OpenAI(
			base_url=vertex_ai_endpoint_url,
			api_key=gcp_token,  # Use the GCP token as the API key
		)

	#load prompt
	prompt_path = Path(f"prompts/{config.prompt_name}_prompt.txt")
	prompt_text = prompt_path.read_text(encoding="utf-8")
	if "sqlite_hallucination" in config.prompt_name:
		prompt_dict = load_prompt_steps(prompt_path)
		db_directory = Path("sql_interpretation/databases")
		db_directory.mkdir(parents=True, exist_ok=True)
		db_name = f"{db_directory}/{args.config_name}_database.db"

		logger.info(f"Prompt Dictionary:\n{pprint.pformat(prompt_dict)}")

	elif config.prompt_name == "sqlite_id_select":
		prompt_dict = load_id_select_prompt_steps(prompt_path)
		db_directory = Path("sql_interpretation/databases")
		db_name = f"{db_directory}/{args.config_name.replace('id_select_queries', 'value_examples')}_database.db"

		logger.info(f"Prompt Dictionary:\n{pprint.pformat(prompt_dict)}")

	elif "two_step" in config.prompt_name:
		prompt_dict = parse_twostep_prompt_file(str(prompt_path))
		logger.info(f"Prompt Dictionary:\n{pprint.pformat(prompt_dict)}")

	elif "multistep_atomic_claim" in config.prompt_name or "json_keyvalue" in config.prompt_name:
		prompt_dict = load_multistep_prompt_steps(str(prompt_path))
		logger.info(f"Prompt Dictionary:\n{pprint.pformat(prompt_dict)}")

		
        
	results_directory = Path("inference/results")
	results_directory.mkdir(parents=True, exist_ok=True)
	result_filename = f"{results_directory}/{args.config_name}_responses.pt"
		
	#check whether a (unfinished) dictionary already exists, in case of exception during OpenAI query
	if Path(result_filename).is_file():
		hallucination_detection_prediction_dict = torch.load(result_filename, weights_only=False)
		logger.info("Loaded responses from file")
	else:
		#initialise the dialogue id, InstructGPT response dictionary
		hallucination_detection_prediction_dict = {}


	if "plus_direct" in config.prompt_name: #load the direct prediction responses to add them as additional input
		if config.dataset == "DiaHalu":
			direct_prediction_filename = f"{results_directory}/{args.config_name.replace('sqlite_pipeline_plus_direct_value_examples', 'baseline')}_responses.pt"
			direct_prediction_filename = direct_prediction_filename.replace("_with_mwoz_ontology", "")
			direct_prediction_filename = direct_prediction_filename.replace("gemini_15pro", "gemini_flash")
			direct_prediction_filename = direct_prediction_filename.replace("gpt4", "gemini_flash")
		else:
			direct_prediction_filename = f"{results_directory}/{args.config_name.replace('sqlite_pipeline_plus_direct_value_examples', 'baseline_less_strict')}_responses.pt"
		direct_prediction_dict = torch.load(direct_prediction_filename)



	logger.info("Start inference")

	counter = 0

	for datapoint in tqdm(dataset):
		if config.dataset == "RAGTruth":
			response_id = f"{datapoint['split']}-{datapoint['response_id']}"
			source_info = datapoint["source_info"]
			response = datapoint["response"]
		elif config.dataset == "DiaHalu":
			response_id = f"DiaHalu-{datapoint['ID']}"
			source_info = "" # DiaHalu is dialogue-based, prompt uses context from dialogue
			response = datapoint["text"]
		else:
			raise ValueError(f"Unknown dataset {config.dataset}")

		if response_id in hallucination_detection_prediction_dict:
			continue

		if config.prompt_name == "sqlite_hallucination":
			try:
				if "gemini" in config.model_name:
					pred_response, conversation_log, messages, avg_logprobs = execute_text_to_sql_prompt_dict(
						prompt_dict, 
						source_info, 
						response, 
						llm, 
						db_name, 
						return_column_values=config.return_column_values,
						)
				else:
					pred_response, conversation_log, messages, avg_logprobs = execute_text_to_sql_prompt_dict_with_openai_api(
						prompt_dict,
						source_info,
						response,
						client,
						MODEL_ID,
						logger,
						db_name,
						return_column_values=config.return_column_values,
					)
			except Exception as e:
				torch.save(hallucination_detection_prediction_dict, result_filename)
				logger.info(f"Responses saved under {result_filename} after {counter} dialogues because of Exception.")
				raise(e)

			hallucination_detection_prediction_dict[response_id] = {
				"response": pred_response, 
				"conversation_log": conversation_log, 
				"messages": messages, 
				"avg_logprobs": avg_logprobs,
				"datapoint": datapoint
			}

		elif "sqlite_hallucination_plus_direct" in config.prompt_name:
			try:
				if "gemini" in config.model_name:
					pred_response, conversation_log, messages, avg_logprobs = execute_text_to_sql_prompt_dict(
						prompt_dict, 
						source_info, 
						response, 
						llm, 
						db_name, 
						return_column_values=config.return_column_values,
						direct_prediction=direct_prediction_dict[response_id]["response"]
						)
				else:
					pred_response, conversation_log, messages, avg_logprobs = execute_text_to_sql_prompt_dict_with_openai_api(
						prompt_dict,
						source_info,
						response,
						client,
						MODEL_ID,
						logger,
						db_name,
						return_column_values=config.return_column_values,
						direct_prediction=direct_prediction_dict[response_id]["response"]
					)
			except Exception as e:
				torch.save(hallucination_detection_prediction_dict, result_filename)
				logger.info(f"Responses saved under {result_filename} after {counter} dialogues because of Exception.")
				raise(e)

			hallucination_detection_prediction_dict[response_id] = {
				"response": pred_response, 
				"conversation_log": conversation_log, 
				"messages": messages, 
				"avg_logprobs": avg_logprobs,
				"datapoint": datapoint
			}


		elif config.prompt_name == "sqlite_id_select":
			if "gemini" not in config.model_name:
				raise NotImplementedError("OpenAI models are not yet supported for sqlite_id_select prompt.")
			try:
				pred_response, conversation_log, messages, avg_logprobs = execute_text_to_sql_id_select_prompt_dict(
					prompt_dict, 
					source_info, 
					response, 
					llm, 
					db_name, 
					)
			except Exception as e:
				torch.save(hallucination_detection_prediction_dict, result_filename)
				logger.info(f"Responses saved under {result_filename} after {counter} dialogues because of Exception.")
				raise(e)

			hallucination_detection_prediction_dict[response_id] = {
				"response": pred_response, 
				"conversation_log": conversation_log, 
				"messages": messages, 
				"avg_logprobs": avg_logprobs,
				"datapoint": datapoint
			}


		elif "two_step" in config.prompt_name:
			if "gemini" not in config.model_name:
				raise NotImplementedError("OpenAI models are not yet supported for two_step prompts.")
			try:
				pred_response, conversation_log, messages, avg_logprobs = execute_two_step_prompt_dict(prompt_dict, source_info, response, llm)
			except Exception as e:
				torch.save(hallucination_detection_prediction_dict, result_filename)
				logger.info(f"Responses saved under {result_filename} after {counter} dialogues because of Exception.")
				raise(e)

			hallucination_detection_prediction_dict[response_id] = {
				"response": pred_response, 
				"conversation_log": conversation_log, 
				"messages": messages, 
				"avg_logprobs": avg_logprobs,
				"datapoint": datapoint
			}


		elif "multistep_atomic_claim" in config.prompt_name:
			try:
				direct_pred = direct_prediction_dict[response_id]["response"] if "plus_direct" in config.prompt_name else None
				if "gemini" in config.model_name:
					pred_response, conversation_log, messages, avg_logprobs = execute_atomic_claim_prompt_dict(
						prompt_dict,
						source_info,
						response,
						llm,
						direct_prediction=direct_pred,
					)
				else:
					pred_response, conversation_log, messages, avg_logprobs = execute_atomic_claim_prompt_dict_with_openai_api(
						prompt_dict,
						source_info,
						response,
						client,
						MODEL_ID,
						logger,
						direct_prediction=direct_pred,
					)
			except Exception as e:
				torch.save(hallucination_detection_prediction_dict, result_filename)
				logger.info(f"Responses saved under {result_filename} after {counter} dialogues because of Exception.")
				raise(e)

			hallucination_detection_prediction_dict[response_id] = {
				"response": pred_response, 
				"conversation_log": conversation_log, 
				"messages": messages, 
				"avg_logprobs": avg_logprobs,
				"datapoint": datapoint
			}

		elif "json_keyvalue" in config.prompt_name:
			try:
				direct_pred = direct_prediction_dict[response_id]["response"] if "plus_direct" in config.prompt_name else None
				if "gemini" in config.model_name:
					pred_response, conversation_log, messages, avg_logprobs = execute_json_keyvalue_prompt_dict(
						prompt_dict,
						source_info,
						response,
						llm,
						direct_prediction=direct_pred,
					)
				else:
					pred_response, conversation_log, messages, avg_logprobs = execute_json_keyvalue_prompt_dict_with_openai_api(
						prompt_dict,
						source_info,
						response,
						client,
						MODEL_ID,
						logger,
						direct_prediction=direct_pred,
					)
			except Exception as e:
				torch.save(hallucination_detection_prediction_dict, result_filename)
				logger.info(f"Responses saved under {result_filename} after {counter} dialogues because of Exception.")
				raise(e)

			hallucination_detection_prediction_dict[response_id] = {
				"response": pred_response, 
				"conversation_log": conversation_log, 
				"messages": messages, 
				"avg_logprobs": avg_logprobs,
				"datapoint": datapoint
			}

		else:
			prompt_input =f"{prompt_text}\n\nReference:\n{source_info}\n\nResponse:\n{response}"

			if "gemini" in config.model_name:
				llm_response = llm.invoke([HumanMessage(content=prompt_input)])
				pred_response = llm_response.content
				logprobs = llm_response.response_metadata["avg_logprobs"] if "avg_logprobs" in llm_response.response_metadata else -1
			else:
				messages = [
					{"role": "system", "content": "You are a helpful assistant that predicts hallucinations from reference response pairs."},
					{"role": "user", "content": prompt_input}
				]
				try:
					response_obj = completion_with_retries(
						client,
						MODEL_ID,
						messages,
						logger,
					)
					pred_response = response_obj.choices[0].message.content
					#print(response_obj)

				except AttributeError as e:
					torch.save(hallucination_detection_prediction_dict, result_filename)
					pred_response = response_obj
					print("ATTRIBUTE ERROR, RESPONSE IS STRING ALREADY:", pred_response)
					#raise(e)
				logprobs = -1

			hallucination_detection_prediction_dict[response_id] = {
				"response": pred_response, 
				"avg_logprobs": logprobs,
				"datapoint": datapoint
			}
		
		if counter < 5:
			logger.info("Sample_output")
			if "sqlite_hallucination" in config.prompt_name or "two_step" in config.prompt_name or config.prompt_name == "sqlite_id_select" or "multistep_atomic_claim" in config.prompt_name or "json_keyvalue" in config.prompt_name:
				logger.info(conversation_log)
				#logger.info(pprint.pformat(messages))
			else:
				logger.info("PROMPT:\n" + prompt_input + "\n")
				logger.info("RESPONSE:\n" + pred_response)

			

		counter += 1

		#every 50 dialogues save the responses checkpoint
		if counter % 50 == 0:
			torch.save(hallucination_detection_prediction_dict, result_filename)
			logger.info(f"Responses saved under {result_filename} after {counter} dialogues")
			

		if "gemini" not in config.model_name and "gpt-4" not in config.model_name and counter % 1 == 0:
			#also reinitatilise the client since the tokens expire otherwise
			# Obtain Application Default Credentials (ADC) token
		
			credentials, _ = default()
			auth_request = transport.requests.Request()
			credentials.refresh(auth_request)
			gcp_token = credentials.token

			# Construct the Vertex AI MaaS endpoint URL for the OpenAI library
			vertex_ai_endpoint_url = (
				f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/"
				f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/openapi"
			)

			# Initialize the OpenAI client pointing to the Vertex AI MaaS endpoint
			client = openai.OpenAI(
				base_url=vertex_ai_endpoint_url,
				api_key=gcp_token,  # Use the GCP token as the API key
			)
	
	#save the responses as torch file
	torch.save(hallucination_detection_prediction_dict, result_filename)
		
	logger.info(f"Program finished, responses saved under {result_filename}")



if __name__=="__main__":
    main()