# coding=utf-8
#
# Copyright 2026
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

from langchain_google_vertexai import ChatVertexAI, HarmCategory, HarmBlockThreshold
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
import random
from typing import Dict, Tuple, List, Any
import re
import openai
import tiktoken
import backoff
import logging

import logging
import openai
from typing import Dict, List, Tuple

from inference.text_to_sql_hallucination_detection import completion_with_retries

def load_multistep_prompt_steps(file_path: str) -> Dict[str, str]:
    """
    Reads a multi-step prompt text file with headers like '--- STEP 1: ... ---'
    and returns a dictionary mapping 'Step 1', 'Step 2', etc. to their step text.
    """
    steps = {}
    current_step = None
    step_lines = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            match = re.search(r'\bSTEP\s+(\d+)', line, re.IGNORECASE)
            if match:
                if current_step and step_lines:
                    steps[current_step] = "\n".join(step_lines).strip()
                    step_lines = []
                step_num = match.group(1)
                current_step = f"Step {step_num}"
            elif current_step:
                step_lines.append(line.rstrip())

        if current_step and step_lines:
            steps[current_step] = "\n".join(step_lines).strip()

    return steps


def extract_text_content(content: Any) -> str:
    """
    Safely extract plain string content from string or structured content (e.g. list of dicts).
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                if "text" in part:
                    text_parts.append(str(part["text"]))
                else:
                    text_parts.append(str(part))
            else:
                text_parts.append(str(part))
        return "".join(text_parts)
    else:
        return str(content) if content is not None else ""


def execute_atomic_claim_prompt_dict(
    prompt_dict: Dict[str, str],
    reference: Any,
    sampled_response: Any,
    llm: ChatVertexAI,
    direct_prediction: str = None,
    max_retries: int = 5,
) -> Tuple[str, str, List[BaseMessage], List[float]]:
    """
    Execute a multi-step atomic claim decomposition and verification prompt pipeline 
    using a LangChain ChatVertexAI LLM model.
    """
    ref_str = str(reference) if not isinstance(reference, str) else reference
    resp_str = str(sampled_response) if not isinstance(sampled_response, str) else sampled_response
    direct_pred_str = str(direct_prediction) if (direct_prediction is not None and not isinstance(direct_prediction, str)) else direct_prediction

    messages: List[BaseMessage] = [
        SystemMessage(
            content=(
                "You are an expert in claim-level factual verification. "
                "Your task is to decompose responses into atomic claims, evaluate "
                "each against a reference text, and make a final hallucination decision."
            )
        )
    ]
    conversation_log = ""
    logprob_list = []

    claims_output = ""
    verification_output = ""

    for step_name, prompt_text in prompt_dict.items():
        if step_name == "Step 1":
            step_prompt = prompt_text + f"\n\n{resp_str}"

        elif step_name == "Step 2":
            step_prompt = (
                prompt_text.replace("{claims_input}", str(claims_output))
                .replace("{reference_input}", ref_str)
            )

        elif step_name == "Step 3":
            step_prompt = (
                prompt_text.replace("{verification_results_input}", str(verification_output))
                .replace("{response_input}", resp_str)
                .replace("{reference_input}", ref_str)
            )
            if direct_pred_str:
                step_prompt += f"\n\nHere is the initial hallucination prediction:\n{direct_pred_str}"

        else:
            step_prompt = prompt_text

        messages.append(HumanMessage(content=step_prompt))

        llm_response = None
        response = ""
        for attempt in range(max_retries):
            try:
                llm_response = llm.invoke(messages)
                if llm_response and llm_response.content:
                    response = extract_text_content(llm_response.content)
                    if response.strip():
                        break
            except Exception as e:
                logging.warning(f"llm.invoke exception on {step_name}, attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    response = f"Error during generation: {e}"

        if not response:
            response = "No output generated."

        if step_name == "Step 1":
            claims_output = response
        elif step_name == "Step 2":
            verification_output = response

        avg_logprobs = (
            llm_response.response_metadata["avg_logprobs"]
            if llm_response and hasattr(llm_response, "response_metadata") and "avg_logprobs" in getattr(llm_response, "response_metadata", {})
            else -1
        )
        logprob_list.append(avg_logprobs)

        messages.append(AIMessage(content=response))

        conversation_log += (
            f"\n--- {step_name.upper()} ---\n"
            f"Prompt Input:\n{step_prompt}\n\n"
            f"LLM Response:\n{response}\n"
        )

    return response, conversation_log, messages, logprob_list


def execute_json_keyvalue_prompt_dict(
    prompt_dict: Dict[str, str],
    reference: Any,
    sampled_response: Any,
    llm: ChatVertexAI,
    direct_prediction: str = None,
    max_retries: int = 5,
) -> Tuple[str, str, List[BaseMessage], List[float]]:
    """
    Execute a multi-step semi-structured JSON key-value extraction and comparison pipeline 
    using a LangChain ChatVertexAI LLM model.
    """
    ref_str = str(reference) if not isinstance(reference, str) else reference
    resp_str = str(sampled_response) if not isinstance(sampled_response, str) else sampled_response
    direct_pred_str = str(direct_prediction) if (direct_prediction is not None and not isinstance(direct_prediction, str)) else direct_prediction

    messages: List[BaseMessage] = [
        SystemMessage(
            content=(
                "You are an expert in structured information extraction and key-value verification. "
                "Your task is to parse reference text and candidate responses into JSON structures, "
                "compare their attributes, and determine if hallucinations exist."
            )
        )
    ]
    conversation_log = ""
    logprob_list = []

    reference_json = ""
    response_json = ""
    json_comparison = ""

    for step_name, prompt_text in prompt_dict.items():
        if step_name == "Step 1":
            step_prompt = prompt_text + f"\n\n{ref_str}"

        elif step_name == "Step 2":
            step_prompt = prompt_text + f"\n\n{resp_str}"

        elif step_name == "Step 3":
            step_prompt = (
                prompt_text.replace("{reference_json_input}", str(reference_json))
                .replace("{response_json_input}", str(response_json))
            )

        elif step_name == "Step 4":
            step_prompt = (
                prompt_text.replace("{json_comparison_input}", str(json_comparison))
                .replace("{response_input}", resp_str)
                .replace("{reference_input}", ref_str)
            )
            if direct_pred_str:
                step_prompt += f"\n\nHere is the initial hallucination prediction:\n{direct_pred_str}"

        else:
            step_prompt = prompt_text

        messages.append(HumanMessage(content=step_prompt))

        llm_response = None
        response = ""
        for attempt in range(max_retries):
            try:
                llm_response = llm.invoke(messages)
                if llm_response and llm_response.content:
                    response = extract_text_content(llm_response.content)
                    if response.strip():
                        break
            except Exception as e:
                logging.warning(f"llm.invoke exception on {step_name}, attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    response = f"Error during generation: {e}"

        if not response:
            response = "No output generated."

        if step_name == "Step 1":
            reference_json = response
        elif step_name == "Step 2":
            response_json = response
        elif step_name == "Step 3":
            json_comparison = response

        avg_logprobs = (
            llm_response.response_metadata["avg_logprobs"]
            if llm_response and hasattr(llm_response, "response_metadata") and "avg_logprobs" in getattr(llm_response, "response_metadata", {})
            else -1
        )
        logprob_list.append(avg_logprobs)

        messages.append(AIMessage(content=response))

        conversation_log += (
            f"\n--- {step_name.upper()} ---\n"
            f"Prompt Input:\n{step_prompt}\n\n"
            f"LLM Response:\n{response}\n"
        )

    return response, conversation_log, messages, logprob_list


def execute_atomic_claim_prompt_dict_with_openai_api(
    prompt_dict: Dict[str, str],
    reference: Any,
    sampled_response: Any,
    client: openai.Client,
    model_name: str,
    logger: logging.Logger,
    direct_prediction: str = None,
) -> Tuple[str, str, List[Dict[str, str]], List[float]]:
    """
    Execute a multi-step atomic claim decomposition and verification prompt pipeline 
    using an OpenAI model, maintaining conversation history.
    """
    ref_str = str(reference) if not isinstance(reference, str) else reference
    resp_str = str(sampled_response) if not isinstance(sampled_response, str) else sampled_response
    direct_pred_str = str(direct_prediction) if (direct_prediction is not None and not isinstance(direct_prediction, str)) else direct_prediction

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are an expert in claim-level factual verification. "
                "Your task is to decompose responses into atomic claims, evaluate "
                "each against a reference text, and make a final hallucination decision."
            ),
        }
    ]
    conversation_log = ""
    logprob_list = []

    claims_output = ""
    verification_output = ""

    for step_name, prompt_text in prompt_dict.items():
        # Build prompt input for each step
        if step_name == "Step 1":
            step_prompt = prompt_text + f"\n\n{resp_str}"

        elif step_name == "Step 2":
            step_prompt = (
                prompt_text.replace("{claims_input}", str(claims_output))
                .replace("{reference_input}", ref_str)
            )

        elif step_name == "Step 3":
            step_prompt = (
                prompt_text.replace("{verification_results_input}", str(verification_output))
                .replace("{response_input}", resp_str)
                .replace("{reference_input}", ref_str)
            )
            if direct_pred_str:
                step_prompt += f"\n\nHere is the initial hallucination prediction:\n{direct_pred_str}"

        else:
            step_prompt = prompt_text

        # Append user message
        messages.append({"role": "user", "content": step_prompt})

        # Query API
        try:
            response_obj = completion_with_retries(
                client,
                model_name,
                messages,
                logger,
            )
            response = response_obj.choices[0].message.content
        except KeyboardInterrupt as e:
            raise e
        except (openai.APIError, openai.BadRequestError) as e:
            print(f"Messages:\n{messages}")
            raise e
        except (TypeError, AttributeError):
            response = response_obj

        if not response:
            response = "No output generated."

        # Track outputs for downstream steps
        if step_name == "Step 1":
            claims_output = response
        elif step_name == "Step 2":
            verification_output = response

        logprob_list.append(-1)

        # Append assistant response
        messages.append({"role": "assistant", "content": response})

        # Append to log
        conversation_log += (
            f"\n--- {step_name.upper()} ---\n"
            f"Prompt Input:\n{step_prompt}\n\n"
            f"LLM Response:\n{response}\n"
        )

    return response, conversation_log, messages, logprob_list


def execute_json_keyvalue_prompt_dict_with_openai_api(
    prompt_dict: Dict[str, str],
    reference: Any,
    sampled_response: Any,
    client: openai.Client,
    model_name: str,
    logger: logging.Logger,
    direct_prediction: str = None,
) -> Tuple[str, str, List[Dict[str, str]], List[float]]:
    """
    Execute a multi-step semi-structured JSON key-value extraction and comparison pipeline 
    using an OpenAI model, maintaining conversation history.
    """
    ref_str = str(reference) if not isinstance(reference, str) else reference
    resp_str = str(sampled_response) if not isinstance(sampled_response, str) else sampled_response
    direct_pred_str = str(direct_prediction) if (direct_prediction is not None and not isinstance(direct_prediction, str)) else direct_prediction

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are an expert in structured information extraction and key-value verification. "
                "Your task is to parse reference text and candidate responses into JSON structures, "
                "compare their attributes, and determine if hallucinations exist."
            ),
        }
    ]
    conversation_log = ""
    logprob_list = []

    reference_json = ""
    response_json = ""
    json_comparison = ""

    for step_name, prompt_text in prompt_dict.items():
        # Build prompt input for each step
        if step_name == "Step 1":
            step_prompt = prompt_text + f"\n\n{ref_str}"

        elif step_name == "Step 2":
            step_prompt = prompt_text + f"\n\n{resp_str}"

        elif step_name == "Step 3":
            step_prompt = (
                prompt_text.replace("{reference_json_input}", str(reference_json))
                .replace("{response_json_input}", str(response_json))
            )

        elif step_name == "Step 4":
            step_prompt = (
                prompt_text.replace("{json_comparison_input}", str(json_comparison))
                .replace("{response_input}", resp_str)
                .replace("{reference_input}", ref_str)
            )
            if direct_pred_str:
                step_prompt += f"\n\nHere is the initial hallucination prediction:\n{direct_pred_str}"

        else:
            step_prompt = prompt_text

        # Append user message
        messages.append({"role": "user", "content": step_prompt})

        # Query API
        try:
            response_obj = completion_with_retries(
                client,
                model_name,
                messages,
                logger,
            )
            response = response_obj.choices[0].message.content
        except KeyboardInterrupt as e:
            raise e
        except (openai.APIError, openai.BadRequestError) as e:
            print(f"Messages:\n{messages}")
            raise e
        except (TypeError, AttributeError):
            response = response_obj

        if not response:
            response = "No output generated."

        # Track outputs for downstream steps
        if step_name == "Step 1":
            reference_json = response
        elif step_name == "Step 2":
            response_json = response
        elif step_name == "Step 3":
            json_comparison = response

        logprob_list.append(-1)

        # Append assistant response
        messages.append({"role": "assistant", "content": response})

        # Append to log
        conversation_log += (
            f"\n--- {step_name.upper()} ---\n"
            f"Prompt Input:\n{step_prompt}\n\n"
            f"LLM Response:\n{response}\n"
        )

    return response, conversation_log, messages, logprob_list





###example running the functions above

# # 1. Load prompts using your existing function
# atomic_prompts = load_prompt_steps("atomic_claim_baseline_prompts.txt")
# json_prompts = load_prompt_steps("json_keyvalue_baseline_prompts.txt")

# # 2. Run Atomic Claim Baseline
# res_atomic, log_atomic, msgs_atomic, _ = execute_atomic_claim_prompt_dict_with_openai_api(
#     prompt_dict=atomic_prompts,
#     reference=ref_text,
#     sampled_response=cand_response,
#     client=client,
#     model_name="gpt-4o",
#     logger=logger,
#     direct_prediction=initial_pred,
# )

# # 3. Run JSON Key-Value Baseline
# res_json, log_json, msgs_json, _ = execute_json_keyvalue_prompt_dict_with_openai_api(
#     prompt_dict=json_prompts,
#     reference=ref_text,
#     sampled_response=cand_response,
#     client=client,
#     model_name="gpt-4o",
#     logger=logger,
#     direct_prediction=initial_pred,
# )