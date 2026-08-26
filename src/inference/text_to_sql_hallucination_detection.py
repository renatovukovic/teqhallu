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

from langchain_google_vertexai import ChatVertexAI, HarmCategory, HarmBlockThreshold
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
import random
from typing import Dict, Tuple, List
import re
import openai
import tiktoken
import backoff
import logging


#local imports
from sql_interpretation.sql_interpreter import extract_sql_from_response, execute_multiple_queries_with_errors, analyse_sql_query

#exponential backoff
@backoff.on_exception(backoff.expo, openai.RateLimitError)
def chatcompletions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


#completion with retries
def completion_with_retries(client: openai.Client, 
                            model_name: str, 
                            input_messages: list[dict[str, str]],
                            logger: logging.Logger,
                            retries: int = 100,
							temperature: float = 0) -> openai.ChatCompletion:
        
	"""
    Function to get a completion from the model with retries in case of an exception.
    Returns the chatcompletion if it was successful, otherwise returns the exception as a string.
    
    Parameters:
    client: openai.Client
		The openai client object.
	model_name: str
		The name of the model to be used.
	input_prompt: str
		The input prompt for the model.
	logger: logging.Logger
		The logger object for logging.
	retries: int
		The number of retries in case of an exception.
	temperature: float
		temperature to be used by the model
    
      """

	#pprint.pprint(input_messages)

	#retry 100 times if an exception occurs, as the server is overloaded quite often, retry could solve this
	for i in range(retries):
	#catch any exception and save the responses so far
		try: 
			response = chatcompletions_with_backoff(
				client=client,
				model=model_name,
				messages=truncate_messages(input_messages, max_tokens=120000), #truncate messages to fit gpt-4o-mini context window of 128k and allow for response
				temperature=temperature,
				seed=42,
			)       
		except KeyboardInterrupt as e: #catch keyboard interrupt if I want to stop program
			raise(e)
		except openai.APIError as e: #catch any exception like time out or rate limit error and save the responses so far
			if i < retries:
				logger.info("Exception occured, try again.")
				logger.info(e)
				continue #try again
			else:
				raise(e)

		# except openai.InvalidRequestError as e:
		# 	logger.info("Invalid request error occured.")
		# 	logger.info(e)
		# 	logger.info(f"Input messages: {pprint.pformat(input_messages)}")
		# 	raise e

   
		
		break #stop the for loop if getting the response succeeded, i.e. no exception was raised

	try:
		return response
	except UnboundLocalError:
		return "No response"

def truncate_messages(messages: List[Dict[str, str]], max_tokens: int = 120000, model: str = "gpt-4o-mini") -> List[Dict[str, str]]:
    """
    Truncates a list of chat messages to ensure the total token count does not exceed `max_tokens`.
    The system message (first message) is always preserved. If necessary, it trims tokens from
    the start of subsequent messages while maintaining as much content as possible.

    Args:
        messages (List[Dict[str, str]]): List of chat messages, each containing 'role' and 'content'.
        max_tokens (int): Maximum allowed token count for the entire message history.
        model (str): The OpenAI model to use for tokenization.

    Returns:
        List[Dict[str, str]]: The truncated list of messages that fits within `max_tokens`.
    """
    if not messages:
        return []

    enc = tiktoken.encoding_for_model(model)

    # Tokenize each message and store its length
    tokenized_messages = [
        {"role": msg["role"], "content": msg["content"], "tokens": enc.encode(msg["content"])}
        for msg in messages
    ]

    # If there's a system message, keep it separate
    system_message = tokenized_messages[0] if tokenized_messages[0]["role"] == "system" else None
    non_system_messages = tokenized_messages[1:] if system_message else tokenized_messages

    # Calculate total token count
    total_tokens = sum(len(msg["tokens"]) for msg in tokenized_messages)

    # If we exceed max_tokens, start truncating from the beginning of non-system messages
    while total_tokens > max_tokens and non_system_messages:
        first_msg = non_system_messages[0]
        excess_tokens = total_tokens - max_tokens

        if len(first_msg["tokens"]) <= excess_tokens:
            # If the first message is small enough, remove it completely
            total_tokens -= len(first_msg["tokens"])
            non_system_messages.pop(0)
        else:
            # Otherwise, trim only the necessary tokens from the start
            first_msg["tokens"] = first_msg["tokens"][excess_tokens:]
            first_msg["content"] = enc.decode(first_msg["tokens"])
            total_tokens = max_tokens  # Now within the limit

    # Reconstruct the messages list
    truncated_messages = [{"role": msg["role"], "content": msg["content"]} for msg in non_system_messages]

    # Ensure the system message is included at the start if it exists
    if system_message:
        truncated_messages.insert(0, {"role": system_message["role"], "content": system_message["content"]})

    return truncated_messages

def execute_text_to_sql_prompt_dict(
	prompt_dict: Dict[str, str],
	reference: str,
	sampled_response: str,
	llm: ChatVertexAI,
	database_path: str,
	max_db_results: int = 5,
	return_column_values: bool = False,
	direct_prediction: str = None,
	) -> Tuple[str, str, List[BaseMessage], List[float]]:
	"""
	Execute a multi-step text-to-SQL prompt pipeline using a Vertex AI LLM,
	maintaining conversation history via LangChain message objects.

	Args:
		prompt_dict (Dict[str, str]): Dictionary of step prompts (step_name -> prompt_text).
		reference (str): Reference information to include with each prompt.
		initial_response (str): Initial response string for the first step.
		llm: A Vertex AI LLM model instance (e.g., ChatVertexAI).
		database_path (str): the bath to the SQLite DB
		max_db_results (int): maximum number of db results to show for queries
		return_column_values (bool): whether to return column value examples for the pragma queries
		direct_prediction (str): add a direct prediction from a dictionary to combine it with sql pipeline prediction

	Returns:
		Tuple[str, str, List[BaseMessage], List[float]]:
			- Final response string
			- Concatenated conversation log
			- List of LangChain message objects representing the conversation
			- list of avg logprobs
	"""

	messages: List[BaseMessage] = []
	conversation_log = ""
	logprob_list = []

	#get the list of current tables from the database
	table_name_query = "SELECT name FROM sqlite_master WHERE type='table';"
	table_names = execute_multiple_queries_with_errors(database_path, table_name_query)[0][1]
	table_names = [table_name[0] for table_name in table_names]
	#if table names are empty have a message that the DB is empty
	if not table_names:
		table_names = "There are no tables yet, since the database is empty."

	#if there is the table "sqlite_sequence" in the list of tables, remove it
	if "sqlite_sequence" in table_names:
		table_names.remove("sqlite_sequence")

	db_result = table_names

	for step_name, prompt_text in prompt_dict.items():
		# Build the input prompt for this step

		if not step_name == "Step 4":  
			step_prompt = prompt_dict[step_name].replace("{db_result_input}", str(db_result))
		else:
			step_prompt = prompt_dict[step_name]
		if step_name == "Step 1": #add the reference
			step_prompt += f"\n\n{reference}"
		
		if step_name == "Step 4": #add the response
			step_prompt += f"\n\n{sampled_response}"

		if step_name == "Step 5" and direct_prediction: #add the direct prediction if it is given as an input
			step_prompt += f"\n\nHere is the direct prediction:\n{direct_prediction}"
		

		# Append the human message to conversation
		messages.append(HumanMessage(content=step_prompt))

		# Send the full conversation to the LLM
		try:
			llm_response = llm.invoke(messages)
			if not llm_response.content:
				for i in range(10): #10 retries if the response is empty for some reason
					llm_response = llm.invoke(messages)
					if llm_response.content:
						break
		except Exception as e:	
			print(f"Messages:\n{messages}")
			raise(e)

		# Extract text output
		response = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
		avg_logprobs = llm_response.response_metadata["avg_logprobs"] if "avg_logprobs" in llm_response.response_metadata else -1
		logprob_list.append(avg_logprobs)
		
		do_not_execute_update_queries = True

		if step_name != "Step 5":
			# Extract SQL queries from the response
			sql_queries = extract_sql_from_response(response)
			
			if step_name == "Step 3": #update step
				do_not_execute_update_queries = False


			# Execute each SQL query and capture results or errors
			formatted_execution_results = ""
			for query in sql_queries:
				execution_results = execute_multiple_queries_with_errors(database_path, query, do_not_execute_update_queries=do_not_execute_update_queries)
				if step_name == "Step 1":# format the pragma queries to only return the column name and data type
					for result in execution_results:
						query_string = f"Query: \n{result[0]}"
						db_result_list = result[1]
						result_string = "DB Result:"
						if "PRAGMA" in query_string and db_result_list and isinstance(db_result_list, list):
							for column in db_result_list:
								result_string += f"\n{column[1]}: {column[2]}"
							if return_column_values: #get some column values for each column
								try:
									analysis_dict = analyse_sql_query(result[0])
								except Exception:
									continue
								if not analysis_dict["table_names"]:
									continue
								table_name = analysis_dict["table_names"][0]
								get_all_values_for_column_query = f"SELECT {column[1]} FROM {table_name};"
								column_value_results = execute_multiple_queries_with_errors(database_path, get_all_values_for_column_query, do_not_execute_update_queries=do_not_execute_update_queries)
								column_values = column_value_results[0][1]
								column_values = list(set([val[0] for val in column_values])) #get rid of the tuples for the results and remove duplicates
								#if there are too many values, then truncate by choosing randomly
								sample_values_string = ""
								if len(column_values) > 30:
									column_values = random.sample(column_values, 30)
									sample_values_string = "sample of "
								#add the values to the results for each column
								if not column_values: #no values in this column yet so add a string describing that
									column_values = "No values in this column yet."
								result_string += f"\n{sample_values_string}possible values for this column: {column_values}"

						else:
							#check the length of the db result (if it is a list) and truncate it if there are more than max_db_results rows
							#if result is an empty list, turn it to a message string stating that there are no results
							if not db_result_list or not isinstance(db_result_list, list):
								db_result_list = "No results for this query."
							if type(db_result_list) is list and len(db_result_list) > max_db_results:
								db_result_list = db_result_list[:max_db_results]
								result_string += f"\n{db_result_list}\nResults truncated to the first {max_db_results} rows."
							else:
								result_string += f"\n{db_result_list}"

						formatted_execution_results += f"\n\n{query_string}\n{result_string}"

				else:
					#formatted_execution_results += "\n\n".join([f"Query: \n{feedback[0]}\nDB Result: \n{feedback[1]}\n" for feedback in execution_results])
					try:
						for query, db_result_list in execution_results:
							#if result is an empty list, turn it to a message string stating that there are no results
							if not db_result_list:
								db_result_list = "No results for this query."
							#truncate the db result if there are more than max_db_results rows, check if it is a list because it can also be an update message
							if type(db_result_list) is list and len(db_result_list) > max_db_results:
								db_result_list = db_result_list[:max_db_results]
								formatted_execution_results += f"\n\nQuery: \n{query}\nDB Result: \n{db_result_list}\nResults truncated to the first {max_db_results} rows."
							else:
								formatted_execution_results += f"\n\nQuery: \n{query}\nDB Result: \n{db_result_list}"
					except ValueError as e:
						print(f"execution_results: {execution_results}")
						raise(e)
			#set the db_result for the next step
			if not formatted_execution_results:
				formatted_execution_results = "No queries generated so no query results."
			db_result = formatted_execution_results
				

		if not response: #it can happen that gemini-2.5-flash only generates thought tokens but in the end does not return any response
			response = "No output generated after thinking."


		# Append AI response to conversation
		messages.append(AIMessage(content=response))

		# Append to conversation log
		conversation_log += (
			f"\n--- {step_name.upper()} ---\n"
			f"Prompt Input:\n{step_prompt}\n\n"
			f"LLM Response:\n{response}\n"
		)

	return response, conversation_log, messages, logprob_list


def execute_text_to_sql_prompt_dict_with_openai_api(
	prompt_dict: Dict[str, str],
	reference: str,
	sampled_response: str,
	client: openai.Client,
	model_name: str,
	logger: logging.Logger,
	database_path: str,
	max_db_results: int = 5,
	return_column_values: bool = False,
	direct_prediction: str = None,
	) -> Tuple[str, str, List[Dict[str, str]], List[float]]:
	"""
	Execute a multi-step text-to-SQL prompt pipeline using an OpenAI model,
	maintaining conversation history.

	Args:
		prompt_dict (Dict[str, str]): Dictionary of step prompts (step_name -> prompt_text).
		reference (str): Reference information to include with each prompt.
		sampled_response (str): Initial response string for the first step.
		client (openai.Client): The OpenAI client.
		model_name (str): The name of the OpenAI model to use.
		logger (logging.Logger): The logger instance.
		database_path (str): the bath to the SQLite DB
		max_db_results (int): maximum number of db results to show for queries
		return_column_values (bool): whether to return column value examples for the pragma queries
		direct_prediction (str): add a direct prediction from a dictionary to combine it with sql pipeline prediction

	Returns:
		Tuple[str, str, List[Dict[str, str]], List[float]]:
			- Final response string
			- Concatenated conversation log
			- List of message dictionaries representing the conversation
			- list of avg logprobs (always [-1] for OpenAI)
	"""

	messages: List[Dict[str, str]] = [
		{"role": "system", "content": "You are an expert in sqlite3 queries for python. You are a helpful assistant that gets references and responses as input and should fill a database using SQLite3 queries and predict hallucinations."}
	]
	conversation_log = ""
	logprob_list = []

	#get the list of current tables from the database
	table_name_query = "SELECT name FROM sqlite_master WHERE type='table';"
	table_names = execute_multiple_queries_with_errors(database_path, table_name_query)[0][1]
	table_names = [table_name[0] for table_name in table_names]
	#if table names are empty have a message that the DB is empty
	if not table_names:
		table_names = "There are no tables yet, since the database is empty."

	#if there is the table "sqlite_sequence" in the list of tables, remove it
	if "sqlite_sequence" in table_names:
		table_names.remove("sqlite_sequence")

	db_result = table_names

	for step_name, prompt_text in prompt_dict.items():
		# Build the input prompt for this step

		if not step_name == "Step 4":  
			step_prompt = prompt_dict[step_name].replace("{db_result_input}", str(db_result))
		else:
			step_prompt = prompt_dict[step_name]
		if step_name == "Step 1": #add the reference
			step_prompt += f"\n\n{reference}"
		
		if step_name == "Step 4": #add the response
			step_prompt += f"\n\n{sampled_response}"

		if step_name == "Step 5" and direct_prediction: #add the direct prediction if it is given as an input
			step_prompt += f"\n\nHere is the direct prediction:\n{direct_prediction}"
		

		# Append the human message to conversation
		messages.append({"role": "user", "content": step_prompt})

		# Generate a response from ChatGPT
		try: 
			response_obj = completion_with_retries(
				client,
				model_name,
				messages,
				logger,
			)
			response = response_obj.choices[0].message.content
		except KeyboardInterrupt as e:
			raise(e)
		except openai.APIError as e:
			print(f"Messages:\n{messages}")
			raise e
		except openai.BadRequestError as e:
			print(f"Messages:\n{messages}")
			raise e	
		except TypeError as e:
			print(f"Messages:\n{messages}")
			response = response_obj
			#raise(e)
		except AttributeError as e:
			response = response_obj
			print("ATTRIBUTE ERROR, RESPONSE IS STRING ALREADY:", response)
			#raise(e)


		# Extract text output
		avg_logprobs = -1 # OpenAI API does not return avg_logprobs in the same way
		logprob_list.append(avg_logprobs)
		
		do_not_execute_update_queries = True

		if step_name != "Step 5":
			# Extract SQL queries from the response
			try:
				sql_queries = extract_sql_from_response(response)
			except TypeError as e:
				sql_queries = []
			
			if step_name == "Step 3": #update step
				do_not_execute_update_queries = False


			# Execute each SQL query and capture results or errors
			formatted_execution_results = ""
			for query in sql_queries:
				execution_results = execute_multiple_queries_with_errors(database_path, query, do_not_execute_update_queries=do_not_execute_update_queries)
				if step_name == "Step 1":# format the pragma queries to only return the column name and data type
					for result in execution_results:
						query_string = f"Query: \n{result[0]}"
						db_result_list = result[1]
						result_string = "DB Result:"
						if "PRAGMA" in query_string and db_result_list and isinstance(db_result_list, list):
							try:
								for column in db_result_list:
									result_string += f"\n{column[1]}: {column[2]}"
							except IndexError:
								continue
							if return_column_values: #get some column values for each column
								try:
									analysis_dict = analyse_sql_query(result[0])
								except Exception:
									continue
								if not analysis_dict["table_names"]:
									continue
								table_name = analysis_dict["table_names"][0]
								get_all_values_for_column_query = f"SELECT {column[1]} FROM {table_name};"
								column_value_results = execute_multiple_queries_with_errors(database_path, get_all_values_for_column_query, do_not_execute_update_queries=do_not_execute_update_queries)
								column_values = column_value_results[0][1]
								column_values = list(set([val[0] for val in column_values])) #get rid of the tuples for the results and remove duplicates
								#if there are too many values, then truncate by choosing randomly
								sample_values_string = ""
								if len(column_values) > 30:
									column_values = random.sample(column_values, 30)
									sample_values_string = "sample of "
								#add the values to the results for each column
								if not column_values: #no values in this column yet so add a string describing that
									column_values = "No values in this column yet."
								result_string += f"\n{sample_values_string}possible values for this column: {column_values}"

						else:
							#check the length of the db result (if it is a list) and truncate it if there are more than max_db_results rows
							#if result is an empty list, turn it to a message string stating that there are no results
							if not db_result_list or not isinstance(db_result_list, list):
								db_result_list = "No results for this query."
							if type(db_result_list) is list and len(db_result_list) > max_db_results:
								db_result_list = db_result_list[:max_db_results]
								result_string += f"\n{db_result_list}\nResults truncated to the first {max_db_results} rows."
							else:
								result_string += f"\n{db_result_list}"

						formatted_execution_results += f"\n\n{query_string}\n{result_string}"

				else:
					#formatted_execution_results += "\n\n".join([f"Query: \n{feedback[0]}\nDB Result: \n{feedback[1]}\n" for feedback in execution_results])
					try:
						for query, db_result_list in execution_results:
							#if result is an empty list, turn it to a message string stating that there are no results
							if not db_result_list:
								db_result_list = "No results for this query."
							#truncate the db result if there are more than max_db_results rows, check if it is a list because it can also be an update message
							if type(db_result_list) is list and len(db_result_list) > max_db_results:
								db_result_list = db_result_list[:max_db_results]
								formatted_execution_results += f"\n\nQuery: \n{query}\nDB Result: \n{db_result_list}\nResults truncated to the first {max_db_results} rows."
							else:
								formatted_execution_results += f"\n\nQuery: \n{query}\nDB Result: \n{db_result_list}"
					except ValueError as e:
						print(f"execution_results: {execution_results}")
						raise(e)
			#set the db_result for the next step
			if not formatted_execution_results:
				formatted_execution_results = "No queries generated so no query results."
			db_result = formatted_execution_results
				

		if not response: #it can happen that gemini-2.5-flash only generates thought tokens but in the end does not return any response
			response = "No output generated after thinking."


		# Append AI response to conversation
		messages.append({"role": "assistant", "content": response})

		# Append to conversation log
		conversation_log += (
			f"\n--- {step_name.upper()} ---\n"
			f"Prompt Input:\n{step_prompt}\n\n"
			f"LLM Response:\n{response}\n"
		)

	return response, conversation_log, messages, logprob_list


def load_prompt_steps(file_path: str) -> dict[str, str]:
    """
    Reads a multi-step prompt text file and returns a dictionary mapping
    'Step n' to its corresponding text.

    Args:
        file_path (str): Path to the prompt text file.

    Returns:
        dict[str, str]: A dictionary with keys like 'Step 1', 'Step 2', ..., 'Step 5'
                        and values as the step text.
    """
    steps = {}
    current_step = None
    step_lines = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line.startswith("Step ") and "–" in stripped_line:
                # Save the previous step if exists
                if current_step and step_lines:
                    steps[current_step] = "\n".join(step_lines).strip()
                    step_lines = []

                # Extract step key (e.g., "Step 1")
                current_step = stripped_line.split("–")[0].strip()
            elif current_step:
                step_lines.append(line.rstrip())

        # Save the last step
        if current_step and step_lines:
            steps[current_step] = "\n".join(step_lines).strip()

    return steps

#### functions for id select matching approach
def load_id_select_prompt_steps(file_path: str) -> Dict[str, str]:
    """
    Reads a multi-step prompt text file and returns a dictionary mapping
    'Step n' to its corresponding text.

    Any content appearing before 'Step 1' is prepended to the value of 'Step 1'.
    """

    step_header_pattern = re.compile(r"^(Step\s+\d+)\s*[–—-]\s*(.+)$")

    steps: Dict[str, str] = {}
    preamble_lines: list[str] = []
    current_key = None
    buffer: list[str] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip()
            match = step_header_pattern.match(stripped)

            if match:
                # Flush previous step
                if current_key is not None:
                    steps[current_key] = "\n".join(buffer).strip()
                    buffer = []

                step_id, step_title = match.groups()
                current_key = step_id

                # If this is Step 1, prepend the preamble
                if current_key == "Step 1" and preamble_lines:
                    buffer.extend(preamble_lines)
                    buffer.append("")  # spacer
                    preamble_lines = []

            else:
                if current_key is None:
                    # Still in Task Overview / preamble
                    if stripped:  # ignore leading empty lines
                        preamble_lines.append(stripped)
                else:
                    buffer.append(stripped)

        # Flush last step
        if current_key is not None:
            steps[current_key] = "\n".join(buffer).strip()

    return steps




def execute_text_to_sql_id_select_prompt_dict(
	prompt_dict: Dict[str, str],
	reference: str,
	sampled_response: str,
	llm,
	database_path: str,
	max_db_results: int = 100,
	) -> Tuple[str, str, List[BaseMessage], List[float]]:
	"""
	Execute a multi-step text-to-SQL prompt pipeline in READ-ONLY mode.

	- No UPDATE / INSERT / DELETE queries are ever executed
	- Designed to work with the hallucination-detection prompt (Steps 1–4)

	Returns:
		- Final response string
		- Concatenated conversation log
		- List of LangChain message objects
		- List of average logprobs
	"""

	messages: List[BaseMessage] = []
	conversation_log = ""
	logprob_list: List[float] = []

	# ------------------------------------------------------------------
	# Step 0: Get table names
	# ------------------------------------------------------------------
	table_name_query = "SELECT name FROM sqlite_master WHERE type='table';"
	table_names_result = execute_multiple_queries_with_errors(
		database_path,
		table_name_query,
		do_not_execute_update_queries=True,
	)

	table_names = table_names_result[0][1] if table_names_result else []
	table_names = [row[0] for row in table_names] if table_names else []

	if not table_names:
		table_names = "There are no tables, since the database is empty."

	if isinstance(table_names, list) and "sqlite_sequence" in table_names:
		table_names.remove("sqlite_sequence")

	db_result = table_names

	# ------------------------------------------------------------------
	# Main loop over prompt steps
	# ------------------------------------------------------------------
	for step_name, step_text in prompt_dict.items():

		# Inject DB results
		step_prompt = step_text.replace("{db_result_input}", str(db_result))

		# Attach reference and response at the correct steps
		if step_name == "Step 1":
			step_prompt += f"\n\n{reference}"

		if step_name == "Step 3":
			step_prompt += f"\n\n{sampled_response}"

		# Add human message
		messages.append(HumanMessage(content=step_prompt))

		# Invoke LLM
		try:
			llm_response = llm.invoke(messages)
			if not llm_response.content:
				for _ in range(10):
					llm_response = llm.invoke(messages)
					if llm_response.content:
						break
		except Exception as e:
			print(f"Messages at failure:\n{messages}")
			raise e

		response = llm_response.content or "No output generated after thinking."
		avg_logprobs = llm_response.response_metadata.get("avg_logprobs", -1)
		logprob_list.append(avg_logprobs)

		# ------------------------------------------------------------------
		# SQL extraction + execution (READ-ONLY)
		# ------------------------------------------------------------------
		sql_queries = extract_sql_from_response(response)
		formatted_execution_results = ""

		for query in sql_queries:
			execution_results = execute_multiple_queries_with_errors(
				database_path,
				query,
				do_not_execute_update_queries=True,  # HARD READ-ONLY
			)

			if step_name == "Step 1":
				# Special formatting for PRAGMA output
				for executed_query, db_result_list in execution_results:
					result_string = "DB Result:"
					query_string = f"Query:\n{executed_query}"

					if "PRAGMA" in executed_query and isinstance(db_result_list, list):
						for column in db_result_list:
							result_string += f"\n{column[1]}: {column[2]}"

							if "id" not in column[1].lower(): #skip this for IDs
								try:
									analysis = analyse_sql_query(executed_query)
								except Exception:
									continue

								if not analysis["table_names"]:
									continue

								table_name = analysis["table_names"][0]
								value_query = f"SELECT {column[1]} FROM {table_name};"
								value_results = execute_multiple_queries_with_errors(
									database_path,
									value_query,
									do_not_execute_update_queries=True,
								)

								values = value_results[0][1]
								values = list(set(v[0] for v in values)) if values else []

								if not values:
									values = "No values in this column."
								elif len(values) > max_db_results:
									values = values[:max_db_results]
								
								result_string += "\npossible values in this column:"
								result_string += f"\n{values}"
					else:
						if not db_result_list:
							db_result_list = "No results for this query."
						elif isinstance(db_result_list, list) and len(db_result_list) > max_db_results:
							db_result_list = db_result_list[:max_db_results]
							result_string += f"\n{db_result_list}\nResults truncated."
						else:
							result_string += f"\n{db_result_list}"

					formatted_execution_results += f"\n\n{query_string}\n{result_string}"

			else:
				# Standard SELECT formatting
				for executed_query, db_result_list in execution_results:
					if not db_result_list:
						db_result_list = "No results for this query."
					elif isinstance(db_result_list, list) and len(db_result_list) > max_db_results:
						db_result_list = db_result_list[:max_db_results]
						formatted_execution_results += (
							f"\n\nQuery:\n{executed_query}\nDB Result:\n{db_result_list}\n"
							f"Results truncated to first {max_db_results} rows."
						)
					else:
						formatted_execution_results += (
							f"\n\nQuery:\n{executed_query}\nDB Result:\n{db_result_list}"
						)

		if not formatted_execution_results:
			formatted_execution_results = "No queries generated so no query results."

		db_result = formatted_execution_results

		# Append AI message
		messages.append(AIMessage(content=response))

		# Update conversation log
		conversation_log += (
			f"\n--- {step_name.upper()} ---\n"
			f"Prompt Input:\n{step_prompt}\n\n"
			f"LLM Response:\n{response}\n"
		)

	return response, conversation_log, messages, logprob_list



#functions for two step sql prompt

def parse_twostep_prompt_file(file_path: str) -> Dict[str, str]:
    """
    Parse a two-step hallucination detection prompt text file into a dictionary.

    The function splits the file content at "Step 1" and "Step 2"
    to directly extract instructions for each step.

    Args:
        file_path (str): Path to the .txt file containing the two-step prompt.

    Returns:
        Dict[str, str]: Dictionary with "Step 1" and "Step 2" as keys
                        and their corresponding instructions as values.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Split based on "Step 1" and "Step 2"
    parts = text.split("Step ")
    steps: Dict[str, str] = {}

    for part in parts:
        if part.strip().startswith("1"):
            steps["Step 1"] = part[1:].strip()  # remove leading "1"
        elif part.strip().startswith("2"):
            steps["Step 2"] = part[1:].strip()  # remove leading "2"

    return steps

def execute_two_step_prompt_dict(
    prompt_dict: Dict[str, str],
    reference: str,
    sampled_response: str,
    llm: ChatVertexAI,
    max_retries: int = 5,
) -> Tuple[str, str, List[HumanMessage | AIMessage], List[float]]:
    """
    Execute a simplified two-step hallucination detection prompt pipeline using an LLM.

    Unlike the text-to-SQL pipeline, this function does NOT perform any SQL execution.
    It only builds prompts for each step, appends reference/response text where appropriate,
    and collects LLM outputs with conversation history.

    Args:
        prompt_dict (Dict[str, str]): Dictionary of step prompts (step_name -> prompt_text).
        reference (str): The gold reference text for comparison.
        sampled_response (str): The system-generated response text to evaluate.
        llm: A chat LLM instance (e.g., ChatVertexAI or similar).
        max_retries (int): Maximum retries if the LLM response is empty.

    Returns:
        Tuple[str, str, List[HumanMessage | AIMessage], List[float]]:
            - Final LLM response string
            - Concatenated conversation log
            - List of LangChain message objects (Human + AI)
            - List of avg logprobs for each step
    """
    messages: List[HumanMessage | AIMessage] = []
    conversation_log = ""
    logprob_list: List[float] = []
    response = ""

    for step_name, prompt_text in prompt_dict.items():
        # Add reference to Step 1
        if step_name == "Step 1":
            step_prompt = f"{prompt_text}\n\nReference:\n{reference}\n\nResponse:\n{sampled_response}"
        else:
            step_prompt = prompt_text

        # Append human message
        messages.append(HumanMessage(content=step_prompt))

        # Query the LLM with retries
        llm_response = None
        for _ in range(max_retries):
            llm_response = llm.invoke(messages)
            if llm_response.content:
                break

        if not llm_response or not llm_response.content:
            response = "No output generated after retries."
        else:
            response = llm_response.content

        # Extract avg_logprobs if available
        avg_logprobs = (
            llm_response.response_metadata["avg_logprobs"]
            if "avg_logprobs" in getattr(llm_response, "response_metadata", {})
            else -1
        )
        logprob_list.append(avg_logprobs)

        # Append AI message
        messages.append(AIMessage(content=response))

        # Append to log
        conversation_log += (
            f"\n--- {step_name.upper()} ---\n"
            f"Prompt Input:\n{step_prompt}\n\n"
            f"LLM Response:\n{response}\n"
        )

    return response, conversation_log, messages, logprob_list
