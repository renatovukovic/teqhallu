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

import torch
import json
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import re
from langchain_google_vertexai import ChatVertexAI, HarmCategory, HarmBlockThreshold
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
import langchain_core

torch.serialization.add_safe_globals([langchain_core.messages.human.HumanMessage])
torch.serialization.add_safe_globals([langchain_core.messages.ai.AIMessage])

def parse_llm_response(response_str):
    """
    Extract the boolean hallucination prediction from the LLM response string.
    Expected format: {"hallucinations in response": true/false, ...}
    """
    try:
        # Try finding a JSON block in the response
        json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return data.get("hallucinations in response", False)
    except Exception:
        pass
    
    # Fallback: simple text check if JSON parsing fails
    if '"hallucinations in response": true' in response_str.lower():
        return True
    return False

def normalize_type(t):
    """Normalize hallucination type strings for comparison."""
    t = t.lower().strip()
    if t == "reasoning":
        return "reasoning error"
    if t == "over-reliance":
        return "overreliance"
    return t

def parse_llm_types(response_str):
    """
    Extract the hallucination types from the LLM response string.
    Expected format: {"type": "Type1 | Type2", ...}
    """
    try:
        json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if data.get("hallucinations in response", False):
                types_val = data.get("type", "")
                if isinstance(types_val, str):
                    # split by | or other common delimiters
                    split_types = re.split(r'[|/,]', types_val)
                    return [normalize_type(t) for t in split_types if t.strip()]
                elif isinstance(types_val, list):
                    return [normalize_type(str(t)) for t in types_val]
    except Exception:
        pass
    return []

def parse_ground_truth_types(index_val):
    """
    Extract the hallucination types from the GT index field.
    Expected format: "[[turn_idx, type], ...]" or a list already
    """
    try:
        if isinstance(index_val, str):
            # Handle potential single quotes in stringified JSON
            cleaned_val = index_val.replace("'", '"')
            indices = json.loads(cleaned_val)
        else:
            indices = index_val
        
        if isinstance(indices, list):
            types = []
            for item in indices:
                if isinstance(item, list) and len(item) > 1:
                    types.append(normalize_type(item[1]))
            return types
    except Exception:
        pass
    return []

def evaluate_hallucination_types(all_gt_types, all_pred_types):
    """
    Calculate and print F1 scores for each hallucination type and a micro F1 score.
    Order: Non-factual & Incoherence & Irrelevance & Overreliance & Reasoning Error & Micro F1
    """
    canonical_types = ["non-factual", "incoherence", "irrelevance", "overreliance", "reasoning error"]
    
    type_f1s = []
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for t in canonical_types:
        y_true_binary = [1 if t in gt else 0 for gt in all_gt_types]
        y_pred_binary = [1 if t in pred else 0 for pred in all_pred_types]
        
        tp = sum(1 for gt, pred in zip(y_true_binary, y_pred_binary) if gt == 1 and pred == 1)
        fp = sum(1 for gt, pred in zip(y_true_binary, y_pred_binary) if gt == 0 and pred == 1)
        fn = sum(1 for gt, pred in zip(y_true_binary, y_pred_binary) if gt == 1 and pred == 0)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        type_f1s.append(f1)
    
    # Micro F1 calculation
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n--- Hallucination Type Metrics ---")
    print("Non-factual & Incoherence & Irrelevance & Overreliance & Reasoning Error & Micro F1")
    output_parts = [f"{f1*100:.2f}" for f1 in type_f1s]
    output_parts.append(f"{micro_f1*100:.2f}")
    print(" & ".join(output_parts))

def evaluate(results_file):
    print(f"Loading results from {results_file}")
    results = torch.load(results_file)
    
    y_true = []
    y_pred = []
    
    all_gt_types = []
    all_pred_types = []
    
    categories = {
        "TP": [], # True Positive
        "TN": [], # True Negative
        "FP": [], # False Positive
        "FN": []  # False Negative
    }
    
    missing_label_count = 0

    for res_id, data in results.items():
        # Label in DiaHalu is 1 for hallucination, 0 for not
        if 'datapoint' in data and 'label' in data['datapoint']:
            label = bool(data['datapoint']['label'])
            prediction = parse_llm_response(data['response'])
            
            y_true.append(label)
            y_pred.append(prediction)
            
            # Extract types for detailed evaluation
            gt_types = parse_ground_truth_types(data['datapoint'].get('index', []))
            pred_types = parse_llm_types(data['response'])
            
            all_gt_types.append(gt_types)
            all_pred_types.append(pred_types)
            
            entry = {
                "ID": res_id,
                "text": data['datapoint'].get('text', 'N/A'),
                "llm_response": data['response'],
                "ground_truth": label,
                "prediction": prediction
            }
            
            if label and prediction:
                categories["TP"].append(entry)
            elif not label and not prediction:
                categories["TN"].append(entry)
            elif not label and prediction:
                categories["FP"].append(entry)
            elif label and not prediction:
                categories["FN"].append(entry)
        else:
            missing_label_count += 1

    if not y_true:
        print("No valid datapoints with labels found in results.")
        return

    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    confusion_mat = confusion_matrix(y_true, y_pred)

    print("\n--- Evaluation Metrics ---")
    print(f"Total datapoints: {len(y_true)}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"True Negatives: {confusion_mat[0][0]}")
    print(f"False Positives: {confusion_mat[0][1]}")
    print(f"False Negatives: {confusion_mat[1][0]}")
    print(f"True Positives: {confusion_mat[1][1]}")
    
    # Detailed type evaluation
    evaluate_hallucination_types(all_gt_types, all_pred_types)
    
    # Subset evaluation for MultiWOZ and DSTC
    evaluate_by_source(results, ["MultiWOZ"])

    if missing_label_count > 0:
        print(f"Warning: {missing_label_count} datapoints were skipped due to missing labels.")

def evaluate_by_source(results, target_sources):
    """
    Evaluate precision, recall, and F1 only for datapoints from specific sources.
    """
    y_true = []
    y_pred = []
    
    for res_id, data in results.items():
        if 'datapoint' in data and 'label' in data['datapoint']:
            source = data['datapoint'].get('source', '')
            if source in target_sources:
                label = bool(data['datapoint']['label'])
                prediction = parse_llm_response(data['response'])
                y_true.append(label)
                y_pred.append(prediction)

    if not y_true:
        print(f"\n--- Subset Evaluation ({', '.join(target_sources)}) ---")
        print("No datapoints found for these sources.")
        return

    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(f"\n--- Subset Evaluation ({', '.join(target_sources)}) ---")
    print(f"Total datapoints: {len(y_true)}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True, help="Config name used for inference")
    args = parser.parse_args()
    
    results_file = f"inference/results/{args.config_name}_responses.pt"
    evaluate(results_file)
