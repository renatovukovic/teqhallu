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
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict
from collections import Counter

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load a .jsonl (JSON Lines) file into a list of dictionaries.

    Args:
        path (Path): Path to the .jsonl file.

    Returns:
        List[Dict[str, Any]]: List of parsed JSON objects.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Provided path is not a file: {path}")

    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:  # skip empty lines
                entries.append(json.loads(line))
    return entries


def combine_sources_and_responses(
    responses: List[Dict[str, Any]],
    sources: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Combine RAGTruth source entries with their model responses.

    Args:
        responses (List[Dict[str, Any]]): Loaded response entries.
        sources (List[Dict[str, Any]]): Loaded source entries (QA or Summary).

    Returns:
        List[Dict[str, Any]]: Each entry contains:
            - response id
            - source_id
            - task_type
            - source
            - question (QA) or None
            - context/source_info
            - prompt
            - model
            - response
            - hallucinations
    """
    # Build source lookup
    source_map: Dict[str, Dict[str, Any]] = {s["source_id"]: s for s in sources}

    combined_entries: List[Dict[str, Any]] = []

    for r in responses:
        source_id = r.get("source_id")
        source_entry = source_map.get(source_id, {})

        task_type = source_entry.get("task_type")
        source_info = source_entry.get("source_info")

        # Determine question and context depending on task type
        if task_type == "QA" and isinstance(source_info, dict):
            question: Optional[str] = source_info.get("question")
            context: Optional[str] = source_info.get("passages")
        else:
            question = None
            context = source_info if isinstance(source_info, str) else None

        combined_entry: Dict[str, Any] = {
            "response_id": r.get("id"),
            "split": r.get("split"),
            "source_id": source_id,
            "source_info": source_entry.get("source_info"),
            "task_type": task_type,
            "source": source_entry.get("source"),
            "question": question,
            "context": context,
            "prompt": source_entry.get("prompt"),
            "model": r.get("model"),
            "response": r.get("response"),
            "hallucinations": [
                {
                    "text": label.get("text"),
                    "start": label.get("start"),
                    "end": label.get("end"),
                    "label_type": label.get("label_type"),
                    "meta": label.get("meta"),
                }
                for label in r.get("labels", [])
            ],
        }

        combined_entries.append(combined_entry)

    return combined_entries


def group_by_task_type(entries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group RAGTruth entries by their task type (QA or Summary).

    Args:
        entries (List[Dict[str, Any]]): List of combined entries (sources + responses).

    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary where keys are task types and values
        are lists of entries of that type.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for entry in entries:
        task_type = entry.get("task_type", "Unknown")
        grouped[task_type].append(entry)

    return dict(grouped)  # convert defaultdict to regular dict


def summarize_task_types(grouped_entries: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    Summarize RAGTruth entries by task type.

    Args:
        grouped_entries (Dict[str, List[Dict[str, Any]]]): Dictionary with task_type as keys
            and lists of entries as values.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary where each task type maps to:
            - "num_entries": number of entries of that task type
            - "sources": set of unique source names for that task type
            - "splits": dictionary mapping split name to count of entries
    """
    summary: Dict[str, Dict[str, Any]] = {}

    for task_type, entries in grouped_entries.items():
        sources: Set[str] = set(entry.get("source", "Unknown") for entry in entries)

        # Count how many entries per split
        split_counts = Counter(entry.get("split", "Unknown") for entry in entries)

        summary[task_type] = {
            "num_entries": len(entries),
            "sources": sources,
            "splits": dict(split_counts),  # convert Counter to regular dict
        }

    return summary