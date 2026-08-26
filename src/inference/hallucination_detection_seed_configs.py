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



#### RAGTruth prediction configs ####
seeds = [1, 2, 3, 4, 5]
tasks = ["QA", "Summary", "Data2txt"]
#baseline less strict prompt

for task in tasks:
    for seed in seeds:
        var_name = f"ragtruth_test_{task.lower()}_baseline_less_strict_gemini_flash_seed{seed}"
        globals()[var_name] = hallucination_detection_config(
            model_name="gemini2.5-flash",
            dataset="RAGTruth",
            tasktype=task,
            splits=["test"],
            prompt_name="baseline_not_too_strict",
            seed=seed
        )

