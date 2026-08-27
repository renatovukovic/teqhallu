# teqhallu
Code for the paper **Leveraging Low-Level Symbolic Competences for Unsupervised Grounding in Hallucination Detection**.

The repository contains inference pipelines and evaluation code for hallucination detection on the RAGTruth and DiaHalu benchmarks. The main experiment workflow is configured for RAGTruth and uses Gemini 2.5 Flash by default.

## Repository layout

```text
Data/
	ragtruth_data/                 RAGTruth JSONL files
	DiaHalu-main/                  DiaHalu benchmark and documentation
src/
	pyproject.toml                 uv project definition and dependencies
	inference/                     configs, inference code, prompts, and results
	evaluation/                    evaluation modules
	data_processing/               RAGTruth loading and preprocessing
	run_hallucination_detection.sh
	run_evaluation.sh
```

Inference results are saved as `.pt` files in `src/inference/results/`. Runtime logs are written to `src/logs/` when a logging-enabled runner is used.

## Requirements

- macOS or Linux
- Python 3.13 or newer
- [`uv`](https://docs.astral.sh/uv/)
- Access credentials for the model provider used by the selected config

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The project metadata is in `src/pyproject.toml`, not at the repository root. Run `uv` commands from `src`:

```bash
cd path/to/teqhallu/src
uv sync --dev
```

`uv sync` creates or updates the project environment from `pyproject.toml`. To run a command in that environment, use `uv run`:

```bash
uv run python --version
uv run python -u -m inference.hallucination_detection_inference --help
```

The shell scripts change to their own directory before running, so they can also be launched from the repository root. When running modules manually, use `src` as the working directory.

## Authentication

### Gemini 2.5 Flash

The Gemini and Vertex AI Model-as-a-Service configs require a Google Cloud project ID and Application Default Credentials. Configure credentials in the environment where the experiment is run, for example:

```bash
gcloud auth application-default login
```

Create `src/.env` with the provider-specific values:

```dotenv
VERTEX_AI_PROJECT_ID=your-google-cloud-project-id
OPENAI_ORGANIZATION=your-openai-organization-id
OPENAI_PROJECT_ID=your-openai-project-id
OPENAI_PROJECT_API_KEY=your-openai-project-api-key
```

Only the variables required by the selected model family need to be set. The account must have permission to use the selected Vertex AI model.

### Llama, Qwen, and GPT-OSS on Vertex AI MaaS

These configs use Vertex AI Model-as-a-Service endpoints and Google Application Default Credentials. The selected model must be enabled in Vertex AI Model Garden and its terms/EULA must be accepted. The inference code selects the endpoint region from the config name: Llama uses `us-central1`, Qwen uses `us-south1`, and other open models use `us-central1`.

### OpenAI GPT configs

GPT configs require `OPENAI_ORGANIZATION`, `OPENAI_PROJECT_ID`, and `OPENAI_PROJECT_API_KEY` in `src/.env`. The inference code loads this file from the `src` working directory.

Do not commit `.env` or expose credentials in logs or result files.

## RAGTruth data

The expected files are already located at:

```text
Data/ragtruth_data/response.jsonl
Data/ragtruth_data/source_info.jsonl
```

The preprocessing functions in `src/data_processing/ragtruth_processing.py` load the two files, join responses to source information, and group records by task type. The supported task types are `QA`, `Summary`, and `Data2txt`; the current configs evaluate the `test` split.

## Running paper experiments

All configuration objects are defined in `src/inference/hallucination_detection_configs.py`. The current runner contains these six Gemini RAGTruth configurations:

| Method | Task configurations |
| --- | --- |
| Baseline | `ragtruth_test_qa_baseline_gemini_flash_config`, `ragtruth_test_summary_baseline_gemini_flash_config`, `ragtruth_test_data2text_baseline_gemini_flash_config` |
| SQLite pipeline with column values and direct prediction | `ragtruth_test_qa_sqlite_pipeline_plus_direct_value_examples_gemini_flash_config`, `ragtruth_test_summary_sqlite_pipeline_plus_direct_value_examples_gemini_flash_config`, `ragtruth_test_data2text_sqlite_pipeline_plus_direct_value_examples_gemini_flash_config` |

Run all six inference experiments sequentially:

```bash
cd path/to/teqhallu/src
./run_hallucination_detection.sh
```

A single configuration can be run directly:

```bash
uv run python -u -m inference.hallucination_detection_inference \
	--config_name ragtruth_test_qa_baseline_gemini_flash_config
```

The inference script writes the result to:

```text
src/inference/results/<config_name>_responses.pt
```

SQLite-based configurations also create per-configuration databases under `src/sql_interpretation/databases/`. Prompts are selected from `src/prompts/` using each config's `prompt_name`.

## Other configurations

Use `src/inference/hallucination_detection_configs.py` as the source of truth. `get_config("<name>")` resolves every named configuration used by the inference and evaluation commands. The file contains the following RAGTruth method families:

- Gemini: baseline, less-strict baseline, SQLite, SQLite with column value examples, SQLite with column value examples plus direct prediction, ontology-based ID selection, two-step SQL, two-step subset, multistep atomic claims, and JSON key-value.
- Llama, Qwen, and GPT-OSS: baseline, less-strict baseline, SQLite with column value examples, and SQLite with column value examples plus direct prediction.
- Seed experiments: generated configurations for seeds `1` through `5` and tasks `QA`, `Summary`, and `Data2txt` are defined in `src/inference/hallucination_detection_seed_configs.py` and imported by the main config module.
- DiaHalu: configurations are defined in `src/inference/hallucination_detection_configs_dialhalu.py` and imported by the main config module.

To discover exact names without opening the file:

```bash
grep -E '^ragtruth_test_.*_config[[:space:]]*=' src/inference/hallucination_detection_configs.py
grep -E '^dialhalu_test_.*_config[[:space:]]*=' src/inference/hallucination_detection_configs_dialhalu.py
```

To switch the paper runner to another experiment, edit the `config_names` array in `src/run_hallucination_detection.sh`. Each selected name must be defined in `hallucination_detection_configs.py` or imported there from one of the companion config files.

## Evaluating results

After the corresponding `.pt` files have been generated, evaluate the six active RAGTruth configurations with:

```bash
cd path/to/teqhallu/src
./run_evaluation.sh
```

The evaluator can also be called directly with one or more config names:

```bash
uv run python -u -m evaluation.evaluate_ragtruth \
	--config_name ragtruth_test_qa_baseline_gemini_flash_config
```

Optional path arguments are available when the data or results are stored elsewhere:

```bash
uv run python -u -m evaluation.evaluate_ragtruth \
	--config_name ragtruth_test_qa_baseline_gemini_flash_config \
	--dataset_directory ../Data/ragtruth_data \
	--results_directory inference/results
```

For each config, `evaluation/evaluate_ragtruth.py` reports accuracy, precision, recall, F1, ROC-AUC when both classes are present, and the true-negative/false-positive/false-negative/true-positive counts. When multiple configs are supplied, it also prints a LaTeX-ready row containing per-config precision, recall, F1, and macro averages.

## DiaHalu

The DiaHalu benchmark file is at `Data/DiaHalu-main/DiaHalu_Bench.jsonl`. Its source documentation is in `Data/DiaHalu-main/README.md`. DiaHalu configs and the evaluator are separate from the RAGTruth workflow:

The DiaHalu runners use the eight Gemini Flash configurations below, defined in `src/inference/hallucination_detection_configs_dialhalu.py`:

| Method | Task configurations |
| --- | --- |
| Baseline | `dialhalu_test_reasoning_baseline_gemini_flash_config`, `dialhalu_test_world_knowledge_baseline_gemini_flash_config`, `dialhalu_test_task_oriented_baseline_gemini_flash_config`, `dialhalu_test_chit_chat_baseline_gemini_flash_config` |
| Full TeQHallu | `dialhalu_test_reasoning_sqlite_pipeline_plus_direct_value_examples_gemini_flash_config`, `dialhalu_test_world_knowledge_sqlite_pipeline_plus_direct_value_examples_gemini_flash_config`, `dialhalu_test_task_oriented_sqlite_pipeline_plus_direct_value_examples_gemini_flash_config`, `dialhalu_test_chit_chat_sqlite_pipeline_plus_direct_value_examples_gemini_flash_config` |

Run all eight DiaHalu inference experiments sequentially:

```bash
cd path/to/teqhallu/src
./run_hallucination_detection_dialhalu.sh
```

Evaluate the generated DiaHalu result files:

```bash
cd path/to/teqhallu/src
./run_evaluation_dialhalu.sh
```

To run one configuration directly, pass its exact name to the corresponding module:

```bash
uv run python -u -m inference.hallucination_detection_inference \
	--config_name dialhalu_test_reasoning_baseline_gemini_flash_config

uv run python -u -m evaluation.evaluate_dialhalu \
	--config_name dialhalu_test_reasoning_baseline_gemini_flash_config
```

Additional DiaHalu configurations for Llama, Qwen, GPT-OSS, Gemini 1.5 Pro, GPT-4, and the MultiWOZ ontology are also defined in `src/inference/hallucination_detection_configs_dialhalu.py` and can be run by supplying their exact names.

## Reproducibility notes

- Inference is performed against external model APIs; outputs can vary by provider, model version, credentials, and configuration.
- Gemini configurations without a seed use temperature `0`. Seeded configurations are generated separately in `hallucination_detection_seed_configs.py`.
- Existing `.pt` files in `src/inference/results/` can be evaluated without rerunning inference, provided their config names match the selected evaluator arguments.
- Keep API credentials outside version control and keep generated logs and databases with the experiment artifacts when preserving a run.

## Citation

Citation information will be added here once the paper's final bibliographic details are available.

```bibtex
% TODO: add the paper citation
```
