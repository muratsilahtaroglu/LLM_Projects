# Evaluation Module

The `eval` directory is the evaluation suite for the LongLlmQlora project. It provides scripts and utilities to assess the performance of language models (LLMs) across various tasks, including generation, language modeling, benchmark datasets, and specialized tasks. Each script targets specific evaluation metrics to measure the capabilities of the LLMs.

## Directory Structure

```plaintext
eval/
├── eval_generation.py    # Evaluate generative capabilities.
├── eval_lm.py            # Language modeling evaluation.
├── eval_mmlu.py          # Multi-task benchmark for understanding and reasoning.
├── eval_passkey.py       # Passkey retrieval evaluation.
├── eval_infbench.py      # InfBench evaluations.
├── eval_longbench.py     # LongBench evaluations.
├── eval_needle.py        # Needle-in-a-haystack evaluations.
├── eval_topic.py         # Topic retrieval evaluation.
├── infbench_utils.py     # Utilities for InfBench evaluation.
├── longbench_utils.py    # Utilities for LongBench evaluation.
└── README.md             # Documentation for the eval module.
```

## Supported Tasks and Metrics

### Supported Tasks:

1. **Generation Tasks**: Measure the LLM’s ability to generate coherent, accurate, and context-aware outputs.
2. **Language Modeling**: Assess perplexity to evaluate the quality of the model’s predictions.
3. **Benchmark Evaluations**:
   - InfBench: Evaluate inference and reasoning.
   - LongBench: Long-context understanding and reasoning.
   - MMLU: Multi-task language understanding.
4. **Specialized Evaluations**:
   - Needle-in-a-haystack: Retrieval in extensive contexts.
   - Passkey Retrieval: Extract key information from lengthy passages.
   - Topic Retrieval: Retrieve the first topic from conversations.

### Metrics:

- **Perplexity**: Evaluates the model's ability to predict text.
- **Accuracy**: Measures the proportion of correct predictions.
- **F1 Score**: Harmonic mean of precision and recall, used for QA tasks.
- **ROUGE**: Measures the overlap between generated and reference text.
- **Fuzzy Matching**: Measures similarity between retrieved and reference values.
- **GPT Evaluation**: Accuracy based on GPT-4o assessments. # Will be added local LLM assessments

## Python Files Overview

### 1. `eval_generation.py`

Evaluates the generative capabilities of LLMs by comparing generated outputs against reference texts. Metrics like ROUGE are used to assess the quality of generations.

**Key Features:**

- Supports minimum and maximum token lengths.
- Allows customizable metrics such as ROUGE.
- Handles large datasets using efficient batching.

### 2. `eval_lm.py`

Evaluates language modeling using perplexity. Measures how well the model predicts a sequence of tokens.

**Key Features:**

- Evaluates perplexity over a specified dataset.
- Supports stride-based processing for streaming evaluations.
- Handles datasets with varying sample lengths.

### 3. `eval_mmlu.py`

Evaluates the model on the MMLU (Massive Multitask Language Understanding) benchmark. Tests the model's reasoning, understanding, and knowledge across diverse topics.

**Key Features:**

- Few-shot learning support.
- Categorizes tasks into STEM, Humanities, Social Sciences, and Others.
- Computes accuracy for each category.

### 4. `eval_passkey.py`

Tests the model’s ability to extract specific information (e.g., passkeys) embedded within extensive contexts.

**Key Features:**

- Supports evaluation over varying context lengths and passkey depths.
- Uses metrics like accuracy and fuzzy matching.
- Generates heatmaps to visualize performance over different context lengths and depths.

### 5. `eval_infbench.py`

Evaluates reasoning and inference capabilities using InfBench datasets. Tasks include QA, summarization, and retrieval.

**Key Features:**

- Supports truncation from the middle for lengthy inputs.
- Uses task-specific templates and metrics.
- Handles multiple tasks and generates aggregated scores.

### 6. `eval_longbench.py`

Assesses the model’s ability to handle long-context tasks using the LongBench dataset.

**Key Features:**

- Truncates inputs intelligently to fit the model's context window.
- Supports tasks like multi-document QA and summarization.
- Computes ROUGE and task-specific metrics.

### 7. `eval_needle.py`

Evaluates the model's ability to locate specific information (needle) within extensive contexts (haystack).

**Key Features:**

- Generates context with embedded target information (needle).
- Evaluates retrieval using ROUGE and GPT-based scoring.
- Provides visualizations for accuracy and context depth.

### 8. `eval_topic.py`

Tests the model’s performance in retrieving topics from conversational datasets.

**Key Features:**

- Focuses on topic retrieval from multi-turn conversations.
- Evaluates using accuracy and F1 scores.
- Groups instances by the number of topics for detailed analysis.

### 9. `infbench_utils.py`

Utility functions for processing InfBench tasks. Handles task-specific templates, scoring functions, and dataset preparation.

**Details:**

- **Normalization Functions:** Includes language-specific normalization for English, and Turkish.
- **Scoring Functions:** Provides methods to compute F1, ROUGE, and other task-specific scores.
- **Task Management:** Supports loading JSONL files and iterating over tasks efficiently.

### 10. `longbench_utils.py`

Utility functions for processing LongBench tasks. Includes template applications, scoring functions, and task categorization.

**Details:**

- **Normalization:** Turkish-specific normalization to handle language nuances.
- **Scoring:** Supports task-specific scoring such as QA, summarization, and classification.
- **Dataset2Metric Mapping:** Maps each dataset to appropriate metrics for seamless integration.

## Usage

### Prerequisites

- Python 3.10+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Running an Evaluation

Each script can be run independently. For example, to evaluate generation:

```bash
python eval_generation.py --eval_data <path_to_eval_data> --output_dir <output_directory>
```

### Common Arguments

- `--eval_data`: Path to the evaluation dataset.
- `--output_dir`: Directory to save evaluation results.
- `--batch_size`: Batch size for evaluation.
- `--metrics`: List of metrics to compute (e.g., ROUGE, accuracy).

### Example

Evaluate a model on InfBench:

```bash
python eval_infbench.py --tasks longbook_qa_eng --output_dir ./results/infbench
```

## Results and Visualization

- Results are saved in JSON format in the specified output directory.
- Heatmaps and other visualizations are generated for certain tasks (e.g., passkey, needle).

## Future Enhancements

- Adding more benchmark datasets.
- Incorporating advanced metrics like BLEU and METEOR.
- Support for real-time evaluation pipelines.

## Contact

For questions or contributions, contact:

- **Email**: [muratsilahtaroglu13@gmail.com](mailto:muratsilahtaroglu13@gmail.com)


