# RAGEvaluation

## Project Overview

The **RAGEvaluation** module provides tools and scripts for evaluating Retrieval-Augmented Generation (RAG) systems. This evaluation suite enables benchmarking the performance of RAG-based pipelines using various metrics such as Precision, Recall, F1-Score, ROUGE-L, Cosine Similarity, and nDCG. Additionally, it offers prompt-based relevance evaluation using pretrained language models.

The primary objectives include:
- Evaluating the relevance of retrieved documents.
- Assessing the quality of answers produced by RAG pipelines.
- Generating detailed evaluation metrics.
- Benchmarking semantic search and reranking methods.

## Key Features
- **Semantic Search Integration:** Retrieve relevant documents using a knowledge index.
- **RAG-Based Answer Evaluation:** Evaluate relevance using both pretrained models and custom metrics.
- **Metric Calculations:** Precision, Recall, F1-Score, ROUGE-L, Cosine Similarity, and nDCG.
- **Prompt-Based Evaluation:** Highly customizable prompts for scoring relevance.

## Installation

### Prerequisites
- Python 3.8 or higher
- Hugging Face `datasets`
- Sentence Transformers
- Pretrained RAG models

### Steps
1. Clone the repository:
```bash
git clone https://github.com/repo/rag_benchmark_analysis.git
cd rag_benchmark_analysis/rag_evaluation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure the environment variables:
   - Update `.env` with your API keys (e.g., OpenAI API key).
```bash
OPEN_AI_API_KEY=your_openai_api_key_here
```

## Running the Services

### Activating the Virtual Environment
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Running RAG Evaluation
1. **RAG Benchmarking:**
   Run the `rag_benchmark.py` script to evaluate RAG pipelines.
   ```bash
   python rag_benchmark.py
   ```

2. **Prompts for Evaluation:**
   Customize evaluation prompts in `rag_evaluation_prompts.py` to match your specific task or criteria.

3. **Metrics Calculation:**
   Use `rag_metrics.py` to compute evaluation metrics such as ROUGE-L, Cosine Similarity, Precision, Recall, and nDCG.

## Notable Files

### Main Scripts
- **`rag_benchmark.py`**:
  - Runs the main RAG evaluation pipeline.
  - Converts datasets, retrieves relevant documents, and evaluates relevance using pretrained models.

- **`rag_metrics.py`**:
  - Contains utilities for calculating evaluation metrics such as Precision, Recall, F1-Score, ROUGE-L, Cosine Similarity, and nDCG.

- **`rag_evaluation_prompts.py`**:
  - Contains customizable prompts for evaluating retrieved document relevance.

### Environment Configuration
- **`.env`**: Stores API keys and environment variables for accessing external services such as OpenAI or local LLMs.

## Usage Examples

### Running RAG Tests

```python
from rag_benchmark import RAGBenchmark
from semantic_search.semantic_search import SemanticSearch
from datasets import load_dataset

# Initialize RAG Benchmark
rag_benchmark = RAGBenchmark()

# Load Evaluation Dataset
eval_dataset = load_dataset("json", data_files="path_to_eval_dataset.json", split="train")

# Set Knowledge Index
knowledge_index = SemanticSearch("path_to_knowledge_index")

# Run RAG Evaluation
rag_benchmark.run_rag_tests(
    eval_dataset=eval_dataset,
    knowledge_index=knowledge_index,
    collection_name=["collection_1", "collection_2"],
    output_file="rag_evaluation_results.json",
    processing_column="topic",
)
```

### Metrics Evaluation

```python
from rag_metrics import RAGMetricsEvaluator

# Initialize Evaluator
metrics_evaluator = RAGMetricsEvaluator(model_name="local_llm_model")

# Calculate Cosine Similarity
similarity_scores = metrics_evaluator.calculate_cosine_similarity(
    retrieved_documents=["doc1", "doc2"],
    topic="example topic"
)

# Calculate ROUGE-L Scores
rouge_scores = metrics_evaluator.calculate_rouge_l_score(
    retrieved_documents=["doc1", "doc2"],
    response="example response"
)
```

## Contributors

Developed by: Murat Silahtaroğlu\
Contact: [muratsilahtaroglu13@gmail.com](mailto\:muratsilahtaroglu13@gmail.com)