# RAGBenchmarkAnalysis

## Project Overview

**`RAGBenchmarkAnalysis`** is a Python-based project designed for advanced evaluation and experimentation in various aspects of clustering, machine learning (ML) ablation experiments, and retrieval-augmented generation (RAG) benchmarking. The purpose of the project is to:

- Identify the best metrics for a given RAG system task. This involves determining the optimal parameters for the task definition that the RAG system will use.
- Provide tools for clustering textual data.
- Evaluate and select the best-performing ML models using ablation experiments.
- Benchmark and evaluate RAG systems.
- Generate and visualize evaluation metrics like precision, recall, nDCG, and cosine similarity.

## Folder Structure

```bash
rag_benchmark_analysis/
├── clustering/                   # Tools for clustering textual datasets.
│   ├── dataset_clustering.py     # Main class for dataset clustering using DBSCAN.
│   ├── dataset_clustering_results/  # Folder to store clustering results.
│   └── test/                     # Test scripts for clustering.
│       └── evaluation_data_clustering.py  # Evaluates the clustering results.
├── ablation_experiment/          # Tools for ML ablation experiments.
│   ├── ablation_results/         # Folder to store results from ablation experiments.
│   ├── __select_best_ML_model.py # Selects the best ML model using K-Fold validation.
│   ├── ablasyon_deneme_alll_ML.py  # Simulates ablation results and trains models.
│   └── ablation_on_ML.py         # Performs and evaluates ablation experiments.
├── rag_evaluation/               # Tools for RAG evaluation and benchmarking.
│   ├── results/                  # Folder to store RAG evaluation results.
│   ├── test/                     # Test scripts for RAG evaluation.
│   │   └── evaluation_rag_benchmark_metrics.py  # Evaluates RAG metrics.
│   ├── .env                      # Environment variables for API keys.
│   ├── rag_benchmark.py          # Benchmarking and evaluation of RAG systems.
│   ├── rag_evaluation_prompts.py # Prompts and criteria for RAG evaluations.
│   └── rag_metrics.py            # Metric calculations for RAG evaluations.
```

## Key Features
- Tools for clustering textual data using DBSCAN.
- Conduct ablation experiments to evaluate ML models.
- Perform benchmarking and evaluation of RAG systems.
- Comprehensive metrics for precision, recall, nDCG, and cosine similarity.

## Installation

### Prerequisites
- Python 3.8 or higher
- Pip

### Steps
1. Clone the repository:
```bash
git clone https://github.com/repo/rag_benchmark_analysis.git
cd rag_benchmark_analysis
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

4. Set up environment variables:
   - Add your API keys (e.g., `OPENAI_API_KEY`) to the `.env` file.

## Running the Services

### Activating the Virtual Environment
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Notable Files

### Clustering
- **`dataset_clustering.py`**: Implements DBSCAN clustering on textual data.
- **`evaluation_data_clustering.py`**: Evaluates clustering results.

### Ablation Experiment
- **`__select_best_ML_model.py`**: Selects the best ML model using K-Fold validation.
- **`ablasyon_deneme_alll_ML.py`**: Simulates ablation results and trains ML models.
- **`ablation_on_ML.py`**: Conducts ablation experiments and visualizes results.

### RAG Evaluation
- **`rag_benchmark.py`**: Benchmarks and evaluates RAG systems.
- **`rag_metrics.py`**: Contains utilities for calculating evaluation metrics.
- **`rag_evaluation_prompts.py`**: Provides prompts and scoring criteria.

## Contributors

Developed by: Murat Silahtaroğlu\
Contact: [muratsilahtaroglu13@gmail.com](mailto\:muratsilahtaroglu13@gmail.com)
