# Clustering

## Key Features

The **Clustering** module provides comprehensive tools for grouping and analyzing similar data points. Designed for diverse datasets, it integrates advanced embedding techniques and clustering algorithms to offer the following:

1. **Data Embedding:**
   - Uses pre-trained SentenceTransformer models for high-quality textual embeddings.

2. **Clustering Optimization:**
   - Determines optimal clustering parameters (e.g., DBSCAN `eps`, `min_samples`) using Silhouette Scores.

3. **Visualization:**
   - Generates t-SNE plots for visualizing cluster separations.
   - Provides k-distance graphs for selecting DBSCAN parameters.

4. **Data Summarization:**
   - Produces detailed summaries of cluster content, including counts and unique values.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Pip

### Steps

1. Clone the repository and navigate to the `clustering` module:

   ```bash
   git clone https://github.com/repo/rag_benchmark_analysis.git
   cd rag_benchmark_analysis/clustering
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Requirements

The following libraries are required to run the Clustering module:

- `numpy`: Numerical computations.
- `pandas`: Data manipulation and analysis.
- `matplotlib`: Data visualization (k-distance and t-SNE plots).
- `seaborn`: Enhanced visualizations for clustering results.
- `scikit-learn`: Clustering and evaluation (DBSCAN, Silhouette Scores).
- `sentence-transformers`: Pre-trained embedding models for textual data.

### Installing Dependencies

To install the required libraries, use the following command:

```bash
pip install -r requirements.txt
```

---

## Running the Clustering Module

### 1. **Generate Embeddings:**

- Use the `generate_embedding` method in `dataset_clustering.py` to create embeddings for your dataset.
- Example usage:

  ```python
  from dataset_clustering import DataClustering

  clustering = DataClustering()
  data = ["example text 1", "example text 2"]
  embeddings = clustering.generate_embedding(data)
  ```

### 2. **Find Optimal Clustering Parameters:**

- Use the `get_best_metrics` method to determine the best `eps` and `min_samples` values for DBSCAN.
- Example usage:

  ```python
  metrics = clustering.get_best_metrics()
  print(metrics)
  ```

### 3. **Perform Clustering:**

- Use the `perform_clustering` method to cluster data using optimal or custom parameters.
- Example usage:

  ```python
  df, parameters = clustering.perform_clustering(metrics=metrics[0])
  print(df.head())
  ```

### 4. **Visualize Clusters:**

- Use the `save_cluster_visualization` method to generate t-SNE plots.
- Example usage:

  ```python
  clustering.save_cluster_visualization("tsne_plot.png")
  ```

### 5. **Summarize Clusters:**

- Use the `save_cluster_summary` method to save a detailed cluster summary.
- Example usage:

  ```python
  clustering.save_cluster_summary("cluster_summary.json", metrics[0])
  ```

### 6. **Evaluate Clustering:**

- Use `evaluation_data_clustering.py` to preprocess and evaluate clustering results on specific datasets.
- Example usage:

  ```python
  from evaluation_data_clustering import generate_data_clustering_projects

  task_name = "youtube"  # Options: 'rag', 'tweet', 'pdf', 'twitter', 'youtube'
  response_list = generate_data_clustering_projects(task_name)

  for response in response_list:
      print(response)
  ```

#### Predefined Dataset Tasks

1. **YouTube Data:**
   - Prepares and clusters titles from a YouTube dataset.
2. **Tweet Data:**
   - Processes tweets, removes duplicates, and clusters textual data.
3. **PDF Data:**
   - Clusters titles extracted from PDF documents.
4. **RAG Data:**
   - Clusters topics from a RAG dataset.

---

## Notable Files

### `dataset_clustering.py`

- Core clustering functionalities, including embedding generation, parameter optimization, and clustering.

### `evaluation_data_clustering.py`

- Preprocessing and evaluation utilities for clustering projects. Examples of usage include evaluating datasets like tweets or YouTube titles.

### `dataset_clustering_results/`

- Directory to store clustering outputs, including:
  - **t-SNE Plots:** Visual representations of clusters.
  - **Cluster Summaries:** JSON files detailing cluster compositions.

---

## Example Workflow

1. **Prepare Data:**
   - Load and preprocess your dataset using the provided scripts.

2. **Generate Embeddings:**
   - Use `generate_embedding` to transform textual data into embeddings.

3. **Optimize Parameters:**
   - Use `get_best_metrics` to find optimal DBSCAN parameters.

4. **Cluster Data:**
   - Use `perform_clustering` to group similar data points.

5. **Visualize Results:**
   - Save and inspect t-SNE plots for cluster visualization.

6. **Summarize Results:**
   - Save detailed summaries of clusters for further analysis.

7. **Evaluate Clustering:**
   - Use `evaluation_data_clustering.py` to test the clustering results on predefined datasets.

---

## Contributors

Developed by: Murat Silahtaroğlu\
Contact: [muratsilahtaroglu13@gmail.com](mailto\:muratsilahtaroglu13@gmail.com)