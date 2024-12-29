# AblationExperiment

## Key Features

The **AblationExperiment** module focuses on evaluating the performance of machine learning models and understanding the effects of different feature combinations and parameters. The key functionalities include:

1. **Model Selection and Evaluation:**

   - Use K-Fold Cross-Validation to evaluate multiple models.
   - Automatically identify the best-performing model based on Mean Squared Error (MSE).

2. **Ablation Experiments:**

   - Generate combinations of features and parameters for ablation studies.
   - Simulate scores to test the effect of various parameters.

3. **Data Preparation and Feature Engineering:**

   - Process helper types and assign coefficients dynamically.
   - Create clean and compatible feature names for model training.

4. **Model Training and Visualization:**

   - Train and evaluate ML models such as Gradient Boosting, XGBoost, LightGBM, and CatBoost.
   - Visualize feature importance to understand parameter relevance.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Pip

### Steps

1. Clone the repository and navigate to the `ablation_experiment` module:

   ```bash
   git clone https://github.com/repo/rag_benchmark_analysis.git
   cd rag_benchmark_analysis/ablation_experiment
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

The following libraries are required to run the Ablation Experiment module:

- `numpy`: For numerical operations.
- `pandas`: For data manipulation and analysis.
- `matplotlib`: For data visualization and feature importance plotting.
- `scikit-learn`: For machine learning model selection, training, and evaluation.
- `xgboost`: For training gradient-boosted decision trees.
- `lightgbm`: For training gradient-boosted decision trees with high efficiency.
- `catboost`: For handling categorical features and training models.

### Installing Dependencies

To install the required libraries, use the following command:

```bash
pip install -r requirements.txt
```

---

## Running the Ablation Experiment Module

### 1. **Select the Best ML Model:**

- Use `__select_best_ML_model.py` to perform K-Fold Cross-Validation and select the best model.
- Example usage:
  ```python
  from __select_best_ML_model import ModelSelector

  models = {
      "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
      "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
  }

  selector = ModelSelector(models=models, k_folds=5)
  results = selector.evaluate_models(X_train, y_train)
  print(f"Best Model: {results['best_model_name']}")
  ```

### 2. **Generate and Train Models:**

- Use `ablasyon_deneme_alll_ML.py` to generate combinations, train models, and visualize results.
- Example usage:
  ```bash
  python ablasyon_deneme_alll_ML.py
  ```

### 3. **Run Comprehensive Ablation Studies:**

- Use `ablation_on_ML.py` to test various parameter combinations and evaluate models.
- Example usage:
  ```python
  from ablation_on_ML import create_ablation_results

  all_combinations = [...]  # Define feature and parameter combinations
  create_ablation_results(
      ablation_type="rag",
      all_combinations=all_combinations,
      metric_type="MSE",
      verion="V1"
  )
  ```

---

## Notable Files

### `__select_best_ML_model.py`

- Implements K-Fold Cross-Validation.
- Dynamically selects the best-performing model based on Mean Squared Error.

### `ablasyon_deneme_alll_ML.py`

- Generates parameter combinations and trains multiple models.
- Visualizes feature importance for trained models.

### `ablation_on_ML.py`

- Conducts comprehensive ablation experiments.
- Simulates scores and evaluates the impact of different feature and parameter combinations.

### `ablation_results/`

- Stores results, including:
  - **Feature Importance Visualizations:** PNG files showing parameter relevance.
  - **Model Performance Metrics:** Results for evaluated models.

---

## Example Workflow

1. **Prepare Data:**

   - Use the provided scripts to clean and preprocess your dataset.

2. **Run Model Selection:**

   - Use `__select_best_ML_model.py` to identify the best-performing model.

3. **Perform Ablation Study:**

   - Use `ablation_on_ML.py` to evaluate the effects of various parameters and feature combinations.

4. **Analyze Results:**

   - Visualize feature importance and inspect performance metrics stored in the `ablation_results/` directory.

---

## Contributors

Developed by: Murat Silahtaroğlu\
Contact: [muratsilahtaroglu13@gmail.com](mailto\:muratsilahtaroglu13@gmail.com)

