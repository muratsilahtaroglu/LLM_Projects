from itertools import combinations, product
import os
import numpy as np
import pandas as pd
from  rag_benchmark_analysis.ablation_experiment.ablation_on_ML import create_ablation_results

# Test script for ablation_on_ML.py
def test_ablation_on_ml():
    print("### Starting Ablation Test ###")

    # Parameters
    k = [5, 10, 20, 50, 100, 200, 500, 1000]
    coefficient_values = np.arange(0.6, 1.6, 0.15)
    threshold_score_values = np.arange(0.75, 1.05, 0.05)
    helper_types = ["sub_queries", "keywords", "helper_keywords", "keywords_join_list"]
    coefficient_helpers_values = [
        list(combo) for r in range(1, len(helper_types) + 1) for combo in combinations(helper_types, r)
    ]
    merge_type_values = ["sum", "proud", "square", "square_sum", "square_sum2", "square_proud", "square_proud2"]

    # Generate all combinations
    all_combinations = list(product(k, coefficient_values, threshold_score_values, coefficient_helpers_values, merge_type_values))

    # Simulate random scores
    np.random.seed(42)
    scores = np.random.rand(len(all_combinations))

    # Run the ablation function
    try:
        create_ablation_results(
            ablation_type="rag",
            scores=scores.tolist(),  # Convert to list as the function expects it
            all_combinations=all_combinations,
            metric_type="TestMetric",
            verion="TestVersion"
        )
        print("### Ablation Test Completed Successfully ###")
    except Exception as e:
        print(f"### Ablation Test Failed ###\n{e}")


if __name__ == "__main__":
    test_ablation_on_ml()
