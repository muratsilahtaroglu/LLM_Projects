import random
import datasets
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from semantic_search.semantic_search import SemanticSearch
import llm_pre_processing.file_operations_utils as fo_utils
from rag_benchmark_analysis.rag_evaluation.rag_benchmark import RAGBenchmark
from rag_benchmark_analysis.rag_evaluation.rag_metrics import RAGMetricsEvaluator

def main():
    random.seed(10)

    # model_name = "llama3.3:70b"
    model_name = "gemma2:27b"

    rag_benchmark = RAGBenchmark()

    rag_metrics_evaluator = RAGMetricsEvaluator(model_name=model_name)

    collection_names = ["youtube", "twitter", "pdf6"]

    # Convert JSON to Hugging Face Dataset
    dataset_path = "edited_data/test_clusters.json"
    eval_dataset = rag_benchmark.convert_json_to_dataset(dataset_path)


    processed_topics = list({output["cluster_name"] for output in eval_dataset})
    cluster_name = processed_topics[2]
    eval_dataset = [a["data"] for a in eval_dataset if a["cluster_name"] == cluster_name]
    eval_dataset = datasets.Dataset.from_list(eval_dataset)

    # Load combinations_with_range.json
    combinations_file_path = "edited_data/semantic_search_combinations_with_range2.json"
    test_settings_list = eval(fo_utils.read_textual_file(file_path=combinations_file_path))

    llm_score_results_path = f"llm_score_results_{cluster_name}_{model_name}.json"

    # Randomly select 20 test settings
    # random_selected_test_settings = random.sample(test_settings_list, k=3)

    knowledge_index = SemanticSearch(all_query_and_info_file="AdvencedRagLLM/semantic_search_api/demo_predictors.json", collection_name="")

    # Storage for results   
    results = []

    # test_settings_list = test_settings_list[2785:]

    # Run RAG tests for each selected test setting
    for test_setting in test_settings_list:
        test_index = test_setting.get("index")

        # Set unique output file for each test setting
        output_path = f"rag_eval_cluster_{cluster_name}_{model_name}.json"

        print(f"Running RAG tests for test setting index: {test_index}", flush=True)

        # Run RAG tests
        rag_benchmark.run_rag_tests(
            eval_dataset=eval_dataset,
            knowledge_index=knowledge_index,
            collection_name=collection_names,
            output_file=output_path,
            processing_column="topic",
            test_settings=test_setting,
            verbose=False
        )

        # Evaluate answers
        print(f"Evaluating answers for test setting index: {test_index}", flush=True)
        
        rag_metrics_evaluator.evaluate_answers(answer_path=output_path, evaluator_name=model_name, llm_score_results_path=llm_score_results_path)




if __name__ == "__main__":
    main()