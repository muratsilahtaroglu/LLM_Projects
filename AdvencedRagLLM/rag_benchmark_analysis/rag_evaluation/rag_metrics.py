from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from typing import List, Dict
from openai import OpenAI
import uuid
import numpy as np
import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import llm_pre_processing.file_operations_utils as fo_utils

try:
    import rag_evaluation_prompts
except:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import rag_evaluation.rag_evaluation_prompts as rag_evaluation_prompts

import ollama_client

ollama_ai = ollama_client.OllamaClient()

class RAGMetricsEvaluator:
    def __init__(self, model_name:str):
        self.model_name = model_name


    def evaluator_ollama_llm(self, model_name:str, topic:str, content: str, text_topic: str,count:int=1):

        """
        Evaluate the relevance of a given content to a topic using a local LLM.

        Args:
            model_name (str): The name of the model to use for evaluation.
            topic (str): The topic to evaluate against.
            content (str): The content to evaluate.
            text_topic (str): The text topic to evaluate against.
            count (int, optional): The number of evaluation requests to send. Defaults to 1.

        Returns:
            str: The evaluation result, or None if no responses were received.
        """
        parsed_data = []

        base_prompt = rag_evaluation_prompts.evaluation_base_prompt.format(
                topic=text_topic,
                content=content
        )

        system_prompt = rag_evaluation_prompts.evaluation_system_prompt
        FULL_PROMPT  = system_prompt + "\n" + base_prompt


        m = round(count/5) if round(count/5)>0 else count
        multi_threading_count = min(m, 5)


        parameters = {}
        for _ in range(multi_threading_count):
            ai_uudi = str(uuid.uuid4())
            parameters.update({ai_uudi: {"model_name": model_name, "prompt": FULL_PROMPT, "task": "","options":None}})
        responses = ollama_ai.get_all_responses(parameters, same_task=True)
        
        if len(responses):
            for ai_uudi, response in responses.items():
                if isinstance(response, list):
                    parsed_data.extend(response)
                else:
                    parsed_data.append(response)
        
        else:
            return None
        return parsed_data[0]


    def evaluator_open_ai(self, topic:str, content:str) -> tuple:
        """
        Evaluate the relevance of a given content to a topic using the OpenAI Chat Model.

        Args:
            topic (str): The topic to evaluate against.
            content (str): The content to evaluate.

        Returns:
            tuple: A tuple containing the evaluation result and the number of tokens used.
        """

        api_key=os.getenv("OPENAI_API_KEY")
        eval_chat_model = OpenAI(api_key=api_key)
        eval_prompt = rag_evaluation_prompts.evaluation_base_prompt.format(
                topic=topic,
                content=content
        )

        system_prompt = rag_evaluation_prompts.evaluation_system_prompt
        
        #token sayılarını da hesapla
        response = eval_chat_model.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": eval_prompt} 
        ],
        max_tokens=2000,
        temperature=0.1, 
        top_p=1.0
        )
        usage_data = response.usage
        tokens_used = usage_data.total_tokens
        eval_result = response.choices[0].message.content
        return eval_result, tokens_used


    def evaluate_answers(self, answer_path: str, evaluator_name: str, llm_score_results_path: str) -> None:
        """Evaluate answers and save updated results in bulk."""
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_file_folder = os.path.join(current_dir,"results")
        answer_path = os.path.join(output_file_folder,answer_path)
        llm_score_results_path = os.path.join(output_file_folder,llm_score_results_path)
        
        # Load evaluated answers
        answers = eval(fo_utils.read_textual_file(file_path=answer_path)) if os.path.exists(answer_path) else []
        llm_score_results = eval(fo_utils.read_textual_file(file_path=llm_score_results_path))if os.path.exists(llm_score_results_path) else {}

        total_tokens_used = 0
        total_requests = 0

        for experiment in (answers):
            if len(experiment["retrieved_docs"]) == 0 or f"eval_score_{evaluator_name}" in experiment:
                continue  # Skip if already evaluated

            topic = experiment["topic"]
            retrieved_docs = experiment["retrieved_docs"]
            llm_eval_score = []
            llm_eval_feedback = []

            if topic not in llm_score_results:
                llm_score_results[topic] = {"retrieved_docs": {}}

            for doc in retrieved_docs:
                if doc in llm_score_results[topic]["retrieved_docs"]:
                    score = llm_score_results[topic]["retrieved_docs"][doc].get(f"val_score_{evaluator_name}")
                    feedback = None
                else:
                    if "gpt" in self.model_name.lower():
                        eval_result, tokens_used = self.evaluator_open_ai(topic, doc)
                        total_tokens_used += tokens_used
                        total_requests += 1
                    else:
                        eval_result = self.evaluator_ollama_llm(
                            model_name=self.model_name,
                            topic="evaluate_documents",
                            content=doc,
                            text_topic=topic,
                            count=1,
                        )
                    try:
                        feedback, score = [item.strip() for item in eval_result.split("[RESULT]")]
                        score = int(score)
                    except:
                        print(f"Unexpected format in eval_result: {eval_result}", flush=True)
                        continue

                    llm_score_results[topic]["retrieved_docs"][doc] = {
                        f"val_score_{evaluator_name}": score,
                    }

                if score is not None:
                    llm_eval_score.append(score)
                if feedback:
                    llm_eval_feedback.append(feedback)

            experiment[f"eval_score_{evaluator_name}"] = llm_eval_score
            experiment[f"eval_feedback_{evaluator_name}"] = llm_eval_feedback
        
        fo_utils.write_textual_file(data=answers,path=answer_path)
        fo_utils.write_textual_file(data=llm_score_results,path=llm_score_results_path)

        if total_tokens_used > 0 and total_requests > 0:
            print(f"Total tokens used: {total_tokens_used}")
            print(f"Total requests: {total_requests}")
            print(f"Average tokens per request: {total_tokens_used / total_requests:.2f}")


    def calculate_cosine_similarity(self, retrieved_documents:List[str], topic:str) -> float:

        model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')  # You can change the model

        # Encode the topic_embedding and retrieved_documents using the SentenceTransformer
        topic_embedding = model.encode(topic, convert_to_tensor=True)
        retrieved_documents_embeddings = model.encode(retrieved_documents, convert_to_tensor=True)
        
        # Compute cosine similarity between response and each content combination
        cosine_scores = util.pytorch_cos_sim(topic_embedding, retrieved_documents_embeddings)

        return [score.item() for score in cosine_scores[0]]


    def calculate_rouge_l_score(self, retrieved_documents:List[str], response: str) -> List[float]:    
        
        """
        Calculates ROUGE-L scores for a given response and retrieved documents.
        
        Args:
            retrieved_documents (List[str]): List of retrieved documents.
            response (str): Response to evaluate.
            
        Returns:
            float: Maximum ROUGE-L score among the retrieved documents.
        """
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = []
        
        for i, doc in enumerate(retrieved_documents):
            rouge_score = scorer.score(doc, response)
            scores.append(rouge_score['rougeL'].fmeasure)
        return scores
    

    def get_ground_truth_documents(self, result_file_path: str, topic: str, score_key: str, cutoff: float = None, dynamic_percentile: bool = True) -> List[str]:
        """
        Extract ground truth documents dynamically based on the score distribution for the topic.

        Args:
            result_file_path (str): Path to the results file.
            topic (str): The topic to process.
            score_key (str): The key for the evaluation score.
            dynamic_percentile (bool): Whether to dynamically calculate the cutoff.

        Returns:
            List[str]: List of ground truth documents.
        """

        current_dir = os.path.dirname(os.path.abspath(__file__))
        result_file_path = os.path.join(current_dir,"results", result_file_path)

        score_data = eval(fo_utils.read_textual_file(file_path=result_file_path)) if os.path.exists(result_file_path) else []
        topic_data = score_data.get(topic, {}).get("retrieved_docs", {})

        doc_scores = [(doc, scores.get(score_key, 0)) for doc, scores in topic_data.items()]

        if not doc_scores:
            return []

        # Extract scores
        scores = [score for _, score in doc_scores]

        # Determine cutoff
        if cutoff is not None:
            cutoff_score = cutoff
        elif dynamic_percentile:
            cutoff_score = np.percentile(scores, 50)  # Default to median
        else:
            cutoff_score = np.percentile(scores, 90)

        return [doc for doc, score in doc_scores if score >= cutoff_score]


    def calculate_precision_recall_f1(self, result_object: Dict, ground_truth_docs: List[str], k: int = 10) -> Dict:
        """
        Calculate Precision, Recall, and F1-score for semantic search results with a cap on k.

        Args:
            result_object (Dict): The result object containing retrieved documents.
            ground_truth_docs (List[str]): List of ground truth relevant documents.
            k (int): Maximum number of retrieved documents.
        
        Returns:
            Dict: A dictionary containing precision, recall, and F1-score.
        """
        retrieved_docs = result_object.get("retrieved_docs", [])[:k]  # Limit retrieved docs to k
        
        # Handle empty cases
        if not retrieved_docs:
            return {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}

        if not ground_truth_docs:
            return {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}

        # Calculate matches
        relevant_retrieved = len(set(retrieved_docs) & set(ground_truth_docs))
        total_retrieved = len(retrieved_docs)
        total_relevant = len(ground_truth_docs)

        # Edge Case: Single retrieved document
        if total_retrieved == 1:
            is_relevant = retrieved_docs[0] in ground_truth_docs
            precision = 1.0 if is_relevant else 0.0
            recall = 1.0 / len(ground_truth_docs) if is_relevant and ground_truth_docs else 0.0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            return {"precision": round(precision, 4), "recall": round(recall, 4), "f1-score": round(f1_score, 4)}
        
        # Adjusted Recall
        recall = min(relevant_retrieved / total_relevant, 1.0) if total_relevant > 0 else 0.0
        
        # Precision
        precision = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1-score": round(f1_score, 4),
        }
    

    # Define DCG
    def calculate_ndcg(self, scores, max_documents=10):
        """
        Calculate nDCG without excessive scaling. Introduce penalties for missing documents.
        Returns values in a natural range [0, 1].
        """
        if not scores:  # No scores
            return 0.0  # Worst possible score

        # Pad scores with zeros if fewer than max_documents
        padded_scores = scores + [0] * (max_documents - len(scores))

        # Ideal scores (sorted descending)
        ideal_scores = sorted(padded_scores, reverse=True)

        # Calculate DCG
        dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(padded_scores))
        idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_scores))

        # Avoid division by zero
        if idcg == 0:
            return 0.0

        # Normalized nDCG
        ndcg = dcg / idcg

        # Introduce penalty for missing or low scores
        missing_docs_penalty = (max_documents - len(scores)) / max_documents  # Missing doc ratio
        adjusted_ndcg = ndcg * (1 - missing_docs_penalty)

        # Clamp to [0, 1]
        return round(max(0.0, min(1.0, adjusted_ndcg)), 5)
    

    def calculate_weighted_average(self, scores: list) -> float:
        """
        Calculate the weighted average of a list of scores.

        Args:
            scores (List[float]): List of scores (relevance values).

        Returns:
            float: Weighted average score.
        """
        if not scores:  # Handle empty list
            return 0.0

        # Assign weights: Higher weights for earlier positions
        weights = [1 / (i + 1) for i in range(len(scores))]

        # Calculate the weighted sum
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))

        # Normalize by the sum of weights
        total_weight = sum(weights)

        return round(weighted_sum / total_weight, 5)

if __name__ == "__main__":
    model_name = "gemma2:27b"
    cluster_name = "29"

    input_file_path = f"/home/clone/clone/rag_benchmark_analysis/rag_evaluation/results/rag_eval_cluster_{cluster_name}_{model_name}2.json"
    output_file_path = f"/home/clone/clone/rag_benchmark_analysis/rag_evaluation/results/rag_evaluation_results_cluster_{cluster_name}_{model_name}_2.json"
    llm_evaluation_points_file_path = f"/home/clone/clone/rag_benchmark_analysis/rag_evaluation/results/llm_score_results_{cluster_name}_{model_name}_2.json" #_file_path
    
    evaluation_metrics = RAGMetricsEvaluator(model_name=model_name)

    # Load evaluated data
    evaluated_data = eval(fo_utils.read_textual_file(file_path=input_file_path))
    if not evaluated_data:
        print("No data found in the input file.")

    # Group results by unique test_settings index
    test_setting_groups = {}
    for data in evaluated_data:
        test_index = data["test_settings"]["index"]
        if test_index not in test_setting_groups:
            test_setting_groups[test_index] = []
        test_setting_groups[test_index].append(data)

    # Process each test_setting
    all_results = []
    for test_index, group_data in test_setting_groups.items():
        results = []

        for data in group_data:
            if len(data["retrieved_docs"]) == 0:
                continue
            score_key = f"eval_score_{model_name}"
            llm_evaluation_score_list= data[score_key]
            k = data["test_settings"]["k"]
            ground_truth_docs = evaluation_metrics.get_ground_truth_documents(result_file_path=llm_evaluation_points_file_path, topic=data["topic"], score_key=f"val_score_{model_name}")
            precision_recall_f1 = evaluation_metrics.calculate_precision_recall_f1(result_object=data, ground_truth_docs=ground_truth_docs, k=k)
            rouge_l_scores = evaluation_metrics.calculate_rouge_l_score(retrieved_documents=data["retrieved_docs"], response=data["topic"])
            average_rogue_l =  round(np.mean(rouge_l_scores), 4) if rouge_l_scores else 0.0
            cosine_sim_scores = evaluation_metrics.calculate_cosine_similarity(retrieved_documents=data["retrieved_docs"], topic=data["topic"])
            weighted_average = evaluation_metrics.calculate_weighted_average(llm_evaluation_score_list)
            ndcg_score = evaluation_metrics.calculate_ndcg(scores=llm_evaluation_score_list, max_documents=k)
            average_cosine_sim = round(np.mean(cosine_sim_scores), 4) if cosine_sim_scores else 0.0


            # Add metrics to the data object
            data["f1-score"] = precision_recall_f1["f1-score"]
            data["precision"] = precision_recall_f1["precision"]
            data["recall"] = precision_recall_f1["recall"]
            data["rouge-l"] = average_rogue_l
            data["cosine_sim"] = average_cosine_sim
            data["weighted_average"] = weighted_average
            data["ndcg_score"] = ndcg_score

            results.append(data)

        # Calculate average nDCG for the current test_setting
        avg_ndcg_score = 0
        total_ndcg_scores = sum(d["ndcg_score"] for d in results)
        if results:
            avg_ndcg_score = total_ndcg_scores / len(results)

        # Store the aggregated result for this test_setting
        all_results.append({
            "test_setting": group_data[0]["test_settings"],  # Use the first test_setting as reference
            "average_ndcg_score": avg_ndcg_score,
            "results": results
        })

    # Sort results by average nDCG score in descending order
    all_results = sorted(all_results, key=lambda x: x["average_ndcg_score"], reverse=True)

    # Save all results to a JSON file
    results_file_path = f"/home/clone/clone/rag_benchmark_analysis/rag_evaluation/results/rag_evaluation_results_cluster_{cluster_name}_{model_name}_2.json"
    fo_utils.write_textual_file(path=results_file_path, data=all_results)

    # Print and optionally save the top 10 aggregated results
    top_n = 10
    top_n_results = all_results[:top_n]
    top_10_file_path = f"/home/clone/clone/rag_benchmark_analysis/rag_evaluation/results/top_{top_n}_results_cluster_{cluster_name}_{model_name}.json"
    # evaluation_metrics.write_file(top_10_file_path, top_n_results)

    # Print top 10 results
    print(f"Top {top_n} Aggregated Results:")
    for result in top_n_results:
        print(result)