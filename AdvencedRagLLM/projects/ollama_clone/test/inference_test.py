import uuid
import time
import random
from sentence_transformers import SentenceTransformer, util
from itertools import combinations
from rouge_score import rouge_scorer
import re
import os
import sys
import statistics

try:
    import ollama_clone.clone_ai
except:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import ollama_clone.clone_ai

class CloneAITester:
    def __init__(self, data_path: str, model_name: str, clone_name: str, queries: list):
        """
        Initialize the inference tester with essential details.
        
        Args:
            data_path (str): Path to the file containing personal information.
            model_name (str): The name of the AI model to use for inference.
            clone_name (str): The name to personalize CloneAI interactions.
            queries (list): List of queries to test CloneAI with.
        """
        self.data_path = data_path
        self.model_name = model_name
        self.clone_name = clone_name
        self.queries = queries
        self.clone_ai = ollama_clone.clone_ai.CloneAI(data_path=data_path, clone_name=clone_name, query="")

    def test_single_query(self,  query: str, is_prompt_related_question_style: bool=False):
        """
        Test the CloneAI system with a single query.
        
        Args:
            query (str): Query string to test with the model.
        
        Returns:
            dict: The response from CloneAI.
        """
        self.clone_ai.query = query
        try:
            st = time.time()
            response = self.clone_ai.get_clone_response_for_inference(model_name=self.model_name, topic="creating_clone_text", count=1, is_prompt_related_question_style=is_prompt_related_question_style)    
            en = time.time()
            total_time = en-st
            response["time"] = total_time
            return response
        except Exception as e:
            return {"error": str(e)}

    def test_all_queries(self,is_prompt_related_question_style: bool=False):
        """
        Test CloneAI with all queries provided during initialization.
        
        Returns:
            list: list of Dictionary containing query-response.
        """
        results = []
        for query in self.queries:
            # print(f"Testing query: {query}", flush=True)
            response = self.test_single_query(query,is_prompt_related_question_style)
            results.append(response)
        return results


    def get_query_from_dataset(self, file_path: str, k:int=20):
        train_data = eval(self.clone_ai.read_file(file_path))
        random.seed(0)
        train_data = random.sample(train_data, k=k)
        train_data = [d["query"] for d in train_data]
        return train_data
    

    def save_results(self, results: list, output_path: str):
        """
        Save the inference test results to a file.
        
        Args:
            results (list): List of query-response pairs.
            output_path (str): Path to save the results as a text file.
        """
        try:
            
            self.clone_ai.save_results(results, output_path)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")


    def run_inference_tests(self, output_path: str = "inference_results.txt", is_prompt_related_question_style: bool=False):
        """
        Run inference tests on all queries and save the results.
        
        Args:
            output_path (str): Path to save the test results.
        """
        print("Starting inference tests...", flush=True)
        results = self.test_all_queries(is_prompt_related_question_style)
        self.save_results(results, output_path)
        print("Inference tests completed.", flush=True)


    def get_full_prompts(self):
        personal_info = self.clone_ai.read_file(self.data_path)

        results = []
        for query in self.queries:
            res_dic = {}
            # print(f"Testing query: {query}", flush=True)
            self.clone_ai.query = query
            task, text_topic = self.clone_ai.get_task_and_topic()
            content, content_list = self.clone_ai.prepare_content_prompts(text_topic)

            FULL_PROMPT = self.clone_ai.get_full_prompt(personal_info, content)
            question_style = self.clone_ai.get_question_distinguisher_response()
            res_dic["query"] = query
            res_dic["topic"] = text_topic
            res_dic["task"] = task
            res_dic["content_list"] = content_list
            res_dic["question_style"] = question_style
            res_dic["full_prompt"] = FULL_PROMPT
            results.append(res_dic)
        self.save_results(results, 'query_full_prompts.json')
        
    def generate_combinations(self, content_list):
        
        combined_contents = []
        for r in range(1, len(content_list) + 1):
            for combo in combinations(content_list, r):
                combined_contents.append("\n".join(combo))
        return combined_contents
    

    def compute_cosine_similarity(self, contents, response):

        model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')  # You can change the model

        # Encode the response and contents using the SentenceTransformer
        response_embedding = model.encode(response, convert_to_tensor=True)
        content_embeddings = model.encode(contents, convert_to_tensor=True)
        
        # Compute cosine similarity between response and each content combination
        cosine_scores = util.pytorch_cos_sim(response_embedding, content_embeddings)[0]

        return cosine_scores[0].item()
    
    
    def compute_rouge_scores_cosine_simularity(self, contents, response):
        
        model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')  # You can change the model

        # Encode the response and contents using the SentenceTransformer
        response_embedding = model.encode(response) #, convert_to_tensor=True
        content_embeddings = model.encode(contents)
        
        # Compute cosine similarity between response and each content combination
        cosine_scores = util.pytorch_cos_sim(response_embedding, content_embeddings)[0]

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = []
        
        for i, content in enumerate(contents):
            rouge_score = scorer.score(content, response)
            scores.append({
                "Combination": content,
                "ROUGE-1 (F1)": rouge_score['rouge1'].fmeasure,
                "ROUGE-L (F1)": rouge_score['rougeL'].fmeasure,
                "Cosine Similarity": cosine_scores[i].item()
            })
        
        # Calculate the maximum scores
        max_rouge_1 = max(scores, key=lambda x: x["ROUGE-1 (F1)"])["ROUGE-1 (F1)"]
        max_rouge_l = max(scores, key=lambda x: x["ROUGE-L (F1)"])["ROUGE-L (F1)"]
        max_cosine_similarity = max(scores, key=lambda x: x["Cosine Similarity"])["Cosine Similarity"]

        result = {
                "rouge-1": max_rouge_1,
                "rouge-l": max_rouge_l,
                "cosine-similarity": max_cosine_similarity
        }

        return result
            

    def merge_results(self, model1_results_file, model2_results_file, model3_results_file, output_file):
        def read_and_evaluate(file_path):
            """Helper function to read and evaluate a file."""
            return eval(self.clone_ai.read_file(file_path))

        def calculate_averages(score_lists):
            """Helper function to compute averages for score lists."""
            return {key: statistics.mean(value) if value else 0 for key, value in score_lists.items()}

        def append_scores(score_lists, score_data):
            """Helper function to append scores to corresponding lists."""
            for key, value in score_data.items():
                if key in score_lists and value is not None:
                    score_lists[key].append(value)

        # Read and evaluate model result files
        model_results = {
            "27b": read_and_evaluate(model1_results_file),
            "2b": read_and_evaluate(model2_results_file),
            "finetuned": read_and_evaluate(model3_results_file),
        }
        personal_information = self.clone_ai.read_file(self.clone_ai.data_path)

        # Initialize lists for scores
        score_lists = {
            "gemma_27b_2b_cos_sim": [],
            "gemma_27b_finetuned_cos_sim": [],
            "gemma_2b_finetuned_cos_sim": [],
            "content_cosine_sim_27b": [],
            "rouge1_27b": [],
            "rougeL_27b": [],
            "content_cosine_sim_2b": [],
            "rouge1_2b": [],
            "rougeL_2b": [],
            "content_cosine_sim_finetuned": [],
            "rouge1_finetuned": [],
            "rougeL_finetuned": [],
        }

        results = []
        for i, model_27b_result in enumerate(model_results["27b"]):
            res_dic = {
                "query": model_27b_result["query"],
                "question_style": model_27b_result["question_style"],
                "topic": model_27b_result["topic"],
                "task": model_27b_result["task"],
                "content": model_27b_result["content"],
                "content_list": model_27b_result["content_list"],
                "full_prompt": model_27b_result["full_prompt"],
                "gemma27b_response": model_27b_result["response"][0],
                "gemma2b_response": model_results["2b"][i]["response"][0],
                "finetuned_model_response": model_results["finetuned"][i]["result"],
                "gemma_27b_2b_cos_sim": self.compute_cosine_similarity(
                    model_27b_result["response"][0], model_results["2b"][i]["response"][0]
                ),
                "gemma_27b_finetuned_cos_sim": self.compute_cosine_similarity(
                    model_27b_result["response"][0], model_results["finetuned"][i]["result"]
                ),
                "gemma_2b_finetuned_cos_sim": self.compute_cosine_similarity(
                    model_results["2b"][i]["response"][0], model_results["finetuned"][i]["result"]
                ),
            }

            if model_27b_result["content"]:
                combined_content_list = model_27b_result["content_list"].copy()
                if not combined_content_list:
                    sections = re.split(r'\n?\d+\)\s', model_27b_result["content"].strip())
                    model_27b_result["content_list"] = [s.strip() for s in sections if s]
                    combined_content_list.extend([personal_information])

                combined_content_list = self.generate_combinations(combined_content_list)

                # Compute ROUGE and cosine similarity scores
                score_data = {
                    "content_cosine_sim_27b": self.compute_rouge_scores_cosine_simularity(
                        combined_content_list, model_27b_result["response"][0]
                    ),
                    "content_cosine_sim_2b": self.compute_rouge_scores_cosine_simularity(
                        combined_content_list, model_results["2b"][i]["response"][0]
                    ),
                    "content_cosine_sim_finetuned": self.compute_rouge_scores_cosine_simularity(
                        combined_content_list, model_results["finetuned"][i]["result"]
                    ),
                }

                res_dic.update({
                    "gemma_27b_content_cosine_rouge-l": score_data["content_cosine_sim_27b"],
                    "gemma_2b_content_cosine_rouge-l": score_data["content_cosine_sim_2b"],
                    "finetuned_model_content_cosine_rouge-l": score_data["content_cosine_sim_finetuned"],
                })

                append_scores(score_lists, {
                    "content_cosine_sim_27b": score_data["content_cosine_sim_27b"].get("cosine-similarity"),
                    "rouge1_27b": score_data["content_cosine_sim_27b"].get("rouge-1"),
                    "rougeL_27b": score_data["content_cosine_sim_27b"].get("rouge-l"),
                    "content_cosine_sim_2b": score_data["content_cosine_sim_2b"].get("cosine-similarity"),
                    "rouge1_2b": score_data["content_cosine_sim_2b"].get("rouge-1"),
                    "rougeL_2b": score_data["content_cosine_sim_2b"].get("rouge-l"),
                    "content_cosine_sim_finetuned": score_data["content_cosine_sim_finetuned"].get("cosine-similarity"),
                    "rouge1_finetuned": score_data["content_cosine_sim_finetuned"].get("rouge-1"),
                    "rougeL_finetuned": score_data["content_cosine_sim_finetuned"].get("rouge-l"),
                "gemma_27b_2b_cos_sim": self.compute_cosine_similarity(
                    model_27b_result["response"][0], model_results["2b"][i]["response"][0]
                ),
                "gemma_27b_finetuned_cos_sim": self.compute_cosine_similarity(
                    model_27b_result["response"][0], model_results["finetuned"][i]["result"]
                ),
                "gemma_2b_finetuned_cos_sim": self.compute_cosine_similarity(
                    model_results["2b"][i]["response"][0], model_results["finetuned"][i]["result"]
                ),
                })
            else:
                res_dic.update({
                    "gemma_27b_content_cosine_rouge-l": {},
                    "gemma_2b_content_cosine_rouge-l": {},
                    "finetuned_model_content_cosine_rouge-l": {},
                })

            results.append(res_dic)

        # Compute and append averages
        averages = calculate_averages(score_lists)
        results.append({"average_scores": averages})

        self.clone_ai.save_results(results, output_file)



if __name__ == "__main__":

    queries = ["What is the meaning of life?", "Tell me capital of Turkey."]
    

    tester = CloneAITester(
    data_path="personal_info.txt",
    model_name="gemma2:27b",
    clone_name="CloneBot",
    queries=queries
)   
    
    is_prompt_related_question_style=False

    # train_data_query = tester.get_query_from_dataset('data/clone_ai_datasetV5.json', k=10)
    # valdata_query = tester.get_query_from_dataset('data/clone_ai_valsetV4_updated2.json', k=10)
    
    # tester.queries = queries 
    # tester.queries = valdata_query 
    # tester.queries = train_data_query 

    # tester.run_inference_tests(output_path="results/inference_test_results_gemma_27b_original_prompt_questions.json", is_prompt_related_question_style=is_prompt_related_question_style)


    tester.merge_results("results/inference_test_results_gemma_27b_question_updated.json",
                          "results/inference_test_results_gemma_2b_question_updated.json",
                          "results/chat_model_v12_question.json",
                          "results/chat_model_v12_question_results.json")


    tester.merge_results("results/inference_test_results_gemma_27b_trainset_updated.json",
                          "results/inference_test_results_gemma_2b_trainset_updated.json",
                          "results/chat_model_v12_trainset.json",
                          "results/chat_model_v12_trainset_results.json")


    tester.merge_results("results/inference_test_results_gemma_27b_valset_updated.json",
                          "results/inference_test_results_gemma_2b_valset_updated.json",
                          "results/chat_model_v12_valset.json",
                          "results/chat_model_v12_valset_results.json")


    # tester.merge_results("results/inference_test_results_gemma_27b_original_prompt_questions.json",
    #                       "results/inference_test_results_gemma_2b_original_prompt_questions.json",
    #                       "results/gemma_v1_question.json",
    #                       "results/gemma_v1_question_results.json")


    # tester.merge_results("results/inference_test_results_gemma_27b_datasetV5.json",
    #                       "results/inference_test_results_gemma_2b_datasetV5.json",
    #                       "results/gemma_v1_trainset.json",
    #                       "results/gemma_v1_trainset_results.json")


    # tester.merge_results("results/inference_test_results_gemma_27b_valsetV4_updated2.json",
    #                       "results/inference_test_results_gemma_2b_valsetV4_updated2.json",
    #                       "results/gemma_v1_valset.json",
    #                       "results/gemma_v1_valset_results.json")