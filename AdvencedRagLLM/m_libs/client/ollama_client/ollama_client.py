import sys
from typing import List
#from ollama import generate
from ollama import Client

import threading

import re
import queue
from itertools import combinations
import pandas as pd
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

stop_parameters = {
  "llama3.1":  [
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>"
    ],
  "llama3.3:70b":  [
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>"
    ],
  'llama3.2:3b':[
       "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>"
      
  ],
  "llama3.1:70b":  [
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>"
    ],
  "gemma2":  [
        "<start_of_turn>",
        "<end_of_turn>"
    ],
  "gemma2:2b":  [
        "<start_of_turn>",
        "<end_of_turn>"
    ],
   "gemma2:9b":  [
        "<start_of_turn>",
        "<end_of_turn>"
    ],
  "gemma2:27b": [
        "<start_of_turn>",
        "<end_of_turn>"
    ],
   
"dolphin-mixtral":    
     [
        "<|im_start|>",
        "<|im_end|>"
    ],
"dolphin-mixtral:8x22b":
  [
        "<|im_start|>",
        "<|im_end|>"
    ],
  
"qwen:110b":[
        "<|im_start|>",
        "<|im_end|>"
    ],
"qwq": [
        "<|im_start|>",
        "<|im_end|>"
    ]

}

class CheckResponse:
    def __init__(self) -> None:
        pass
    def is_response_list(self, response):
        try:
            cleaned_response = re.sub(r'^\[|\]$|,\s*$', '', response.strip())
            response_list = re.findall(r'"(.*?)"', cleaned_response)

            #response= eval(response)
            if type(response_list) == list:
                # print("\n\n***response**\n",response_list,"\n\n")
                response_list = [item.strip() for item in response_list if len(item.split()) > 3]
                # print("***Filtered response**\n", response_list,flush=True)
                return response_list if len(response_list) > 0  else None
            else:
                return False
        except Exception as e:
            print("chek_response Error: ", e)
            return None
        
    def is_the_data_parsed_correctly(self, response=None, data=None):
        try:
            response_list = self.is_response_list(response)
            if response_list:
                edited_response = " ".join(response_list)
                if edited_response == data:
                    return response_list
                else:
                    return None
            #TODO: gerekirse burası güncellenecek
            return response
        except Exception as e:
            print("chek_response Error: ", e)
            return None
    def generate_combinations(self, content_list):
        combined_contents = []
        for r in range(1, len(content_list) + 1):
            for combo in combinations(content_list, r):
                combined_contents.append("\n".join(combo))
        return combined_contents
    def compute_rouge_scores_cosine_simularity(self, contents, response):

        model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')  # You can change the model

        # Encode the response and contents using the SentenceTransformer
        response_embedding = model.encode(response, convert_to_tensor=True)
        content_embeddings = model.encode(contents, convert_to_tensor=True)
        
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
        
        return scores
    def is_the_youtube_data_parsed_correctly(self, response=None):
        import json

        try:
            response_list = json.loads(response)
            # Check if the response is a list
            if not isinstance(response_list, list):
                print("Response is not a list")
                return None
            # Check each item in the list
            for item in response_list:
                if not isinstance(item, dict):
                    print("List item is not a dictionary")
                    return None
                if "title" not in item or "content" not in item:
                    print("Item does not contain 'title' and 'content' keys")
                    return None
            print("Response is in the correct format:", response_list)
            return response_list
        except json.JSONDecodeError as e:
            print("JSON decoding error:", e)
            return None
        
    def is_the_pdf_data_parsed_correctly(self, response=None):
        import json

        try:
            response_list = json.loads(response)
            # Check if the response is a list
            if not isinstance(response_list, list):
                print("Response is not a list")
                return None
            # Check each item in the list
            for item in response_list:
                if not isinstance(item, dict):
                    print("List item is not a dictionary")
                    return None
                if "title" not in item or "content" not in item:
                    print("Item does not contain 'title' and 'content' keys")
                    return None
            print("Response is in the correct format:", response_list)
            return response_list
        except json.JSONDecodeError as e:
            print("JSON decoding error:", e)
            return None
        
    def is_the_tweet_data_parsed_correctly(self, response=None):
        import json

        try:
            response_list = json.loads(response)
            # Check if the response is a list
            if not isinstance(response_list, list):
                print("Response is not a list")
                return None
            # Check each item in the list
            for item in response_list:
                if not isinstance(item, dict):
                    print("List item is not a dictionary")
                    return None
                if "title" not in item:
                    print("Item does not contain 'title' key")
                    return None
            print("Response is in the correct format:", response_list)
            return response_list
        except json.JSONDecodeError as e:
            print("JSON decoding error:", e)
            return None
        
    def is_the_topic_task_extraction_parsed_correctly(self, response=None):

        # cleaned_response = re.sub(r'```.*\n|\n```', '', response)

        # # Step 2: Remove the trailing comma before the closing bracket
        # cleaned_response = re.sub(r',\s*\]', ']', cleaned_response)
        import json

        try:
            response_list = json.loads(response)
            # Check if the response is a list
            if not isinstance(response_list, list):
                print("Response is not a list")
                return None
            # Check each item in the list
            for item in response_list:
                if not isinstance(item, dict):
                    print("List item is not a dictionary")
                    return None
                if "topic" not in item or "task" not in item:
                    print("Item does not contain 'topic' and 'task' keys")
                    return None
            print("Response is in the correct format:", response_list)
            return response_list
        except json.JSONDecodeError as e:
            print("JSON decoding error:", e)
            return None
                
    def is_response_compatible_with_content(self, response=None, content_list:list=None):
        try:
            rouge_score_threshold = 0.1
            cosine_similarity_threshold = 0.75
            if len(content_list)>0:
                combined_content_list = content_list.copy()
                combined_content_list = self.generate_combinations(combined_content_list)
                content_rouge_score_cosine_similarity = self.compute_rouge_scores_cosine_simularity(combined_content_list, response)
                
                df_content_scores = pd.DataFrame(content_rouge_score_cosine_similarity)

                # sorted_content_rouge_score = sorted(content_rouge_score, key=lambda x: x["ROUGE-L (F1)"], reverse=True)[0]["ROUGE-L (F1)"]
                filtered_results = df_content_scores[
                    (df_content_scores["ROUGE-L (F1)"] > rouge_score_threshold) | 
                    (df_content_scores["Cosine Similarity"] > cosine_similarity_threshold)
                ]

                if  len(filtered_results) >0 :
                    return response

            else:
                return response

            # unkownlist_rouge_control = 0.5
            # if len(rouge_content_list):
            #     #TODO: rouge control
            #     rouge_control = 0.5
                
            #     if 0.3>rouge_control or unkownlist_rouge_control >0.3:
                    
            #         return response
            # elif unkownlist_rouge_control >0.3:
            #     return response
            
            # else:
            #     return
                
        except Exception as e:
            print("chek_response Error: ", e)
            return None
            
class OllamaClient:
    def __init__(self, host="localhost",port="11436") -> None:
        self.result_queue = queue.Queue()
        self.check_response = CheckResponse()
        self.client = Client(host=f"http://{host}:{port}")
    def get_ai_response(self, model_name, prompt, options, stream=True):
        
        response_text = ""
        #print("\n\n")
        
        if stream:
            for part in self.client.generate(model_name,prompt=prompt ,options=options, stream=stream):
                
                #print(part['response'], end='', flush=True)
                response_text += part['response'].strip("\n")
        else:
            response = self.client.generate(model_name,prompt=prompt ,options=options)
            response_text = response["response"].strip("\n ")
        return response_text

    def generate_ai_response(self,id, model_name:str, prompt:str,options:dict, task:str=None, check_response_func=None, *args,**kwargs):
        
            response = ""
        
            i = 0
            while i < 10:
                response = self.get_ai_response(model_name, prompt, options=options, stream=False)
                if check_response_func:
                    response = check_response_func(response, *args,**kwargs)
                elif task =="sentences_list":
                    response = self.check_response.is_response_list(response)
                elif task == "compatible_with_content":
                    response = self.check_response.is_response_compatible_with_content(response, *args,**kwargs)
                elif task == "parsing_data":
                    response = self.check_response.is_the_data_parsed_correctly(response, prompt)
                elif task == "youtube_parsing_data":
                    response = self.check_response.is_the_youtube_data_parsed_correctly(response)
                elif task == "pdf_parsing_data":
                    response = self.check_response.is_the_pdf_data_parsed_correctly(response)
                elif task == "tweet_parsing_data":
                    response = self.check_response.is_the_tweet_data_parsed_correctly(response)
                elif task == "topic_task_extraction":
                    response = self.check_response.is_the_topic_task_extraction_parsed_correctly(response)
                if response:
                    self.result_queue.put({id:response})
                    break
                else:
                    options["temperature"] = options["temperature"]+ 0.1 if options["temperature"]< 1.0  else 0.9
                    options["top_p"] = options["top_p"]+ 0.1 if options["top_p"]< 1.0  else 0.9
                    # print("*****\nError response:\n***",response, flush=True)
                    i += 1
            return

    def get_all_responses(self,parameters:dict, same_task:bool=True, multi_threading_count:int=4) -> dict:
        
        thread_list:List[threading.Thread] = []
        i = 0
        for id, parameters_key in parameters.items():
            
            if parameters_key["options"] is None:
                stop_parameter = stop_parameters[parameters_key["model_name"]] if parameters_key["model_name"] in stop_parameters else None
                parameters_key["options"]={
                    'num_predict': 32_000,#2048
                    'temperature': 0.5,
                    'top_p': 0.6,
                    'stop':stop_parameter,
                }
            
            if same_task:
                parameters_key["options"]["temperature"] += 0.1
                parameters_key["options"]["top_p"] += 0.1
                i +=1
            parameters_key["id"] = id
            thread = threading.Thread(target=self.generate_ai_response, kwargs=parameters_key)
            thread_list.append(thread)
        for thread in thread_list:
            thread.start()
            
        for thread in thread_list:
            thread.join()
        responses = {}
        while not self.result_queue.empty():
            job_id_response = self.result_queue.get()
            responses.update(job_id_response)
        return responses

