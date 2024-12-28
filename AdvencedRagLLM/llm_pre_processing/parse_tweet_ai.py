import random
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import sys
import re
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, MathpixPDFLoader, \
UnstructuredPDFLoader, OnlinePDFLoader, PyPDFium2Loader,PDFMinerLoader, PDFMinerPDFasHTMLLoader, PyPDFDirectoryLoader,\
PDFPlumberLoader,AmazonTextractPDFLoader
try:
    import llm_pre_processing.parse_prompts as parse_prompts, llm_pre_processing.file_operations_utils as fo_utils
except:
    import parse_prompts, file_operations_utils as fo_utils

try:
    import ollama_client, _utils
except:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import ollama_client, _utils  



ollama_ai = ollama_client.OllamaClient()


class UpdateTweetText:

    def __init__(self, data_path:str, clone_name:str,repeat_count:int=1) -> None:
        self.data_path = data_path
        self.clone_name = clone_name
        self.repeat_count = repeat_count

    def get_full_prompts(self,all_content):
        system_prompt = parse_prompts.tweet_base_system_prompt.format(name=self.clone_name)
        user_prompt = parse_prompts.tweet_base_prompt.format(name=self.clone_name, content=all_content)
        full_prompt = system_prompt + "\n" + user_prompt
        return full_prompt
    
    
    def get_parsing_tweet_data(self, model_name, topic, count:int=1 ):
        
        parsed_data = []
        all_data = self.get_tweet_text(self.data_path)
        for i in range(len(all_data)):        
            tweet_text = all_data['TEXT'][i] 
            FULL_PROMPT = self.get_full_prompts(tweet_text)

            m = round(count/5) if round(count/5)>0 else count
            multi_threading_count = min(m, 5)
            

            print("Topic: ",topic,flush=True)
            
            parameters = {}
            print(f"Data:{tweet_text}",flush=True)
            for _ in range(multi_threading_count):
        
                ai_uudi = str(uuid.uuid4())
                parameters.update({ai_uudi: {"model_name": model_name, "prompt": FULL_PROMPT, "task": "tweet_parsing_data","options":None}})
            responses = ollama_ai.get_all_responses(parameters, same_task=True)
            
            if len(responses):
                for ai_uudi, response in responses.items():
                    if isinstance(response, list):
                        response[0]['content'] = self.remove_url(tweet_text)
                        response[0]['url'] = self.get_url(tweet_text)
                        parsed_data.extend(response)
                    else:
                        response['content'] = self.remove_url(tweet_text)
                        response['url'] = self.get_url(tweet_text)
                        parsed_data.append(response)
        #parsed_data = list(set(parsed_data))
        #random.shuffle(parsed_data)
        return parsed_data
    
    def get_tweet_text(self, path):
        data = fo_utils.read_textual_file(path)
        text = data[['TEXT']]
        return text
    
    def get_url(self, text):
        try:
            url_pattern = r'https?://(?:www\.)?[-\w./]+(?:\?\S*)?'
            # Find all URLs in the text using the regex pattern
            urls = re.findall(url_pattern, text)
            return urls
        except:
            return None

    def remove_url(self,text):
        try:
            url_pattern = r'https?://(?:www\.)?[-\w./]+(?:\?\S*)?'
            # Find all URLs in the text using the regex pattern
            cleaned_text = re.sub(url_pattern, '', text)
            return cleaned_text
        except:
            return None
    


     
    def save_tweet_text(self,data, path):

        return fo_utils.write_textual_file(data, path)



# model_name = "gemma2:27b" #"gemma2:9b", "llama3.1","gemma2:27b"


# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir) 

# update_tweet_text = UpdateTweetText(os.path.join(parent_dir ,"data", "demo_tweet.json"),"Demo")
# # res = update_tweet_text.get_tweet_text(path)
# parsed_data = update_tweet_text.get_parsing_tweet_data(model_name=model_name, topic="parsing_tweet_data", count=1)
# update_tweet_text.save_tweet_text(parsed_data, os.path.join(parent_dir ,f"edited_data/parsed_tweet_data_{model_name}.json"))
# print()