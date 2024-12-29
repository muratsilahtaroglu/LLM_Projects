import random
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import sys
# import clone_prompts

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import llm_pre_processing.file_operations_utils as fo_utils
except:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import llm_pre_processing.file_operations_utils as fo_utils 

try:
    import clone_prompts
except:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import clone_prompts

import ollama_client  


ollama_ai = ollama_client.OllamaClient()


class TaskTopicExtractor:
    def __init__(self, query: str, repeat_count:int=1) -> None:
        self.query = query
        self.repeat_count = repeat_count

    
    def get_full_prompt(self) -> str:
        system_prompt = clone_prompts.topic_task_extractor_system_prompt.format()
        base_prompt = clone_prompts.topic_task_extractor_base_prompt.format(query=self.query)
        full_prompt = system_prompt + "\n" + base_prompt
        return full_prompt
    

    def get_task_and_topic(self, model_name, topic, count:int=1 ):
        
        parsed_data = []
        # content = self.get_semantic_search_result()        
        FULL_PROMPT = self.get_full_prompt()

        m = round(count/5) if round(count/5)>0 else count
        multi_threading_count = min(m, 5)
        

        # print("Topic: ",topic,flush=True)
        
        parameters = {}
        print(f"\nQuery: {self.query}",flush=True)
        for _ in range(multi_threading_count):
    
            ai_uudi = str(uuid.uuid4())
            parameters.update({ai_uudi: {"model_name": model_name, "prompt": FULL_PROMPT, "task": "topic_task_extraction","options":None}})
        responses = ollama_ai.get_all_responses(parameters, same_task=True)
        
        if len(responses):
            for ai_uudi, response in responses.items():
                if isinstance(response, list):
                    parsed_data.extend(response)
                else:
                    parsed_data.append(response)
        #parsed_data = list(set(parsed_data))
        #random.shuffle(parsed_data)
        return parsed_data
