import random
import uuid
from typing import Optional
import os
import sys
try:
    import llm_pre_processing.parse_prompts as parse_prompts, llm_pre_processing.file_operations_utils as fo_utils
except:
    import parse_prompts, file_operations_utils as fo_utils

#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import ollama_client, _utils
except:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import ollama_client, _utils  
    
ollama_ai = ollama_client.OllamaClient()

class UpdateYoutubeTranscript:

    def __init__(self, data_path:str, clone_name:str,repeat_count:int=1) -> None:
        self.data_path = data_path
        self.clone_name = clone_name
        self.repeat_count = repeat_count
        
    
    def get_full_prompts(self,all_content, title):
        k = 8000
        chuck_size = round(len(all_content)/k)+1
        full_prompt_list = []
        overlap = 20
        for i in range(chuck_size):
            tempt_content = all_content[i*k-overlap:(i+1)*k]
            if i == 0:
                overlap = 500
            system_prompt = parse_prompts.youtube_base_system_prompt.format(name=self.clone_name)
            user_prompt = parse_prompts.youtube_base_prompt.format(name=self.clone_name, content=tempt_content, title=title)
            full_prompt = system_prompt + "\n" + user_prompt
            full_prompt_list.append(full_prompt)
        return full_prompt_list
    
    def get_parsing_youtube_data(self, model_name, topic, count:int=1 ):
        parsed_data = []
        all_data = self.get_youtube_transcript(self.data_path)
        
        for d,data in enumerate(all_data):
            
            video_transcript = data["text"][0]
            video_title = data["video_title"]
            full_prompt_list = self.get_full_prompts(video_transcript, video_title)
            m = round(count/5) if round(count/5)>0 else count
            multi_threading_count = min(m, 5)
            
    
            print("Topic: ",topic,flush=True)
            for f,FULL_PROMPT in enumerate(full_prompt_list):
                parameters = {}
                print(f"Data:{d}, part:{f}",flush=True)
                for _ in range(multi_threading_count):
            
                    ai_uudi = str(uuid.uuid4())
                    parameters.update({ai_uudi: {"model_name": model_name, "prompt": FULL_PROMPT, "task": "youtube_parsing_data","options":None}})
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
    
    def get_youtube_transcript(self, path):

        return eval(fo_utils.read_textual_file(path))
     
    def save_youtube_transcript(self,data, path):

        return fo_utils.write_textual_file(data, path)


# model_name = "llama3.1:70b" #"gemma2:9b", "llama3.1","gemma2:27b", llama3.1:70b
# model_name = "gemma2:27b"
# model_name = "llama3.3:70b"
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir) 
# updata_youtube_transcript = UpdateYoutubeTranscript(os.path.join(parent_dir ,"data", "demo_youtube.json"),"Demo")
# parsed_data = updata_youtube_transcript.get_parsing_youtube_data(model_name=model_name, topic="parsing_youtube_data", count=2)
# updata_youtube_transcript.save_youtube_transcript(parsed_data, os.path.join(parent_dir ,f"edited_data/parsed_youtube_data_{model_name}.json"))
# print()