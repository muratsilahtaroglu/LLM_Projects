import random
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import sys
import clone_prompts
import topic_task_extractor_ai
import time
import queue
import threading

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import semantic_search as sem_search
except:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import semantic_search as sem_search 

try:
    import llm_pre_processing.file_operations_utils as fo_utils
except:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import llm_pre_processing.file_operations_utils as fo_utils 

import ollama_client


ollama_ai = ollama_client.OllamaClient()

class CloneAI:
    def __init__(self, user_info_file_path :str, clone_name:str, query: str, repeat_count:int=1) -> None:
        self.user_info_file_path = user_info_file_path
        self.clone_name = clone_name
        self.query = query
        self.repeat_count = repeat_count

    
    def get_full_prompt(self, personal_info:str, content:str) -> str:
        """
        Returns the full prompt for the AI to generate content.
        
        Parameters:
        personal_info (str): The personal information about the user.
        content (str): The content to generate a response based on.
        
        Returns:
        str: The full prompt for the AI.
        """
        unknown_answer = self.get_random_unknown_answer()
        system_prompt = clone_prompts.clone_system_prompt.format(name=self.clone_name, personal_info=personal_info,
        content=content,
        query=self.query)
        base_prompt = clone_prompts.clone_base_prompt.format(name=self.clone_name, query=self.query, content=content)
        full_prompt = system_prompt + "\n" + base_prompt
        return full_prompt

    
    def get_full_prompt_tr(self, personal_info:str, content:str) -> str:
        """
        Returns the full prompt for the AI to generate content in Turkish.
        
        Parameters:
        personal_info (str): The personal information about the user.
        content (str): The content to generate a response based on.
        
        Returns:
        str: The full prompt for the AI.
        tuple: A tuple containing the full prompt, base prompt, and system prompt.
        """
        unknown_answer = self.get_random_unknown_answer()
        system_prompt = clone_prompts.clone_system_prompt_tr.format(name=self.clone_name, personal_info=personal_info)
        base_prompt = clone_prompts.clone_base_prompt_tr.format(name=self.clone_name, query=self.query, content=content)
        full_prompt = system_prompt + "\n" + base_prompt
        return full_prompt, base_prompt, system_prompt
    

    def get_full_prompt_wo_personal_info(self, content:str) -> str:
        """
        Returns the full prompt for the AI to generate content without personal information.
        
        Parameters:
        content (str): The content to generate a response based on.
        
        Returns:
        str: The full prompt for the AI.
        """
        unknown_answer = self.get_random_unknown_answer()
        system_prompt = clone_prompts.clone_system_prompt_wo_personal_info.format(name=self.clone_name,
        content=content,
        query=self.query)
        base_prompt = clone_prompts.clone_base_prompt_wo_personal_info.format(name=self.clone_name, query=self.query, content=content)
        full_prompt = system_prompt + "\n" + base_prompt
        return full_prompt

    def get_full_prompt_wo_personal_info_tr(self, content:str) -> str:
        """
        Returns the full prompt for the AI to generate content without personal information in Turkish.
        
        Parameters:
        content (str): The content to generate a response based on.
        
        Returns:
        str: The full prompt for the AI.
        tuple: A tuple containing the full prompt, base prompt, and system prompt.
        """
        unknown_answer = self.get_random_unknown_answer()
        system_prompt = clone_prompts.clone_system_prompt_wo_personal_info_tr.format(name=self.clone_name)
        base_prompt = clone_prompts.clone_base_prompt_wo_personal_info_tr.format(name=self.clone_name, query=self.query, content=content)
        full_prompt = system_prompt + "\n" + base_prompt
        return full_prompt, base_prompt, system_prompt


    def get_random_unknown_answer(self) -> str:
        """
        Returns a random string from unknown_answers list.
        
        Returns:
        str: A random string from unknown_answers list.
        """
        return random.choice(clone_prompts.unknown_answers)
    

    def get_full_prompt2(self, personal_info:str, content:str, topic:str) -> str:
        
        
        unknown_answer = self.get_random_unknown_answer()
        system_prompt = clone_prompts.clone_system_prompt2.format(name=self.clone_name, personal_info=personal_info,
        content=content,unknown_answer=unknown_answer,
        query=self.query)
        base_prompt = clone_prompts.clone_base_prompt2.format(name=self.clone_name, query=self.query, content=content, topic=topic,unknown_answer = unknown_answer, personal_info=personal_info)
        full_prompt = system_prompt + "\n" + base_prompt
        return full_prompt
    

    def get_question_preparer_full_prompt(self, topic:str) -> str:
        
        system_prompt = clone_prompts.question_preparation_system_prompt.format()
        base_prompt = clone_prompts.question_preparation_base_prompt.format(topic=topic)
        full_prompt = system_prompt + "\n" + base_prompt
        return full_prompt
    
    def get_question_distinguisher_full_prompt(self) -> str:
        system_prompt = clone_prompts.question_distinguisher_system_prompt.format(query=self.query)
        base_prompt = clone_prompts.question_distinguisher_system_prompt.format(query=self.query, name=self.clone_name)
        full_prompt = system_prompt + "\n" + base_prompt
        return full_prompt

    def get_question_distinguisher_response(self, model_name:str = "gemma2:27b", topic:str ="question_distinguish", count:int=1) -> list:
        """
        This function is used to get the response of question_distinguisher task for the given prompt.
        
        Parameters:
        model_name (str): The name of the model to use for generating text. Defaults to "gemma2:27b".
        topic (str): The topic of the text to generate. Defaults to "question_distinguish".
        count (int): The number of responses to generate. Defaults to 1.
        
        Returns:
        list: A list of responses.
        """
        parsed_data = []

        FULL_PROMPT = self.get_question_distinguisher_full_prompt()

        m = round(count/5) if round(count/5)>0 else count
        multi_threading_count = min(m, 5)
        
        
        parameters = {}

        for _ in range(multi_threading_count):
            ai_uudi = str(uuid.uuid4())
            parameters.update({ai_uudi: {"model_name": model_name, "prompt": FULL_PROMPT, "task": "parsing_data","options":None}})
        responses = ollama_ai.get_all_responses(parameters, same_task=True)
        
        if len(responses):
            for ai_uudi, response in responses.items():
                if isinstance(response, list):
                    parsed_data.extend(response)
                else:
                    parsed_data.append(response)
        #parsed_data = list(set(parsed_data))
        #random.shuffle(parsed_data)
        return parsed_data[0]

    def get_clone_response(self, model_name:str, topic:str, count:int=1 ) -> list:
        
        """
        Generates a response using the CloneAI model based on the given topic and model name.

        Parameters:
        model_name (str): The name of the model to use for generating text.
        topic (str): The topic of the text to generate.
        count (int): The number of responses to generate. Defaults to 1.

        Returns:
        list: A list containing the topic, task, and the generated response(s).
        """

        parsed_data = []
        personal_info = (fo_utils.read_textual_file(self.user_info_file_path))
        task, text_topic = self.get_task_and_topic()
        # print(f"\nTask: {task}", flush=True)
        # print(f"Topic: {text_topic}", flush=True)
            
        content, content_list = self.prepare_content_prompts(text_topic)

        FULL_PROMPT = self.get_full_prompt(personal_info, content)

        question_style = self.get_question_distinguisher_response()
        print(f"\nQuestion style: {question_style}", flush=True)
        if question_style == "Requires semantic search":
            #TODO: If content_list is empty, write the missing topic to a txt file. To detect missing topics when generating a dataset or to detect missing topics in the rag system.
            if len(content_list) == 0:
                parsed_data = [self.get_random_unknown_answer(), self.get_random_unknown_answer()]
                return { "topic": text_topic, "task": task, "response": parsed_data}
        else:
            if len(content_list) > 0:
                content_list = []
        m = round(count/5) if round(count/5)>0 else count
        multi_threading_count = min(m, 5)
                
        parameters = {}
        for _ in range(multi_threading_count):
    
            ai_uudi = str(uuid.uuid4())
            parameters.update({ai_uudi: {"model_name": model_name, "prompt": FULL_PROMPT, "task": "clone_text","options":None, "content_list": content_list, "personal_information": personal_info}})

        responses = ollama_ai.get_all_responses(parameters, same_task=True)
        if len(responses):
            for ai_uudi, response in responses.items():
                if isinstance(response, list):
                    parsed_data.extend(response)
                elif response is None:
                    response = f"I can't answer you about {text_topic}"
                    parsed_data.append(response)
                elif len(response) == 0:
                    response = f"I can't answer you about {text_topic}"
                    parsed_data.append(response)
                else:
                    parsed_data.append(response)
        else:
            response = f"I can't answer you about {text_topic}"
            parsed_data = [response]
            return { "topic": text_topic, "task": task, "response": parsed_data}
        #parsed_data = list(set(parsed_data))
        #random.shuffle(parsed_data)

        return { "topic": text_topic, "task": task, "response": parsed_data}
    

    def get_clone_response_for_rag_benchmark(self, model_name:str, topic:str, count:int=1 ) -> list:
        
        """
        This function is used to generate responses for the RAG benchmark.

        Parameters:
        model_name (str): The name of the model to use for generating text.
        topic (str): The topic of the text to generate.
        count (int): The number of responses to generate. Defaults to 1.

        Returns:
        dict: A dictionary with two keys: content and response. Content is the list of content from the database for the given topic, and response is a list of responses for the given topic.
        """
        parsed_data = []
        personal_info = (fo_utils.read_textual_file(self.user_info_file_path))
        task, text_topic = self.get_task_and_topic()
            
        content, content_list = self.prepare_content_prompts(text_topic)

        FULL_PROMPT = self.get_full_prompt(personal_info, content)

        question_style = self.get_question_distinguisher_response()
        print(f"\nQuestion style: {question_style}", flush=True)
        if question_style == "Requires semantic search":
            if len(content_list) == 0:
                parsed_data = [self.get_random_unknown_answer(), self.get_random_unknown_answer()]
                return {"content": content_list, "response": parsed_data[0]}
        else:
            if len(content_list) > 0:
                content_list = []
        m = round(count/5) if round(count/5)>0 else count
        multi_threading_count = min(m, 5)
                
        parameters = {}
        for _ in range(multi_threading_count):
    
            ai_uudi = str(uuid.uuid4())
            parameters.update({ai_uudi: {"model_name": model_name, "prompt": FULL_PROMPT, "task": "clone_text","options":None, "content_list": content_list, "personal_information": personal_info}})

        responses = ollama_ai.get_all_responses(parameters, same_task=True)
        if len(responses):
            for ai_uudi, response in responses.items():
                if isinstance(response, list):
                    parsed_data.extend(response)
                elif response is None:
                    response = f"I can't answer you about {text_topic}"
                    parsed_data.append(response)
                elif len(response) == 0:
                    response = f"I can't answer you about {text_topic}"
                    parsed_data.append(response)
                else:
                    parsed_data.append(response)
        else:
            response = f"I can't answer you about {text_topic}"
            parsed_data = [response]
            return {"content": content_list, "response": parsed_data[0]}
        #parsed_data = list(set(parsed_data))
        #random.shuffle(parsed_data)

        return {"content": content_list, "response": parsed_data[0]}


    def get_clone_response_for_inference(self, model_name:str, topic:str, count:int=1,  is_prompt_related_question_style: bool=False ) -> list:
        
        """
        This function is used to generate responses for the inference.

        Parameters:
        model_name (str): The name of the model to use for generating text.
        topic (str): The topic of the text to generate.
        count (int): The number of responses to generate. Defaults to 1.
        is_prompt_related_question_style (bool): Whether the question style is related to the prompt. Defaults to False.

        Returns:
        dict: A dictionary with several keys: query, question_style, topic, task, content, content_list, full_prompt, and response. 
        query is the original query, question_style is the style of the question, topic is the topic of the text to generate, task is the task of the text to generate, content is the content from the database for the given topic, content_list is the content list from the database for the given topic, full_prompt is the full prompt used to generate the response, and response is a list of responses for the given topic.
        """
        parsed_data = []
        personal_info = (fo_utils.read_textual_file(self.user_info_file_path))
        task, text_topic = self.get_task_and_topic()
        # print(f"\nTask: {task}", flush=True)
        # print(f"Topic: {text_topic}", flush=True)
            
        content, content_list = self.prepare_content_prompts(text_topic)

        if is_prompt_related_question_style:
            FULL_PROMPT = self.get_full_prompt_wo_personal_info(content)
        else:
            FULL_PROMPT = self.get_full_prompt(personal_info, content)

        question_style = self.get_question_distinguisher_response()
        print(f"\nQuestion style: {question_style}", flush=True)
        if question_style == "Requires semantic search":
            if len(content_list) == 0:
                parsed_data = [self.get_random_unknown_answer(), self.get_random_unknown_answer()]
                return { "query": self.query,"question_style": question_style, "topic": text_topic, "task": task, "content": content,"content_list": content_list, "full_prompt": FULL_PROMPT, "response": parsed_data}
        else:
            if len(content_list) > 0:
                content_list = []
            

        m = round(count/5) if round(count/5)>0 else count
        multi_threading_count = min(m, 5)
                
        parameters = {}
        for _ in range(multi_threading_count):
    
            ai_uudi = str(uuid.uuid4())
            parameters.update({ai_uudi: {"model_name": model_name, "prompt": FULL_PROMPT, "task": "clone_text","options":None, "content_list": content_list, "personal_information": personal_info}})

        responses = ollama_ai.get_all_responses(parameters, same_task=True)
        if len(responses):
            for ai_uudi, response in responses.items():
                if isinstance(response, list):
                    parsed_data.extend(response)
                elif response is None:
                    response = f"I can't answer you about {text_topic}"
                    parsed_data.append(response)
                elif len(response) == 0:
                    response = f"I can't answer you about {text_topic}"
                    parsed_data.append(response)
                else:
                    parsed_data.append(response)
        else:
            response = f"I can't answer you about {text_topic}"
            parsed_data = [response]
            return { "query": self.query,"question_style": question_style, "topic": text_topic, "task": task, "content": content, "content_list": content_list, "full_prompt": FULL_PROMPT, "response": parsed_data}

        return { "query": self.query,"question_style": question_style, "topic": text_topic, "task": task, "content": content,"content_list": content_list, "full_prompt": FULL_PROMPT, "response": parsed_data}
    

    def get_task_and_topic(self, model_name:str = "gemma2:27b"):
        """
        Gets the task and topic for the given query from the semantic search using the given model name.

        Args:
            model_name (str, optional): The model name to use. Defaults to "gemma2:27b".

        Returns:
            tuple: A tuple of the task and topic.
        """
        tte = topic_task_extractor_ai.TaskTopicExtractor(self.query)
        task_and_topic = tte.get_task_and_topic(model_name=model_name, topic="extracting_task_and_topic")
        task = task_and_topic[0]['task']
        topic = task_and_topic[0]['topic'] 
        return task, topic


    def get_semantic_search_result(self, topic: str, similarity_documents_args: dict = None, rerank_method:str="merge_auto", length_threshold:int=700) -> dict:
        content_types = ["youtube", "twitter", "pdf"]
        all_query_and_info_file = "clone_search_predictors.json"
        thread_list = []
        result_queue = queue.Queue()

        def fetch_content(content_type, similarity_documents_args, length_threshold):
            """
            Fetches content for a given content type using semantic search.

            Args:
                content_type (str): The type of content to fetch.
                similarity_documents_args (dict): Additional settings to refine the similarity document arguments.
                length_threshold (int): The maximum length for the content.

            Side Effects:
                Updates the knowledge index's collection name and app token.
                Performs a semantic search to retrieve relevant documents.
                Stores the retrieved results in the result queue.
            """
            semantic_search = sem_search.SemanticSearch(
                collection_name=content_type, 
                all_query_and_info_file=all_query_and_info_file
            )
            semantic_search.set_app_token_by_collection_name()
            similarity_doc_args = {
            "all_query_and_info_file": all_query_and_info_file,
            "app_token": semantic_search.app_token,
            }
            similarity_doc_args.update(similarity_documents_args)
            if rerank_method == "evaluation":
                result = semantic_search.set_query_and_relations_response_for_evaluation(topic, similarity_documents_args=similarity_doc_args, rerank_method=rerank_method)    
            else: result = semantic_search.set_query_and_relations_response(topic, similarity_documents_args=similarity_doc_args, rerank_method=rerank_method, length_threshold=length_threshold)
            result_queue.put({content_type: result})

        for content_type in content_types:
            thread = threading.Thread(target=fetch_content, args=(content_type,similarity_documents_args,length_threshold))
            thread_list.append(thread)

        for thread in thread_list:
            thread.start()

        for thread in thread_list:
            thread.join()

        # Collect results from the queue
        content = {}
        while not result_queue.empty():
            content.update(result_queue.get())

        return content
    

    def prepare_content_prompts(self,topic:str, similarity_document_args: dict = None):
        """
        Prepares content prompts based on semantic search results for a given topic.

        Args:
            topic (str): The topic for which to prepare content prompts.
            similarity_document_args (dict, optional): Additional arguments for retrieving similarity documents. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - prompt (str): A formatted string with the content organized by type (e.g., PDF, text).
                - content_list (list): A list of strings each containing a title and its associated content.
        """

        content = self.get_semantic_search_result(topic, similarity_documents_args=similarity_document_args)
        content_list = []

        prompt = ""
        for k,v in content.items():
            if len(v)>0:
                if k == "pdf" :
                    k = "pdf"
                prompt += f"""\n\n## {k.title()} Content:"""
                for i in range(len(v)):
                    prompt += f"""\n{i+1}) {v[i]["title"]}:\n{v[i]["content"]}"""
                    content_list.append(f"""{v[i]["title"]}:\n{v[i]["content"]}""")
            
        return prompt, content_list

    
    def prepare_content_prompts_all(self,topic:str, similarity_document_args: dict = None, rerank_method:str="merge_auto", length_threshold:int=700):
        
        """
        Prepares content prompts based on semantic search results for a given topic.

        Args:
            topic (str): The topic for which to prepare content prompts.
            similarity_document_args (dict, optional): Additional arguments for retrieving similarity documents. Defaults to None.
            rerank_method (str, optional): The reranking method to use. Defaults to "merge_auto".
            length_threshold (int, optional): The length threshold for the content. Defaults to 700.

        Returns:
            tuple: A tuple containing:
                - prompt (str): A formatted string with the content organized by type (e.g., PDF, text).
                - content_list (list): A list of strings each containing a title and its associated content.
        """
        content = self.get_semantic_search_result(topic, similarity_documents_args=similarity_document_args, rerank_method=rerank_method, length_threshold=length_threshold)
        content_list = []

        prompt = ""
        c=0
        for k,v in content.items():
            if len(v)>0:
                for i in range(len(v)):
                    prompt += f"""\n{c+1}) {v[i]["title"]}:\n{v[i]["content"]}"""
                    content_list.append(f"""{v[i]["title"]}:\n{v[i]["content"]}""")
                    c += 1
            
        return prompt, content_list


    def prepare_content_prompts_all_for_evaluation(self,topic:str, similarity_document_args: dict = None, rerank_method:str="merge_auto", length_threshold:int=700):
       
        """
        Prepares content prompts for evaluation based on semantic search results for a given topic.

        Args:
            topic (str): The topic for which to prepare content prompts.
            similarity_document_args (dict, optional): Additional arguments for retrieving similarity documents. Defaults to None.
            rerank_method (str, optional): The reranking method to use. Defaults to "merge_auto".
            length_threshold (int, optional): The length threshold for the content. Defaults to 700.

        Returns:
            list: A list of strings each containing a title and its associated content.
        """
        content = self.get_semantic_search_result(topic, similarity_documents_args=similarity_document_args, rerank_method=rerank_method, length_threshold=length_threshold)
        content_list = []

        c=0
        for k,v in content.items():
            if len(v)>0:
                for i in range(len(v)):
                    content_list.append(f"""{v[i]}""")
            
        return content_list


    def save_results(self,data, path:str):

        return fo_utils.write_textual_file(data, path)

def main():
    #Example for clone ai
    # Configuration
    model_name = os.getenv("MODEL_NAME", "gemma2:27b") #"gemma2:9b", "llama3.3","gemma2:2b"

    user_info_path = os.getenv("USER_INFO_PATH", "./personal_info.txt")


    queries = ["What is the state of capitalism worldwide?",
        "Write a message about tourism in Turkey.",
        "What do you think about music in life?"]
    
    total_time = []
    for query in queries:
        clone = CloneAI(user_info_path,"CloneBot", query)
        st = time.time()
        res = clone.get_clone_response(model_name=model_name,topic="creating_clone_text", count=2)
        en = time.time()
        total_time.append(en-st)
        print(f"\nResponse: {res}",flush=True)
        print(f"Time Taken: {en-st}",flush=True)
        print("\n---------------------------------------------------------\n", flush=True)

    avg_total_time = sum(total_time) / len(total_time)
    print(f'\n Average Time Taken: {avg_total_time}')
if __name__ == "__main__": 
    main()   
   