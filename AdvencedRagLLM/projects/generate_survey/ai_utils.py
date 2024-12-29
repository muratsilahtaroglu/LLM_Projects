from typing import TYPE_CHECKING, List, Literal, Optional, TypedDict
from dataclasses import dataclass, field
import pandas as pd
import os, requests, json,re
import time, uuid, random
from jinja2 import Template
import sys
from langchain.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI
import ast
import logging
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from fastapi import HTTPException
from autogpt.llm.base import ChatSequence
from m_libs.client.ollama_client import ollama_client 
import openai
load_dotenv()
from transformers import AutoTokenizer
import requests
import json



GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MessageRole = Literal["system", "user", "assistant"]
MessageType = Literal["ai_response", "action_result"]


class MessageDict(TypedDict):
    role: MessageRole
    content: str

class CalculateTime:
    def __init__(self, info:str) -> None:
        self.info = info
    
    def __enter__(self):
        self.st = time.time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.et = time.time()
        elapsed_time = self.et - self.st
        print(f'\n{self.info} runtime:', elapsed_time, 'seconds')
        print(f'\n{self.info} runtime:', elapsed_time/60, 'minute')
        print(f'\n{self.info} runtime:', elapsed_time/(60*60), 'hour\n')
        
@dataclass
class Message:
    """OpenAI Message object containing a role and the message content"""

    role: MessageRole
    content: str

    def raw(self) -> MessageDict:
        return {"role": self.role, "content": self.content}


class Checks:
    #get_gemini_response(parapeters=data)
    def _is_response_ok(self,response:requests.Response,detail="Request failed with status code:"):
        if not response.ok:
            print("Request failed with status code:", response.status_code)
            time.sleep(60)
            return False
        return True

    def _is_valid_response(self, response_text:str,*args,**kwargs):

        valid_formats = [
    "[score: 5 (Strongly Agree), reason: From my tweets, it is clearly that",
    "[score: 4 (Agree), reason: From my tweets, it is clearly that",
    "[score: 3 (Neutral), reason: From my tweets, it is not clearly that",
    "[score: 2 (Disagree), reason: From my tweets, it is clearly that",
    "[score: 1 (Strongly Disagree), reason: From my tweets, it is clearly that"
        ]
        for format in valid_formats:
            if response_text.startswith(format):
                return response_text
        print("Response JSON does not contain 'text' key:", )
        #time.sleep(60)
        return False

checks = Checks()

class AISurveyResponse:
    def get_response_from_chat(self, messages, model = "gpt4-o", temperature= 0, max_tokens = 50):
        
        chat_completion_kwargs = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_key": OPEN_AI_API_KEY
        }
        response = openai.ChatCompletion.create(
        messages=messages.raw(),**chat_completion_kwargs
        )
        return response["choices"][0]["message"].content

    def get_base_messages(self, system_promt:str="", user_prompt:str = "", model:str = "gpt4-o"):
        messages = [
                    Message("system", system_promt),
                    Message("user", user_prompt),
                ]
        messages = ChatSequence.for_model(model, messages)
        return messages

    def get_gpt_responses(self, model:str = "gpt4-o" ,temperature=0.1,system_promt:str="", user_prompt:str = "",max_tokens=70,repeat_count=1):

        """
        Generates multiple responses using the GPT model with specified parameters.

        Parameters:
        model (str): The name of the model to use for generating the responses. Defaults to "gpt4-o".
        temperature (float): Controls randomness in the response generation. Defaults to 0.1.
        system_promt (str): The system prompt to generate a response for.
        user_prompt (str): The user prompt to generate a response for.
        max_tokens (int): The maximum number of tokens in the response. Defaults to 70.
        repeat_count (int): The number of times to generate a response. Defaults to 1.

        Returns:
        list: A list of generated response contents.
        """
        gpt_responses = []
        messages = self.get_base_messages(system_promt = system_promt, user_prompt= user_prompt,model=model)
        for _ in range(repeat_count):
            
            gpt_resp = self.get_response_from_chat(messages=messages,max_tokens= max_tokens, temperature= temperature,model=model )
            gpt_responses.append(gpt_resp)
            temperature += 0.1
        
        return gpt_responses

    def get_gemini_response(self, model:str = "gemini-pro", parameters={}):

        """
        Generates a response using the Gemini model with specified parameters.

        Parameters:
        model (str): The name of the model to use for generating the response. Defaults to "gemini-pro".
        parameters (dict): A dictionary containing parameters for the model, including:
            - "temperature" (float): Controls randomness in the response generation.
            - "top_p" (float): Nucleus sampling parameter.
            - "top_k" (int): Limits the sampling pool to top-k tokens.
            - "max_output_tokens" (int): The maximum number of tokens in the response.
            - "prompt" (str): The prompt text to generate a response for.

        Returns:
        str: The generated response content.
        """

        gemini_llm = ChatGoogleGenerativeAI(model=model, temperature=parameters["temperature"], top_p=parameters["top_p"],
                                            top_k=parameters["top_k"], max_output_tokens= parameters["max_output_tokens"])
        gemini_response = gemini_llm.invoke(parameters["prompt"]).content
            
        return gemini_response

    def get_ai_response(self, system_prompt: str = "", user_prompt: str = "", max_tokens: int = 70, repeat_count: int = 1):
        """
        Generates a response using the Gemini model with specified parameters.

        Parameters:
        system_prompt (str): The prompt text to generate a response for.
        user_prompt (str): The prompt text to generate a response for.
        max_tokens (int): The maximum number of tokens in the response.
        repeat_count (int): The number of responses to generate.

        Returns:
        list: A list of generated responses.
        """
        gemini_responses = []
        temperature = 0.1
        max_repeat_count = 3
        j = 0
        parameters = {
            "temperature": temperature,
            "top_p": 0.5,
            "top_k": 1,
            "max_output_tokens": 2048,
            "prompt": f"{system_prompt}\n{user_prompt}"
        }  
        parameters
        while j < max_repeat_count:
            print("repeat count: ",j)
            try:
                response = self.get_gemini_response(parameters=parameters)
                if checks._is_valid_response(response):
                    gemini_responses.append(response)
                    break
                j += 1
            except Exception as e:
                print("Gemini response error: ",e)
                parameters["temperature"] = temperature
                temperature += 0.1
                j += 1
        
        if  len(gemini_responses) == 0:
            temperature = 0.1
            # If gemini_responses is still empty, try with alternative model
            j = 0
            while j < max_repeat_count:
                try:
                    response = self.get_gpt_responses(model="gpt-4", system_promt=system_prompt,temperature=temperature, prompt=user_prompt, max_tokens=80, repeat_count=1)
                    
                    if response:
                        print("GPT Response: ", j)
                        is_valid = checks._is_valid_response(response[0])
                        if is_valid:
                            gemini_responses.extend(response)
                            break
                    j +=1
                except Exception as e:
                    print("GPT response error: ",e)
                    j +=1
                    temperature += 0.1
                    
        if len(gemini_responses)==0:
            
            print("default answer")
            return ["[score: 3 (Neutral), reason:  From my tweets, it is not clearly that]"] * repeat_count
        return gemini_responses[:repeat_count]
    
    def get_ollama_response(self, system_prompt: str = "", user_prompt: str = "", max_tokens: int = 70, repeat_count: int = 2, port="11437"):
        """
        Get responses from ollama service.

        Parameters:
        system_prompt (str): The prompt text to generate a response for.
        user_prompt (str): The prompt text to generate a response for.
        max_tokens (int): The maximum number of tokens in the response.
        repeat_count (int): The number of responses to generate.
        port (str): The port number for the ollama service.

        Returns:
        list: A list of generated responses.
        """
        ollama_ai = ollama_client.OllamaClient(port=port)
        multi_threading_count = repeat_count
        model_name = "gemma2:27b" #llama3.3:70b "gemma2:27b", "phi4:14b"
        parameters = {}
        for _ in range(multi_threading_count):
            FULL_PROMPT = system_prompt + "\n" + user_prompt
            ai_uudi = str(uuid.uuid4())
            parameters.update({ai_uudi: {"model_name": model_name, "prompt": FULL_PROMPT, "task": "survey_data","check_response_func":checks._is_valid_response,"options":None}})
           
        responses = ollama_ai.get_all_responses(parameters, same_task=True)
        ollama_responses = []
        if len(responses):
            for ai_uudi, response in responses.items():
                if isinstance(response, list):
                    ollama_responses.extend(response)
                else:
                    ollama_responses.append(response)
        ollama_responses = list(set(ollama_responses))
        random.shuffle(ollama_responses)
        return ollama_responses
