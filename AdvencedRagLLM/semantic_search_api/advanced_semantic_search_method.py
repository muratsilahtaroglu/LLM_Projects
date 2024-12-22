
import os
import numpy as np
import time
from uuid import UUID, uuid4
from textwrap import dedent
import datetime as dt
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
import logging
import json
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings,HuggingFaceInstructEmbeddings

from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores.usearch import USearch
from langchain.vectorstores.chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import ast
import math
from typing import Any, List, Tuple, Dict,Literal, Optional
from collections import defaultdict
import numpy as np
import torch
try:
    from semantic_search_api import shared_utils, ollama_client
except:
    import shared_utils, ollama_client

class TempDb:
    
    def __init__(self, *args) -> None:
        self.args = args

    def __enter__(self):
        return self 

    def __exit__(self, exc_type, exc_value, traceback):
        for arg in self.args:
            if arg:
                del arg
    
class SemanticSearchApp:
    """
    SemanticSearchApp class is designed to manage and initialize a semantic search application.
    It handles the setup of the embedding models, vector database, and the configuration of
    a large language model (LLM) for processing and querying data.

    Attributes:
        embedding_type (Literal): The type of embedding model to use, such as "hugging_face" or "open_ai".
                                  Defaults to "hugging_face".
        llm_name (str): The name of the large language model (LLM) to use. Defaults to "gemini".
                        If "gemini" or an empty string is provided, the Gemini model from 
                        Google Generative AI is used.
        device (str): The device to be used for computation, e.g., "cuda:0" for GPU or "cpu".
                      Defaults to "cuda:0".
        query_and_releations (List[dict]): A list of dictionaries that contains the queries and their relationships.
                                           This data is used for semantic search operations.
        all_data (list): A list containing all the data that will be processed or queried.
                         This might include documents, text snippets, or any other relevant information.
        persist_directory (str): The directory where the vector database will be stored or loaded from.
                                 Defaults to "vectordb".
        collection_name (str): The name of the vector database collection to use.
                               Defaults to "langchain".
    """
    def __init__(self,embedding_type:Literal["hugging_face","open_ai"]="hugging_face",llm_name="local_ai",device="cuda:0",query_and_releations=None,all_data=None,all_data_count=0,persist_directory="vectordb",collection_name="langchain"):
        if llm_name is None or llm_name =="" or llm_name.lower()=="local_ai":
            llm = ollama_client.OllamaClient()
        elif llm_name.lower()=="gemini":
            llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)
        elif llm_name.lower() == "open_ai":
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,  # Adjust the temperature as needed
                max_tokens=2000,   # Adjust the max tokens as needed
                timeout=60,       # Adjust the timeout as needed
                max_retries=2
            )
        
        self.llm_name = llm_name
        self.llm = llm
        self.all_data = [] if all_data is None else all_data
        self.all_data_count = all_data_count
        self.query_and_releations:List[dict] = [] if query_and_releations is None else query_and_releations
        self.persist_directory = persist_directory
        self.collection_name=collection_name
        self.embedding_type = embedding_type
        self.device= device
        self.initialize()
    
    def initialize(self):
        """
        Initializes the embedding model and the vector database. This method is called during the
        instantiation of the SemanticSearchApp class.
        """
        self.set_embedding()
        self.set_main_vector_db()

    def to_dict(self):
        """
        Converts the SemanticSearchApp instance into a dictionary representation.
        This method allows the object's state to be easily serialized or passed around
        as a dictionary.

        Returns:
            dict: A dictionary containing the current state of the SemanticSearchApp instance.
                The keys correspond to the attribute names, and the values are the current
                values of those attributes.
        """
        return {
            "llm_name": self.llm_name,
            "embedding_type": self.embedding_type,
            "device":self.device,
            #"all_data": self.all_data,
            "all_data_count": self.main_vectordb._collection.count(),
            "query_and_releations": self.query_and_releations,
            "persist_directory":self.persist_directory,
            "collection_name":self.collection_name
        }
    
    @staticmethod
    def from_dict(d:dict) -> List['SemanticSearchApp']:
        ...
            
    def set_embedding(self):
        """
        Sets the embedding model based on the specified embedding type. The method checks if an embedding model
        already exists in the `embeding_models` dictionary. If not, it initializes a new embedding model
        according to the specified type and device.

        The method also handles GPU and CPU allocation, raising an exception if an invalid device is specified.

        Raises:
            HTTPException: If the specified CUDA device ID is not found among the available devices.
        """
        
        # Check if the embedding type already exists in the embeding_models dictionary.
        embedding = shared_utils.embeding_models.get(self.embedding_type)
        # Determine if CUDA is available. If not, default to CPU.
        if not torch.cuda.is_available():
            self.device = "cpu"
        else:
            # Generate a list of available CUDA devices.
            cuda_list = list(map(lambda x: f"cuda:{x}", range(torch.cuda.device_count())))
        if embedding:
            print(f"Existing {self.embedding_type} embedding database set", flush=True)
            return

        
        # Set the model's device configuration based on the available devices and user input.
        if self.device == "cpu":
            model_kwargs = {'device': 'cpu'}
        elif self.device in cuda_list:
            model_kwargs = {"device": self.device}
        else:
            # Raise an exception if the specified CUDA device is not found.
            raise HTTPException(status_code=702, detail=f"Device is {self.device}. CUDA ID not found. You can try one of {cuda_list}")
        
        # Initialize the embedding model based on the specified embedding type.
        if self.embedding_type == "hugging_face_instruct":
            embedding = HuggingFaceInstructEmbeddings()
        elif self.embedding_type == "open_ai":
            embedding = OpenAIEmbeddings()
        elif self.embedding_type == "hugging_face":
            embedding = HuggingFaceEmbeddings(model_kwargs=model_kwargs)
        else:
            # If a custom embedding type is provided, initialize with the specified model name.
            embedding = HuggingFaceEmbeddings(model_name=self.embedding_type, model_kwargs=model_kwargs)
        #TODO: add ollama support for embedding models
        # Store the initialized embedding model in the embeding_models dictionary for future use.
        shared_utils.embeding_models[self.embedding_type] = embedding
        print(f"{self.embedding_type} Embedding database created", flush=True)
            
    def set_main_vector_db(self):
        """
        Initializes and sets up the main vector database (vector DB) for storing and querying embeddings.
        The method uses the embedding model specified by `self.embedding_type` and associates it with a
        persistent storage directory for the vector database.

        Workflow:
        1. Retrieve the embedding model from the `embeding_models` dictionary based on `self.embedding_type`.
        2. Ensure the persistence directory exists or create it if it doesn't.
        3. Initialize the vector database (`Chroma`) with the specified embedding function, persistence directory,
        and collection name.
        4. Retrieve and print the count of existing documents in the collection.

        Attributes:
            main_vectordb (Chroma): The initialized vector database using the specified embedding model and
                                    persistent storage.

        Raises:
            KeyError: If the embedding model for `self.embedding_type` is not found in `embeding_models`.
        """
        
        # Retrieve the embedding model from the `embeding_models` dictionary.
        embedding = shared_utils.embeding_models.get(self.embedding_type)
        
        # If the embedding is not found, the application may raise an error here (though not explicitly handled in this code).
        # Ensure the persistence directory exists; if not, create it.
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize the Chroma vector database with the persistence directory, embedding function, and collection name.
        self.main_vectordb = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embedding,
            collection_name=self.collection_name
        )
        
        # Retrieve the count of documents in the collection to give feedback about the existing data.
        doc_count = self.main_vectordb._collection.count()
        
        # Print the number of existing elements in the collection.
        print(f"There are {doc_count} existing element(s) in the {self.collection_name} collection (directory: {self.persist_directory})")
        
    def create_or_get_vector_db_data_is_non_exist(self, data=None, metadatas=None):
        """
        This method checks the current state of the vector database to determine if the provided data already exists.
        If the data does not exist in the database, it adds the new data to the vector database. The method also returns
        a message describing the operation performed.

        Args:
            data (list, optional): A list of data elements (e.g., text documents) to be added to the vector database.
                                Defaults to None.
            metadatas (list, optional): A list of metadata associated with each data element. Defaults to None.

        Returns:
            str: A response message indicating the number of elements already existing in the database, 
                the number of new elements added, and the total number of elements in the database after the operation.
        """

        # Get the current document count in the vector database.
        doc_count = self.main_vectordb._collection.count()
        existing_data, temp_data = [], []

        # If no data is provided, return a message indicating the current document count.
        if not len(data):
            response = (f"No new data elements provided. "
                        f"There are {doc_count} existing element(s) in the {self.collection_name} collection "
                        f"(directory: {self.persist_directory}).")
            return response

        # If the database is not empty, check for existing data.
        elif doc_count > 0:
            # Perform a similarity search to find existing documents in the database.
            collection = self.main_vectordb._collection
            all_data = collection.get(include=['metadatas', 'documents', 'embeddings'])
            documents = all_data["documents"]
            #existing_data = [doc_and_score[0].page_content for doc_and_score in docs_and_scores]
            existing_data = documents
            # Filter out the data that already exists in the database.
            existing_data_set = set(existing_data)
            # Use list comprehension for efficient filtering
            if len(existing_data):
                temp_data = data
                unexisting  = [(t_d, metadatas[i]) for i, t_d in enumerate(data) if t_d not in existing_data_set]
                
                # Unpack filtered results into data and metadatas
                data, metadatas = zip(*unexisting ) if unexisting  else ([], [])
            # if len(existing_data):
            #     temp_data = data
            #     temp_metadatas = metadatas
            #     metadatas = []
            #     data = []
            #     for i, t_d in enumerate(temp_data):
            #         if t_d not in existing_data:
            #             data.append(t_d)
            #             metadatas.append(temp_metadatas[i])

        # Print the status of existing elements in the collection.
        print(f"There are {doc_count} existing element(s) in the {self.collection_name} collection (directory: {self.persist_directory}).")

        # Combine existing data with the new data.
        self.all_data = existing_data + data
        print(f" In the {self.collection_name}.")
        # Add new texts to the vector database.
        if len(data) and isinstance(data[0], str):
            chuck_n = math.ceil(len(data) / 40_000)
            for n in range(chuck_n):
            
                self.main_vectordb.add_texts(texts=data[n*40_000:(n+1)*40_000], metadatas=metadatas[n*40_000:(n+1)*40_000], ids=None)
                doc_count = self.main_vectordb._collection.count()
                res = (f"Added {len(data[n*40_000:(n+1)*40_000])} new elements. "
                        f"The total number of elements in the database is now {doc_count}.")
                print( res,flush=True)
        doc_count = self.main_vectordb._collection.count()
        # Prepare the response message based on the operation performed.
        if len(temp_data) - len(existing_data):
            response = (f"{len(temp_data) - len(data)} elements already existed in the database. "
                        f"Therefore, only {len(data)} new elements were added. "
                        f"The total number of elements now is {doc_count}.")
        else:
            response = (f"Added all {len(data)} new elements. "
                        f"The total number of elements in the database is now {doc_count}.")

        return response

        
    def create_vector_db(self, data: List[str | Document], metadatas: List[dict] = None):
        """
        Creates or updates the vector database with the provided data and metadata. This method checks
        if the data already exists in the database using the `create_or_get_vector_db_data_is_non_exist` method.
        If the data does not exist, it is added to the vector database.

        Args:
            data (List[str | Document]): A list of data elements to be added to the vector database.
                                        Each element can be a string (e.g., text) or a Document object.
            metadatas (List[dict], optional): A list of metadata dictionaries corresponding to each data element.
                                            These are optional and default to None.

        Returns:
            str: A response message indicating the result of the operation, such as the number of elements
                added or already existing in the database.

        Raises:
            Exception: Catches and logs any exceptions that occur during the operation.
        """
        try:
            # Check for existing data and add new data if it does not already exist in the database.
            response = self.create_or_get_vector_db_data_is_non_exist(data, metadatas)

            # Print the response message to provide feedback on the operation.
            print(response, flush=True)

            # Clear the CUDA cache to free up memory after the operation.
            torch.cuda.empty_cache()

            # Return the response message.
            return response

        except Exception as e:
            # Log any errors that occur during the operation, providing the exception message for debugging.
            logging.error(f"Error initializing SemanticSearchApp: {str(e)}")

    # method that breaks down the given question into smaller and more specific questions and keywords
    def get_sub_questions_keywords_by_llm(self, query:str, local_prompt_path:str=None, prompt:str="",error:str=""):
        """
        Breaks down the given query into smaller, more specific sub-questions and keywords using a 
        large language model (LLM). The method can use a custom prompt provided by the user, load a prompt
        from a file, or fall back to a default prompt if none is provided.

        Args:
            query (str): The main query that needs to be broken down into sub-queries and keywords.
            local_prompt_path (str, optional): The path to a local file containing a custom prompt template. 
                                            The template must include a placeholder for the query (e.g., "{question}" or "{query}").
            prompt (str, optional): A custom prompt string provided directly by the user. This prompt should
                                    include placeholders for the query if necessary. Defaults to an empty string.
            error (str, optional): An optional error message or additional context to be included in the prompt.
                                This can be used to provide feedback or corrections based on previous attempts.
                                Defaults to an empty string.

        Returns:
            dict: A dictionary containing sub-queries, helper keywords, and main keywords generated by the LLM,
                with the main query included for reference.

        Raises:
            Exception: Catches and logs any exceptions that occur during the prompt processing or LLM invocation.
        """
        try:
            # Use the provided prompt if it's not empty
            if  prompt and prompt != "":
                prompt = prompt
            # If a local prompt path is provided, load the prompt from the file
            elif  local_prompt_path and local_prompt_path != "":
                if os.path.exists(local_prompt_path):
                    with open(local_prompt_path, 'r') as file:
                        # Read the entire file content
                        file_content = file.read()
                        # Replace placeholders in the template with the actual query
                        if "{question}" in file_content:
                            prompt = file_content.format(question=query)
                        
                        elif "{query}" in file_content:
                            prompt = file_content.format(query=query)
                        
                        else:
                            raise "prompt must be include question or query variable"
                else:
                    raise f"local_prompt_path: {local_prompt_path} is not found"
            # If no custom prompt is provided, use a default prompt
            
                #- If you have a query about a country, you can also use the most well-known cities at helper_keywords
#                 prompt = dedent(f"""
# You are an assistant who does semantic search. To help you, before doing Semantic search, you convert the main query into sub-queries and keywords.
# - Users will ask you questions and their aim is to get relevant places  with semantic search.
# ### Answer Format:
# {{"sub_queries": ["Sub-query 1", "Sub-query 2", ...],
# "helper_keywords": ["Keyword 1", "Keyword 2", ...],
# "keywords": ["Keyword 1", "Keyword 2", ...]}}"

# ### Answer Format Explanation:
# - sub_queries should be smaller, more specific questions that lead to answering the main query.
# - keywords are important words that help semantic search only in the main query. Avoid generic terms; focus on specific terms directly tied to the query's intent.
# - helper_keywords are important words and phrases that can be used to aid semantic search and search for information or ask sub-queries. These should be more precise than keywords but still relevant to aid in identifying the correct information.
# - Avoid using overly broad or generic terms in both helper_keywords and keywords to ensure only the most relevant information is retrieved.
# - Ensure sub_queries, helper_keywords, and keywords are tailored to effectively target the intended search results for the main query.
# - Respond only with the output in the exact format specified in the above Answer Format, with no explanation or conversation.

# ## Example
# ### Main Query:
# Karadelik ve Yerçekimi arasında nasıl bir ilişki vardır?
# ### Answer: (Please Your response format is in English, but create sub_queries, helper_keywords and keywords in the language of the Main Query.)
# {{
# "sub_queries": ["Karadeliklerin yerçekimi üzerindeki etkisi nedir?", "Yerçekimi karadeliklerde nasıl değişir?", "Karadeliklerin yerçekimi ile ilişkisi nasıldır?"],
# "helper_keywords": ["karadelik yerçekimi etkisi", "yerçekimi karadelik ilişkisi", "karadeliklerde yerçekimi değişimi"],
# "keywords": ["karadelik", "yerçekimi"]
# }}
# ### Main Query:
# Naim nerede doğdu?
# ### Answer: (Please Your response format is in English, but create sub_queries, helper_keywords and keywords in the language of the Main Query.)
# {{
# "sub_queries": ["Naim'in doğum yeri nedir?", "Naim hangi şehirde doğdu?", "Naim'in memleketi nedir?"],
# "helper_keywords": ["Naim doğum yeri", "Naim doğum", "Naim memleket","Naim Şehir"],
# "keywords": ["Naim","Doğum"]
# }}
# ### Last Main Query:
# {query}
# {error}
# ### Answer: (Please Your response format is in English, but create sub_queries, helper_keywords and keywords in the language of the Last Main Query.)
# """)
            else:
                prompt = dedent(f"""
    You are an assistant who does semantic search. Before performing Semantic search, convert the main query into sub-queries and keywords.
    - Users will ask questions to receive relevant places using semantic search.
    ### Important Guidelines:
    1. **Output Format**: Only use the following format:
    {{"main_query_language":"Detected Language",
    "sub_queries": ["Sub-query 1", "Sub-query 2", ...],
    "helper_keywords": ["Keyword 1", "Keyword 2", ...],
    "keywords": ["Keyword 1", "Keyword 2", ...]}}
    2. **Formatting Requirements**:
    - Avoid `\\n`, `\\`, `|`, and unnecessary JSON-like indicators such as `"```json"` in the response.
    - Respond *exactly* in the specified format. Do not add explanations or additional characters.

    ### Answer Format Explanation:
    - `sub_queries` should contain smaller, specific questions that can lead to answering the main query.
    - `keywords` are the important terms specific to the query, used to aid semantic search.
    - `helper_keywords` should be targeted phrases that help narrow down information for the query.
    - Ensure **sub_queries**, **helper_keywords**, and **keywords** effectively target the main query’s intent.
    - Respond only with the output in the exact format specified in the above Output Format, with no explanation or conversation.

    ### Language Guidelines:
    - Respond in the language of the last query. If the query is in English, respond in English. If the query is in another language, adjust sub_queries, helper_keywords, and keywords to match that language.
    
    ## Example-1
    ### Main Query:
    Karadelik ve Yerçekimi arasında nasıl bir ilişki vardır?
    ### Answer:
    {{
    "main_query_language": "Turkish",
    "sub_queries": ["Karadeliklerin yerçekimi üzerindeki etkisi nedir?", "Yerçekimi karadeliklerde nasıl değişir?", "Karadeliklerin yerçekimi ile ilişkisi nasıldır?"],
    "keywords": ["karadelik", "yerçekimi"],
    "helper_keywords": ["karadelik yerçekimi etkisi", "yerçekimi karadelik ilişkisi", "karadeliklerde yerçekimi değişimi"],
    }}
    
   ## Example-2
   ### Main Query:
   Is the purpose of your visit to Turkey shopping?
   ### Answer:
    {{
        "main_query_language": "English",
        "sub_queries": [ "What are you planning to do in Turkey?", "Are you interested in buying goods in Turkey?"],
        "keywords": ["Turkey","visit","shopping"],
        "helper_keywords": ["visit purpose Turkey","shopping intentions Turkey"]
   }}
   
    ### Last Main Query: 
    {query}
    {error}
    ### Answer:
    """)

            # Invoke the LLM with the prepared prompt
            response = self.llm.invoke(prompt)
           # Convert the response content from the LLM into a Python dictionary
            if hasattr(response, "content"):
                sub_questions_keywords = ast.literal_eval(response.content)
            else:
               sub_questions_keywords = ast.literal_eval(response)
          
                
            # Add the original query to the response for reference
            sub_questions_keywords["Query"] = query
            return sub_questions_keywords

        except Exception as e:
            # Log any errors that occur during the process
            logging.error(f"Error getting sub questions and keywords: {str(e)}")
        
    def set_query_and_releations(self, query:str="", local_prompt_path:str = "", prompt:str="",error= "",query_uuid:str=uuid4()):
        """
        This method generates sub-queries, keywords, and helper keywords for a given query and stores the result 
        in `query_and_releations`. It first checks if the query already exists in the `query_and_releations` list. 
        If the query is not found, it invokes the large language model (LLM) to generate the necessary components.

        Args:
            query (str, optional): The main query for which sub-queries and keywords are to be generated. Defaults to an empty string.
            local_prompt_path (str, optional): The path to a local file containing a custom prompt template. Defaults to an empty string.
            prompt (str, optional): A custom prompt string provided directly by the user. Defaults to an empty string.
            error (str, optional): An optional error message or additional context to be included in the prompt. Defaults to an empty string.
            query_uuid (str, optional): A unique identifier for the query, used to identify it within `query_and_releations`. Defaults to a newly generated UUID.

        Returns:
            dict: A dictionary containing the query, sub-queries, keywords, helper keywords, and other related information.

        Workflow:
        1. Check if the query already exists in `query_and_releations`.
        2. If the query does not exist, generate sub-queries, keywords, and helper keywords using the LLM.
        3. Attempt to retrieve sub-queries and keywords up to 5 times, adjusting the LLM's temperature to improve output.
        4. Once successfully generated, store the result in `query_and_releations`.
        5. Return the stored result.

        Raises:
            Exception: Logs and prints any errors encountered during the LLM invocation or data processing.
        """
        i = 0
        # Check if the query already exists in the `query_and_releations` list.
        #TODO:for ile yapılan işemler numpy ile yapılacak
        
        if len(self.query_and_releations):
            for  query_and_releation in self.query_and_releations:
                if list(query_and_releation.values())[0]["query"] ==query:
                    return query_and_releation
        current_dir = os.path.dirname(os.path.abspath(__file__))
        all_query_and_relations_path = os.path.join(current_dir, "all_query_and_relations.json")
        try:
            with open(all_query_and_relations_path, 'r', encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception as e:
            print(e)
            existing_data = {}
        if existing_data and query in existing_data:
            first_entry = list(existing_data.values())[0]
            sub_questions, keywords, helper_keywords, keywords_join_list = (first_entry["sub_queries"],first_entry["keywords"],first_entry["helper_keywords"],first_entry["keywords_join_list"])
            
        
        else:
            sub_questions = []
            keywords = []
            helper_keywords = []
            self.llm.temperature = 0.1
            # Attempt to generate sub-queries and keywords using the LLM, retrying up to 5 times if necessary.

            while i<5:
                try:
                    time.sleep(2) # Pause briefly between attempts
                    sub_questions_keywords = self.get_sub_questions_keywords_by_llm(query,local_prompt_path, prompt,error)
                    sub_questions = sub_questions_keywords["sub_queries"]
                    keywords = sub_questions_keywords["keywords"]
                    helper_keywords = sub_questions_keywords["helper_keywords"]
                    i=5 # Exit the loop if successful
                    
                except Exception as e:
                    time.sleep(5) # Wait longer before retrying if an error occurs
                    self.llm.temperature += 0.1 # Adjust the LLM temperature to control creativity
                    i+=1
                    error = "- Attention please. Respond only with the output in the exact format specified in the above Answer Format, with no explanation or conversation."
                    logging.error(f"{i}.error: {str(e)}")
            # Combine keywords and helper keywords into a single list for easier processing
            keywords_join_list = [" ".join(keywords)]
            keywords_join_list.append(" ".join(helper_keywords))
        #keywords_join_list.append(" ".join([keyword for keyword in keywords if keyword in query]))
        
        # Create the final dictionary entry for the query and its related information
        query_and_releation = {
                    str(query_uuid): {
                        "query":query,
                        "sub_queries": sub_questions,
                        "keywords": keywords,
                        "helper_keywords": helper_keywords,
                        "keywords_join_list": keywords_join_list,
                        "created_at": dt.datetime.now().isoformat()
                    }
                }
        # Add the new entry to the `query_and_releations` list
        self.query_and_releations.append(query_and_releation)
        return query_and_releation
        
    def get_merge_results(self, threshold_score:float=0.5, max_result_n:int=5, all_queries:list=None, external_help_queries = None,all_queries_state:bool=True,sub_questions_state:bool=True,
                          main_keywords_state:bool=True, helper_keywords_state:bool=False, keywords_join_list_state:bool=True, merge_type:str="sum", k:int=10,
                          query_and_releations=None, coefficient=None, coefficient_helpers=None):
        help_queries = []
        embedding = shared_utils.embeding_models.get(self.embedding_type)
        if len(all_queries)>0 and all_queries_state:
            
            all_queries_vector_db = USearch.from_texts(all_queries, embedding=embedding )
    
            help_documents  = self.get_similarity_text(all_queries_vector_db, query=query_and_releations["query"], threshold_score=0.3)
            del all_queries_vector_db
            
            help_queries = [unique_document["page_content"] for unique_document in help_documents if unique_document["page_content"] != query_and_releations["query"]]

        states_and_keys = [(sub_questions_state, "sub_queries"),(main_keywords_state, "keywords"),(helper_keywords_state, "helper_keywords"),(keywords_join_list_state, "keywords_join_list")]
        for state, key in states_and_keys:
            if state:
                help_queries.extend(query_and_releations[key])

        help_queries = help_queries[1:] if len(help_queries[0]) == 0 else help_queries
            
        if len(help_queries):
            self.help_texts_vector_db = USearch.from_texts(help_queries, embedding=embedding)
            if  "auto" in merge_type:
                with TempDb(self.help_texts_vector_db):
                    return self.get_auto_merge_similaritiy_texts(threshold_score=threshold_score,merge_type=merge_type, search_n=k, merged_n=max_result_n, query_and_releations= query_and_releations,
                                                        h_coefficient=coefficient,coefficient_helpers=coefficient_helpers)
            with TempDb(self.help_texts_vector_db):
                return self.get_merge_similaritiy_texts(threshold_score=threshold_score,merge_type=merge_type, search_n=k, merged_n=max_result_n, query_and_releations= query_and_releations,
                                                        h_coefficient=coefficient,coefficient_helpers=coefficient_helpers)
        else:
            return []
        
    def get_similarity_texts(self, db:USearch|Chroma, queries: list, k: int = 4, threshold_score: float = 0.9,max_result_n:int=15, metadata_filter=None):
        docs_and_scores = []
        score_k=1
        if not self.embedding_type in shared_utils.embeding_models:
            self.initialize()
            db = self.main_vectordb
        for query in queries:
            if isinstance(db,Chroma):
                docs_and_scores.extend(db.similarity_search_with_score(query, k=k,filter=metadata_filter))
                #score_k=1000
            else:
                docs_and_scores.extend(db.similarity_search_with_score(query, k=k))
        return self._get_unique_similarity_documents(docs_and_scores, threshold_score,max_result_n,score_k)
        
    def get_similarity_text(self, db: Chroma | USearch, query: str, k: int = 4, threshold_score: float = 0.5, max_result_n: int = 15, metadata_filter=None):
        """
        Retrieves similar documents from a vector database based on a query. The method can handle different
        types of vector databases (e.g., Chroma or USearch) and returns unique, relevant documents based on
        similarity scores and specified thresholds.

        Args:
            db (Chroma | USearch): The vector database instance used to perform the similarity search.
            query (str): The query string used to find similar documents in the vector database.
            k (int, optional): The number of top documents to retrieve from the database. Defaults to 4.
            threshold_score (float, optional): The score threshold for filtering documents. Only documents
                                            with a score below this threshold are considered. Defaults to 0.5.
            max_result_n (int, optional): The maximum number of unique documents to return. Defaults to 15.
            metadata_filter (dict, optional): An optional filter to apply to the search based on document metadata.

        Returns:
            list: A list of unique, relevant documents that meet the threshold score criteria, limited by max_result_n.
        """

        #score_k = 1  # Default score scaling factor
        if not self.embedding_type in shared_utils.embeding_models:
            self.initialize()
            db = self.main_vectordb
        # Perform similarity search with scoring based on the type of database
        if isinstance(db, Chroma):
            docs_and_scores = db.similarity_search_with_score(query, k=k, filter=metadata_filter)
            #score_k = 1000  # Adjust score scaling factor for Chroma
            #threshold_score = 1000*threshold_score # TODO:Buraya tekrar bak
        else:
            docs_and_scores = db.similarity_search_with_score(query, k=k)
        
        # Clear the CUDA cache to manage GPU memory
        torch.cuda.empty_cache()
        
        # Process the retrieved documents and return unique results
        return self._get_unique_similarity_documents(docs_and_scores, threshold_score, max_result_n)

    def _get_unique_similarity_documents_new(self, docs_and_scores, threshold_score, max_result_n=15, score_k=1):
        """
        Processes a list of documents and their similarity scores, groups identical documents,
        averages their scores, and filters based on a threshold score, returning unique results.

        Args:
            docs_and_scores (list): List of tuples (document, similarity_score).
            threshold_score (float): Minimum threshold for a document to be included.
            max_result_n (int, optional): Maximum number of unique documents to return. Defaults to 15.
            score_k (int, optional): Factor to normalize scores before averaging. Defaults to 1.

        Returns:
            list: A list of unique documents that meet the threshold criteria, sorted by relevance.
        """
        # Convert to dictionary format and normalize scores
        docs_and_scores_dict = [
            {**doc.__dict__, "scores": float(s) if isinstance(s, np.float32) else s}
            for doc, s in docs_and_scores
        ]
        
        # Find and normalize by the maximum score if necessary
        max_score = max(doc["scores"] for doc in docs_and_scores_dict)
        if max_score > 1:
            score_k = max_score

        # Group documents by content, normalize, and calculate mean scores
        doc_groups = defaultdict(list)
        for doc in docs_and_scores_dict:
            doc_groups[doc["page_content"]].append(doc["scores"] / score_k)

        # Filter groups by the threshold score
        docs_texts_and_scores = [
            (text, np.mean(scores))
            for text, scores in doc_groups.items()
            if np.mean(scores) < threshold_score
        ]

        if not docs_texts_and_scores:
            return []  # No documents meet the criteria

        # Sort by scores (ascending)
        docs_texts_and_scores.sort(key=lambda x: x[1])

        # Handle cases with no metadata
        if not docs_and_scores_dict[0].get("metadata"):
            return docs_and_scores_dict[:max_result_n]

        # Collect unique documents based on metadata
        unique_data_metadata = set()
        unique_documents = []

        for text, score in docs_texts_and_scores:
            for doc in docs_and_scores_dict:
                if (
                    doc["page_content"] == text
                    and doc["metadata"] not in unique_data_metadata
                ):
                    unique_data_metadata.add(doc["metadata"])
                    doc["scores"] = score
                    unique_documents.append(doc)

        # Filter documents for uniqueness based on 'data.text' in metadata
        unique_common_data_text = set()
        unique_common_documents = []

        for doc in unique_documents:
            current_text = doc["metadata"].get("data.text", "")
            if not any(
                current_text in existing_text or existing_text in current_text
                for existing_text in unique_common_data_text
            ):
                unique_common_data_text.add(current_text)
                unique_common_documents.append(doc)

        # Return the top results
        return unique_common_documents[:max_result_n]
    def _get_unique_similarity_documents(self,docs_and_scores, threshold_score, max_result_n:int=15,score_k=1):
        """
        This method processes a list of documents and their similarity scores, grouping identical documents,
        averaging their scores, and filtering based on a threshold score. The method ensures that only unique
        documents are returned, with an option to limit the number of results.

        Args:
            docs_and_scores (list): A list of tuples where each tuple contains a document and its similarity score.
            threshold_score (float): A threshold below which documents are considered relevant.
            max_result_n (int, optional): The maximum number of unique documents to return. Defaults to 15.
            score_k (int, optional): A factor by which each score is divided before averaging. Defaults to 1.

        Returns:
            list: A list of unique documents that meet the threshold score criteria, limited by max_result_n.
        """
        # Convert docs_and_scores into a dictionary format, adjusting scores if necessary.
        docs_and_scores_dict = [{**doc.__dict__, "scores": float(s) if isinstance(s, np.float32) else s} for doc, s in docs_and_scores]
        m = 0
        #find max score
        for doc in docs_and_scores_dict:
            if m < doc["scores"]:
                m = doc["scores"]
        if m >1:
            score_k = m
        # Group documents by their content, averaging their scores.
        doc_groups = defaultdict(list)
        for doc in docs_and_scores_dict:
            doc_groups[doc["page_content"]].append(doc["scores"]/score_k)
        # Calculate the mean score for each group and filter based on the threshold score.
        docs_texts_and_scores = [(text, np.mean(scores)) for text, scores in doc_groups.items()]
        docs_texts_and_scores = [(text,score) for text, score in docs_texts_and_scores if score<threshold_score]
         # If no documents meet the criteria, return an empty list.
        if not docs_texts_and_scores:
            return []
        
        # Sort the documents by their scores in ascending order (lower scores are more relevant).
        docs_texts_and_scores = sorted(docs_texts_and_scores, key=lambda x: x[1])

        unique_data_metadata = []
        unique_documents = []
        unique_common_documents = []
        unique_common_data_text = [] 
        # If there is no metadata in the first document, return the top unique documents.
        if not docs_and_scores_dict[0]["metadata"]:
            return list(docs_and_scores_dict)[:max_result_n]
        
        # Iterate through the sorted documents and collect unique ones based on metadata.
        for text, score in docs_texts_and_scores:
            for doc in docs_and_scores_dict:
                if doc["page_content"] == text and doc["metadata"] not in unique_data_metadata:
                    
                        unique_data_metadata.append(doc["metadata"])
                        doc["scores"] = score
                        unique_documents.append(doc)
        
        # Filter out documents to ensure uniqueness based on 'data.text' in metadata.
        for doc in unique_documents:
            current_text = doc["metadata"]["data.text"]
            # Check if the current text is unique by ensuring no existing text is a subset or superset of it.

            if not any(current_text in existing_text or existing_text in current_text for existing_text in unique_common_data_text):
                unique_common_data_text.append(current_text)
                unique_common_documents.append(doc)
        # Return the top unique documents based on the maximum number specified.
        return list(unique_common_documents)[:max_result_n]
    
    def get_auto_merge_similaritiy_texts(self,threshold_score=10, merge_type:str="auto1", search_n=None,merged_n=5, query_and_releations=None,h_coefficient=1.5, coefficient_helpers=None) -> Tuple[List[str], List[float]]:
        merge_types = ['sum', 'proud', 'square', 'square_sum', 'square_sum2', 'square_proud', 'square_proud2']
        if merge_type=="auto1":
            merged_texts = []
            for m_t in merge_types:
                similaritiy_texts = self.get_merge_similaritiy_texts(threshold_score=1.0, merge_type=m_t, search_n=200, 
                                                merged_n=None, query_and_releations= query_and_releations,
                                                h_coefficient=h_coefficient,coefficient_helpers=coefficient_helpers)
                merged_texts.extend(similaritiy_texts)
        elif merge_type=="auto2":
            pass
        elif merge_type=="auto3":
            pass
        doc_groups = defaultdict(lambda: {"doc": None,"page_content":"", "total_score": 0})
        if not merged_texts:
            return []
        for doc in merged_texts:
            metadata = doc["metadata"]
            metadata["page_content"] = doc["page_content"]
            if doc_groups[str(metadata)]["doc"] is None:
                doc_groups[str(metadata)]["page_content"] = doc["page_content"]
                del doc["page_content"]
                doc_groups[str(metadata)]["doc"] = doc
            #TODO:Burda toplarken  merge_types lar normalize edilmeli veya kat kayıları değiştrilmelidir
            doc_groups[str(metadata)]["total_score"] += doc["scores"]
        merged_documents = []
        for group in doc_groups.values():
            group["doc"]["scores"] = group["total_score"]
            del group["doc"]["metadata"]["page_content"]
            group["doc"]["page_content"] = group["page_content"]
            merged_documents.append(group["doc"])
        sorted_merged_documents = sorted(merged_documents, key=lambda x: x['scores'],reverse=True)
        return sorted_merged_documents[:merged_n]
 
    def get_merge_similaritiy_texts(self,threshold_score=10, merge_type:str="sum", search_n=None,merged_n=None, query_and_releations=None,h_coefficient=None, coefficient_helpers=None) -> Tuple[List[str], List[float]]:
        
        search_n = search_n if search_n else len(self.all_data)
        merged_n = merged_n if merged_n else len(self.all_data)
        if not self.embedding_type in shared_utils.embeding_models:
            self.initialize()
        unique_documents  = self.get_similarity_text(db=self.main_vectordb,query=query_and_releations["query"],k=search_n,threshold_score=threshold_score,max_result_n=search_n)
        unique_texts = [unique_document["page_content"] for unique_document in unique_documents]
        unique_scores = [unique_document["scores"] for unique_document in unique_documents]
        if len(unique_scores)>0:
            m = max(unique_scores)
            unique_scores = [ m-s for s in unique_scores]
        query_scores_dict = dict(zip(unique_texts, unique_scores))
        
        scores_helper_and_main_document  = self.get_similarity_text(db=self.help_texts_vector_db,query=query_and_releations["query"],k=search_n,threshold_score=threshold_score,max_result_n=search_n)
        
        texts_helper_and_main_query = [unique_document["page_content"] for unique_document in scores_helper_and_main_document]
        scores_helper_and_main_query = [unique_document["scores"] for unique_document in scores_helper_and_main_document]
        if len(scores_helper_and_main_query)>0:
            m = max(scores_helper_and_main_query)
            scores_helper_and_main_query = [m-s for s in scores_helper_and_main_query]
            
        query_and_help_query_scores = dict(zip(texts_helper_and_main_query, scores_helper_and_main_query))
        coefficient_helpers_texts= []
        for c_helper in coefficient_helpers:
            coefficient_helpers_texts.extend(query_and_releations[c_helper])
        for help_query in texts_helper_and_main_query:
            #TODO: birsonraki versiyonda her helper_query için faklı h_coefficient tanımlanmalı gerekirse AI kullanılabilir sub querler oluşturulurken benzerlik kat sayısı eklenebilir ve negatif textler eklenebilir.
            coefficient  = 1
            if h_coefficient and coefficient_helpers and  help_query in coefficient_helpers_texts:
                coefficient  = h_coefficient
            
            unique_document_helper = self.get_similarity_text(db=self.main_vectordb,query=help_query,k=search_n,threshold_score=threshold_score,max_result_n=search_n)
            if len(unique_document_helper) > 0:
                unique_texts_helper = [unique_document["page_content"] for unique_document in unique_document_helper]
                unique_scores_helper = [unique_document["scores"] for unique_document in unique_document_helper]
                if len(unique_scores_helper)>0:
                    m = max(unique_scores_helper)
                    unique_scores_helper = [m-s for s in unique_scores_helper]
                # Yardımcı sorgu skorlarını birleştir
                unique_texts_helper_scores = dict(zip(unique_texts_helper, unique_scores_helper))
                for text, helper_text_score in unique_texts_helper_scores.items():
                    
                    if text in query_scores_dict:
                        # Eğer metin zaten text_scores_dict'te varsa, skorları birleştir
                        if merge_type == "sum":
                            query_scores_dict[text] += query_and_help_query_scores[help_query] + helper_text_score*coefficient
                        elif merge_type == "proud":
                            query_scores_dict[text] = (query_and_help_query_scores[help_query] * helper_text_score*coefficient ) + query_scores_dict[text]
                        elif merge_type == "square":
                            query_scores_dict[text] = math.sqrt(((helper_text_score*coefficient)**2 ) + query_scores_dict[text]**2 + query_and_help_query_scores[help_query]**2) 
                        elif merge_type == "square_sum":
                            query_scores_dict[text] = math.sqrt(((helper_text_score*coefficient)**2 ) + query_scores_dict[text]**2) + query_and_help_query_scores[help_query]
                        elif merge_type == "square_sum2":
                            query_scores_dict[text] = math.sqrt(((helper_text_score*coefficient)**2 ) + query_and_help_query_scores[help_query]**2) + query_scores_dict[text]
                        elif merge_type == "square_proud":
                            query_scores_dict[text] = math.sqrt(((helper_text_score*coefficient)**2 ) + query_scores_dict[text]**2) * query_and_help_query_scores[help_query]
                        elif merge_type == "square_proud2":
                            query_scores_dict[text] = math.sqrt(((helper_text_score*coefficient)**2 ) + query_and_help_query_scores[help_query]**2) * query_scores_dict[text]
                    else:
                        query_scores_dict[text] = (query_and_help_query_scores[help_query] + helper_text_score*coefficient )*coefficient

        return self._sort_merge_similarity_texts(query_scores_dict,unique_documents, merged_n)
    def _sort_merge_similarity_texts_new(
        self, text_scores_dict: Dict[str, float], unique_documents: List[dict], merged_n: int) -> List[dict]:
        """
        Sorts and merges similarity scores and unique documents, returning the top merged results.

        Args:
            text_scores_dict (dict): A dictionary of text-content and their similarity scores.
            unique_documents (list): A list of unique documents with metadata and content.
            merged_n (int): The number of top results to return after merging.

        Returns:
            list: A sorted list of merged documents with updated scores.
        """
        # Sort text scores in descending order
        sorted_text_scores = sorted(text_scores_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Extract the top N texts and their scores
        merged_texts, merged_scores = zip(*sorted_text_scores[:merged_n]) if sorted_text_scores else ([], [])
        
        # Filter unique_documents to include only those in the top N merged texts
        filtered_documents = [
            doc for doc in unique_documents if doc["page_content"] in merged_texts
        ]
        
        # Update scores in filtered documents and prepare merged_documents
        merged_documents = [
            {**doc, "scores": text_scores_dict[doc["page_content"]]}
            for doc in filtered_documents
        ]
        
        # Sort merged documents by scores in descending order
        merged_documents.sort(key=lambda x: x["scores"], reverse=True)

        # TODO: Apply Roberto AI-based filtering in future versions for rekanker.

        return merged_documents
    def _sort_merge_similarity_texts(self,text_scores_dict:dict,unique_documents:list,merged_n:int) -> Tuple[List[str], List[float]]:
        
        sorted_text_scores = sorted(text_scores_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_text_scores_dict = dict(sorted_text_scores)
        merged_texs = list(sorted_text_scores_dict.keys())
        merged_scores = list(sorted_text_scores_dict.values())
        merged_texs, merged_scores = merged_texs[:merged_n], merged_scores[:merged_n]
        unique_documents = [unique_document for unique_document in unique_documents if unique_document["page_content"] in merged_texs]
        merged_documents = []
        for unique_document in unique_documents:
            for text,score in sorted_text_scores_dict.items():
                if unique_document["page_content"]== text:
                    unique_document["scores"]=score
                    merged_documents.append(unique_document)
        merged_documents = sorted(merged_documents, key=lambda x: x["scores"], reverse=True)

        return merged_documents