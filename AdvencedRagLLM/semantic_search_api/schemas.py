from uuid import UUID, uuid4
from pydantic import BaseModel, Field
import datetime as dt
import json,math
from fastapi import HTTPException
from typing import Any, List, Tuple, Dict,Literal, Optional

class AnyError(HTTPException):
    def __init__(self, error_message):
        self.message = error_message
        super().__init__(self.message)
        
class UserAppDirectory(BaseModel):
    """
    UserAppDirectory is a model to define the user's application directory configuration.
    
    Attributes:
        all_query_and_info_file (str): The full path to the JSON file that contains
                                       all queries and related information.
                                       Defaults to 'all_query_info.json'.
    """
    all_query_and_info_file: str = "all_query_info.json" # your search predictor json full path

    
class UserSearchParameters(UserAppDirectory):
    """
    UserSearchParameters class extends the UserAppDirectory to include parameters 
    specifically related to a user search operation, such as the application token.

    Attributes:
        app_token (str): A string that represents the token associated with the user's application.
                         This token is used for authenticating and identifying the specific application 
                         during the search operation. Defaults to an empty string.
        all_query_and_info_file (str): The full path to the JSON file that contains
                                       all queries and related information.
                                       Defaults to 'all_query_info.json'.
    """
    
    app_token: str = ""  # User's application token for authentication

class SemanticSearchAppParameters(UserAppDirectory):
    """
    SemanticSearchAppParameters extends UserAppDirectory to include parameters
    for configuring a semantic search application.

    Attributes:
        all_query_and_info_file (str): The full path to the JSON file that contains
                                       all queries and related information.
                                       Defaults to 'all_query_info.json'.
        device (str): The device to be used for computation, e.g., 'cuda:2' for GPU.
                      Defaults to 'cuda:2'. If you don't have enough free gpu use cpu
        collection_name (str): The name of the vector database collection. Defaults to 'langchain'.
        embedding_type (str or Literal): The type of embedding model to use. Can be one of several
                                         predefined types or a custom model string.
                                         Defaults to 'emrecan/bert-base-turkish-cased-mean-nli-stsb-tr'.
        llm_name (str or Literal): The name of the large language model (LLM) to use.
                                   Defaults to 'gemini'.
        vectordb_directory (str): The directory where the vector database is stored.
                                  Defaults to 'vectordb'.
        data (list): A list of data elements to be processed. Defaults to ['data1'].
        metadata (list[dict]): A dictionary containing metadata associated with the list data.
                         Defaults to [{'source': 'metadata1','data.text':'text'}].
    """
    device: str = "cuda:2"
    collection_name: str = "langchain"
    embedding_type: Literal[
        "hugging_face", 
        "hugging_face_instruct", 
        "sentence-transformers/stsb-xlm-r-multilingual", 
        "open_ai"
    ] | str = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    llm_name: Literal["gemini", "open_ai","local_ai"] | str = "local_ai"
    vectordb_directory: str = "vectordb"
    data: list = ["data1"]
    metadata: List[dict] = [{'source': 'metadata1','data.text':'text'}]
    
class CreateHelperVectorDb(BaseModel):
    all_queries_data:list = Field(default_factory=list)
    help_texts_data:list = Field(default_factory=list) # Dışardan istenilirse eklenecek

class SetQuery(UserAppDirectory):
    """
    SetQuery class is used to define the structure for setting up queries within an application.
    It extends the UserAppDirectory class to include additional attributes specific to query settings.

    Attributes:

        all_query_and_info_file (str): The full path to the JSON file that contains
                                       all queries and related information.
                                       Defaults to 'all_query_info.json'.
        app_token (Optional[str]): A token associated with the user's application.
                                   This is used to authenticate and identify the specific app.
                                   Defaults to None.
        queries (list): A list of semantic queries that the user wants to set up.
                        Defaults to ["Do you like Türkiye?"].
        local_prompt_path (str): The file path to a local prompt file that can be used
                                 to generate sub-queries. It is optional and can be left empty.
                                 Defaults to an empty string.
        prompt (str): A direct prompt string to create sub-queries. It is optional and
                      can be left empty if not needed. Defaults to an empty string.
    """
    app_token: Optional[str] = None
    queries: list = ["Do you like Türkiye ?"]  # Semantic queries
    local_prompt_path: str = ""  # Path to local prompt file (optional)
    prompt: str = ""  # Direct prompt for creating sub-queries (optional)
    
class GetQueryResults(UserSearchParameters):
    """
    GetQueryResults class extends the UserSearchParameters to include additional parameters
    specific to retrieving query results from a search operation.

    Attributes:
        app_token (str): A string that represents the token associated with the user's application.
                         This token is used for authenticating and identifying the specific application 
                         during the search operation. Defaults to an empty string.
        all_query_and_info_file (str): The full path to the JSON file that contains
                                       all queries and related information.
                                       Defaults to 'all_query_info.json'.
        k (int): The number of top documents to retrieve for each query. 
                 This controls the breadth of the search results. Defaults to 10.
        threshold_score (float): The score threshold for filtering documents.
                                 Only documents with a score below this threshold will be considered relevant.
                                 Defaults to 0.5.
        max_result_n (int): The maximum number of unique documents to return from the search results.
                            This limits the final output to the top `max_result_n` results. Defaults to 10.
    """
    
    k: int = 10  # Number of top documents to retrieve
    threshold_score: float = 0.9  # Threshold score for filtering relevant documents
    max_result_n: int = 10  # Maximum number of unique documents to return
    
    
class GetMergeQueryResults(GetQueryResults):
    """
    GetMergeQueryResults class is used to configure and manage the parameters for merging query results
    in a semantic search operation. This class provides various options to fine-tune how the queries 
    and their results are handled, including the number of documents retrieved, thresholds, and merging strategies.

    Attributes:
        app_token (str): A string that represents the token associated with the user's application.
                         This token is used for authenticating and identifying the specific application 
                         during the search operation. Defaults to an empty string.
        all_query_and_info_file (str): The full path to the JSON file that contains
                                       all queries and related information.
                                       Defaults to 'all_query_info.json'.
        k (int): The number of top documents to retrieve for each query. 
                 This controls the breadth of the search results. Defaults to 10.
        threshold_score (float): The score threshold for filtering documents.
                                 Only documents with a score below this threshold will be considered relevant.
                                 Defaults to 0.5.
        max_result_n (int): The maximum number of unique documents to return from the search results.
                            This limits the final output to the top `max_result_n` results. Defaults to 10.                       
        all_queries (list): A list of all previously used queries in semantic search. This is optional.
                            Defaults to an empty list containing one empty string [""].
        external_help_queries (list): Additional helper queries to be included in the semantic search.
                                      These are extra queries that may assist in refining the search results.
                                      Defaults to an empty list containing one empty string [""].
        all_queries_state (bool): A boolean indicating whether to include all previously used queries
                                  in the semantic search. Defaults to False.
        sub_questions_state (bool): A boolean indicating whether to include AI-generated sub-queries.
                                    Defaults to False.
        main_keywords_state (bool): A boolean indicating whether to include AI-generated main keyword queries.
                                    Defaults to True.
        helper_keywords_state (bool): A boolean indicating whether to include AI-generated helper keyword queries.
                                      Defaults to False.
        keywords_join_list_state (bool): A boolean indicating whether to include AI-generated combined keyword queries.
                                         Defaults to True.
        coefficient (float): A coefficient that modifies the impact of the included helper queries.
                             This allows fine-tuning the importance of these queries in the final result.
                             Defaults to 1.2.
        coefficient_helpers (List[Literal["sub_queries", "keywords", "helper_keywords", "keywords_join_list"]]):
            A list of query types to include as helpers in the semantic search, each associated with the coefficient.
            Defaults to ["keywords_join_list"] and ensures uniqueness.
        merge_type (Literal["auto1", "sum", "proud", "square", "square_sum", "square_sum2", "square_proud", "square_proud2"]):
            The method used to merge the results of the semantic search queries. Different strategies can be applied
            to optimize the search outcome. Defaults to "auto1". auto1 combines the advantages of all strategies.
    """

    all_queries: list = [""]  # All previously used queries in semantic search (optional)
    external_help_queries: list = [""]  # Extra helper queries for the search
    all_queries_state: bool = False  # Include all previous queries in the search
    sub_questions_state: bool = True  # Include AI-generated sub-queries
    main_keywords_state: bool = True  # Include AI-generated main keyword queries
    helper_keywords_state: bool = True  # Include AI-generated helper keyword queries
    keywords_join_list_state: bool = True  # Include AI-generated combined keyword queries
    coefficient: float = 1.2  # Coefficient for modifying the impact of helper queries
    coefficient_helpers: List[Literal["sub_queries", "keywords", "helper_keywords", "keywords_join_list"]] = Field(
        default_factory=lambda: ["keywords_join_list","helper_keywords"], unique_items=True
    )  # Types of helper queries to include, defaulting to "keywords_join_list"
    merge_type: Literal[
        "auto1", "sum", "proud", "square", "square_sum", "square_sum2", "square_proud", "square_proud2"
    ] = "auto1"  # Method for merging the search results

class SemanticSearchResponse(BaseModel):
    """
    SemanticSearchResponse class is used to model the response from a semantic search operation.
    It contains the queries that were processed and the documents that were returned as results.

    Attributes:
        queries (list): A list of queries that were used in the semantic search operation.
                        Each item in the list represents a single query. This field uses a 
                        default factory to initialize an empty list if no queries are provided.
        documents (list): A list of documents that represent the search results corresponding 
                          to the queries. Each document could be the result of a specific query.
                          This field uses a default factory to initialize an empty list if no 
                          documents are provided.
    """
    queries: list = Field(default_factory=list)
    documents: list = Field(default_factory=list)

    
class SemanticSearchMergeResponse(BaseModel):
    """
    SemanticSearchMergeResponse class is used to model the response from a semantic search operation
    that involves merging multiple queries and their associated documents.

    Attributes:
        queries (dict): A dictionary where the keys represent the queries or query identifiers,
                        and the values are the actual queries. This field uses a default factory
                        to initialize an empty dictionary if no data is provided.
        documents (list): A list that contains the documents or results associated with the queries.
                          Each document could be the result of the semantic search operation.
                          This field uses a default factory to initialize an empty list if no data is provided.
    """
    queries: dict = Field(default_factory=dict)
    documents: list = Field(default_factory=list)
 
class Response(BaseModel):
    """
    Response class is a basic model for representing the outcome of an operation or request.
    It provides a standard structure for responses, indicating success or failure and 
    optionally including a message.

    Attributes:
        message (str): A message providing additional information about the response.
                       This could be an error message, a success confirmation, or any
                       other relevant information. Defaults to an empty string.
        is_success (bool): A boolean flag indicating whether the operation was successful.
                           Defaults to True, assuming success unless otherwise specified.
    """
    message: str = ""
    is_success: bool = True

class CreateAppResponse(Response):
    """
    CreateAppResponse is a model that defines the structure of the response
    returned when creating a new application.

    Attributes:
        message (str): A message indicating the success or failure of the operation.
                       Defaults to an empty string.
        collection_name (str): The name of the vector database project.
        app_token (str): A token associated with the user's application.
        vectordb_directory (str): The directory where the user's vector database is stored.
    """
    collection_name: str ="langchain" # Vector database project name
    app_token: str ="" # User app token
    vectordb_directory: str="vector_db"  # User vector database directory 
    
class ResponseQuery(BaseModel):
    """
    ResponseQuery class is used to model the response structure when queries are processed
    within an application. This class holds details about the queries and their associated metadata.

    Attributes:
        queries_and_uuids (list): A list containing the processed queries along with their unique identifiers (UUIDs).
                                  This field uses a default factory to initialize an empty list if not provided.
        app_token (Optional[str]): An optional token that identifies the application. This is used to associate
                                   the response with a specific app.
        expected_queries_len (int): The number of queries that were expected to be processed.
                                    Defaults to 0.
        queries_len (int): The actual number of queries that were processed. This provides a way to verify
                           that the expected number of queries matches the actual number processed.
                           Defaults to 0.
        created_at (datetime): A timestamp indicating when the response was created.
                               It uses a default factory to set the current date and time at the moment of instantiation.
    """
    queries_and_uuids: list = Field(default_factory=list)
    app_token: Optional[str] = None
    expected_queries_len: int = 0
    queries_len: int = 0
    created_at: dt.datetime = Field(default_factory=dt.datetime.now)
    
class QuerySession(BaseModel):
    """
    QuerySession class is used to model a session containing a set of queries and 
    their associated creation timestamp.

    Attributes:
        queries (dict): A dictionary that stores the queries. The keys can represent
                        unique identifiers or categories, and the values are the actual queries.
                        This field uses a default factory to initialize an empty dictionary if not provided.
        created_at (datetime): A timestamp indicating when the query session was created.
                               It uses a default factory to set the current date and time at the moment of instantiation.
    """
    queries: dict = Field(default_factory=dict)
    created_at: dt.datetime = Field(default_factory=dt.datetime.now)

    
class ResponseQueryInfoListItem(BaseModel):
    """
    ResponseQueryInfoListItem class is used to represent an item in a list of query information.
    It stores a dictionary that contains the query and its related information or relationships.

    Attributes:
        query_and_releations (dict): A dictionary that holds the query as a key and its associated 
                                     relationships or related information as the value. This could 
                                     include data such as related queries, documents, scores, or any 
                                     other relevant metadata.
    """
    
    query_and_releations: dict  # Dictionary holding the query and its associated relationships
    


def clean_float_values(data):
    if isinstance(data, list):
        return [clean_float_values(item) for item in data]
    elif isinstance(data, dict):
        return {key: clean_float_values(value) for key, value in data.items()}
    elif isinstance(data, float):
        if math.isinf(data):
            return "Infinity" if data > 0 else "-Infinity"
        elif math.isnan(data):
            return "NaN"
        else:
            return data
    else:
        return data

def transform_base_model_args_json( args):
    """
    Transforms a given `args` object into a JSON-serializable dictionary.
    This method attempts to use `model_dump()` first, which is likely specific to certain model objects.
    If that method is not available, it falls back to using `dict()` to transform the object.

    Args:
        args: The object to be transformed into a JSON-serializable dictionary. This object could be
            an instance of a Pydantic model or any other class that has a `dict()` or `model_dump()` method.

    Returns:
        dict: A dictionary representation of the `args` object, suitable for JSON serialization.
    """
    try:
        args_json = args.model_dump()
    except:
        args_json = args.dict()
    return args_json

