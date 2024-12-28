import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import os,sys
#fast api
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir) #AdvencedRagLLM added to path
try:
    
    from advanced_semantic_search_method import *
    from schemas import *
    import shared_utils
    import file_operations_utils as file_operations_utils
    
except:
    from semantic_search_api.advanced_semantic_search_method import *
    from semantic_search_api.schemas import *
    from semantic_search_api import shared_utils
    import file_operations_utils as file_operations_utils
    
    
from fastapi import FastAPI, HTTPException
from typing import Dict
from uuid import UUID, uuid4
import torch, math, json
import logs
logger = logs.Logger()
shared_utils:Dict[str,Dict[UUID, SemanticSearchApp]]


def update_search_predictors_from_json(file_path: str):
    """
    Updates the search predictors from a JSON file. This method reads a JSON file that contains
    serialized data for search predictors, validates the file, and then updates the global 
    `search_predictors` dictionary with the new or updated search predictor objects.

    Args:
        file_path (str): The path to the JSON file that contains the search predictor data.

    Raises:
        HTTPException: If the provided file path does not have a `.json` extension.

    Workflow:
    1. Check if the file has a `.json` extension.
    2. Verify that the file exists and is not empty.
    3. Load the JSON data from the file.
    4. Iterate over the data items, converting them into `SemanticSearchApp` objects and updating
       the `search_predictors` dictionary.
    5. Print a confirmation message once the update is complete.
    """

    # Check the file extension to ensure it is a JSON file.
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() != '.json':
        raise HTTPException(status_code=404, detail=f"{file_path} is not a json file")

    # Check if the file exists and is a valid file.
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Warning: {file_path} does not exist as a file")
        
    
    # Check if the file is empty.
    elif os.path.getsize(file_path) == 0:
        raise HTTPException(status_code=404, detail=f"Warning: {file_path} is empty")  

    # Load the JSON data from the file.
    with open(file_path, 'r', encoding="utf-8") as f:
        data: dict = json.load(f)
        #SemanticSearchApp.from_dict(data)
    if  len(data):
        # Iterate over the items in the JSON data and update the search_predictors dictionary.
        #TODO: buraya eğer file path dosayası veya içeriğinin bir kısmı manuel olrak silinmiş ise güncelleme yapması gerekiyor!!! 
        for key, value in data.items():
            uuid_key = UUID(key)
            
            if file_path in shared_utils.search_predictors:
                if uuid_key in shared_utils.search_predictors[file_path]:
                    continue
                search_predictor = SemanticSearchApp(**value)
                #search_predictor.initialize()  # Uncomment if initialization is needed
                shared_utils.search_predictors[file_path][uuid_key] = search_predictor
            else:
                search_predictor = SemanticSearchApp(**value)
                #search_predictor.initialize()  # Uncomment if initialization is needed
                shared_utils.search_predictors[file_path] = {uuid_key: search_predictor}
    else:
        shared_utils.search_predictors[file_path] = {}

    logger.info("search_predictors is updated from JSON")



def save_search_predictors_as_json(file_path: str, search_predictors: Dict[str, Dict[UUID, SemanticSearchApp]], queries_and_uuids=None):
    """
    Saves the search predictors to a JSON file. This method converts the search predictor objects
    associated with a specific file path into a JSON-serializable format and writes them to a file.

    Args:
        file_path (str): The path to the JSON file where the search predictor data will be saved.
        search_predictors (Dict[str, Dict[UUID, SemanticSearchApp]]): A dictionary where each key is a file path 
                                                                      and the value is another dictionary. The inner 
                                                                      dictionary maps UUIDs to `SemanticSearchApp` objects.

    Workflow:
    1. Convert the `SemanticSearchApp` objects in the `search_predictors` dictionary to a JSON-serializable format.
    2. Write the converted data to a JSON file at the specified `file_path`.
    3. Print a confirmation message once the data has been successfully written to the file.
    """

    # Convert the search predictors associated with the file path to a JSON-serializable dictionary
    
    data = {str(key): model.to_dict() for key, model in search_predictors[file_path].items()}
    
    # Write the JSON-serializable data to the specified file path
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    # Print a confirmation message
    logger.info("Model predictors data is successfully written")
    if queries_and_uuids:
        all_query_and_relations_path = os.path.join(current_dir, "all_query_and_relations.json")
        try:
            with open(all_query_and_relations_path, 'r', encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception as e:
            logger.error(e)
            existing_data = {}
        for queries_and_uuid in queries_and_uuids:
            for key, new_data in queries_and_uuid.items():
                if new_data['query'] not in existing_data:
            
                    existing_data[new_data['query']] = new_data
                    with open(all_query_and_relations_path, 'w', encoding="utf-8") as f:
                        json.dump(existing_data, f, indent=4, ensure_ascii=False)
                        logger.info("All Query and Relatins data updated successfully.")

default_search_predictors_file_path = os.path.join(current_dir, "demo_predictors.json")
        
# Call the method to update search predictors from the specified JSON default file
update_search_predictors_from_json(default_search_predictors_file_path)

def check_session(all_query_and_info_file: str, app_token: str = None, key: str = None):
    """
    Checks if a session, identified by a file path, app token, and key, exists in the search predictors.
    This method ensures that the specified session information is valid and exists within the system's 
    search predictors dictionary.

    Args:
        all_query_and_info_file (str): The path to the file containing the query and session information. 
                                       This serves as the main identifier for locating the relevant search predictor.
        app_token (str, optional): A string representation of the UUID for the app token. 
                                   This is used to verify the existence of a specific session within the search predictor.
        key (str, optional): A string that identifies a specific query or relation within the session. 
                             This is checked against the session's query and relations.

    Raises:
        HTTPException: Raised if the file path, app token, or key does not exist in the search predictors dictionary.
                       A 404 status code is used for not found errors, while a 422 status code is used for general exceptions.

    Workflow:
    1. Verify that the specified file path exists in the `search_predictors` dictionary.
    2. If an `app_token` is provided, convert it to a UUID and verify its existence within the relevant search predictor.
    3. If a `key` is provided, check if it exists within the session's `query_and_releations`.
    4. Raise appropriate HTTP exceptions if any of the checks fail.
    """

    try:
        # Check if the specified file path exists in the search_predictors dictionary
        if not all_query_and_info_file in shared_utils.search_predictors:
            raise HTTPException(status_code=404, detail=f"{all_query_and_info_file} does not exist in search_predictors")
        
        # If an app_token is provided, convert it to UUID and check its existence
        if app_token:
            app_token = UUID(app_token)
            if not app_token in shared_utils.search_predictors[all_query_and_info_file]:
                raise HTTPException(status_code=404, detail=f"{app_token} does not exist in search_predictors or {all_query_and_info_file}")
        
        # If a key is provided, check if it exists in the specified session's query and relations
        if key:
            if not any(key in item for item in shared_utils.search_predictors[all_query_and_info_file][app_token].query_and_releations):
                raise HTTPException(status_code=404, detail=f"{key} does not exist in the session related to {app_token} or {all_query_and_info_file}")

    except Exception as e:
        # Catch any other exceptions and raise them as an HTTPException with status code 422
        raise HTTPException(status_code=422, detail=f"{e}")
    
def get_demo_data_and_queries():
    file_path = "data/demo_data.csv"
    try:
        import AdvencedRagLLM.semantic_search_api.schemas as schemas
    except:
        import  schemas

    import math,pandas as pd
    
    data = pd.read_csv(file_path,encoding="utf-8")
    data.dropna(subset=["content"], inplace=True)
    data.rename(columns= {"Unnamed: 0":"csv_index"},inplace=True)
  
    metadata_df = data[["data.text","csv_index"]]
    metadata = metadata_df.to_dict(orient="records")
    
    #Example queries
    queries =[ "Have you ever been in Turkiye?",
              "What is the relationship between Black Holes and Gravity?",
              "What is the biggest city in Turkey?",
              "How many people live in Turkey?"]
    
    queries = schemas.clean_float_values(queries)
    metadata = schemas.clean_float_values(metadata)
    data = data["content"].to_list()
    return data,metadata,queries

##################################  FastAPI  ################################## 
app = FastAPI()
@app.post("/dev")
async def dev(args:SemanticSearchAppParameters, dev_type:str="demo") -> ResponseQuery:
    dev_types = ["demo"]
    logger.info(dev_types)
    if dev_type not in dev_types:
        raise HTTPException(status_code=502, detail=f"dev_type is {dev_type}. dev_type is not found. You can try one of {dev_types}")
    
    if dev_type == "demo":
        data,metadata,queries = get_demo_data_and_queries()
   
    args.data = data
    args.metadata = metadata
    response = await create_app(args=args)
    
    query_args = SetQuery()
    query_args.all_query_and_info_file = args.all_query_and_info_file
    query_args.app_token = response.app_token
    query_args.queries=queries
    result = await set_query_and_releations(query_args)
    return result

@app.post("/semantic_search/load_and_check_app_info")
async def load_and_check_app_info(args: UserAppDirectory) -> List[CreateAppResponse]:
    """
    Loads the search predictors from the specified JSON file, checks the validity of the session,
    and returns information about the loaded semantic search applications.

    Args:
        args (UserAppDirectory): An instance of `UserAppDirectory` containing the path to the JSON file 
                                 with search predictor data.

    Returns:
        List[CreateAppResponse]: A list of `CreateAppResponse` objects, each containing information about 
                                 a loaded semantic search application (e.g., collection name, vector database directory, app token).
        str: If no apps are found in the specified JSON file, returns a message indicating this.

    Workflow:
    1. Update the `search_predictors` dictionary by loading data from the specified JSON file.
    2. Validate the session information using the `check_session` function.
    3. Iterate through the loaded semantic search applications, collecting their information into a list of `CreateAppResponse` objects.
    4. Return the list of `CreateAppResponse` objects. If no apps are found, return a message indicating this.
    """

    search_predictors_info = []
    # Update the search predictors with data from the specified JSON file
    update_search_predictors_from_json(args.all_query_and_info_file)
    
    # Check if the session exists and is valid
    check_session(args.all_query_and_info_file)
    
    # Collect information about each loaded semantic search application
    for app_token, semantic_search_app in shared_utils.search_predictors[args.all_query_and_info_file].items():
        
        s_p_info = CreateAppResponse(
            collection_name=semantic_search_app.collection_name,
            vectordb_directory=semantic_search_app. persist_directory,
            app_token=str(app_token)
        )
        search_predictors_info.append(s_p_info)
    
    # If no apps are found in the JSON file, return a message indicating this
    if not len(search_predictors_info):
        return f"{args.all_query_and_info_file} does not include any apps"
    
    # Return the list of CreateAppResponse objects
    return search_predictors_info

@app.post("/semantic_search/create_app")
async def create_app(args: SemanticSearchAppParameters) -> CreateAppResponse | Response:
    """
    Creates a new SemanticSearchApp based on the provided parameters, stores it in the global search_predictors
    dictionary, and saves the updated state to a JSON file.

    Args:
        args (SemanticSearchAppParameters): The parameters required to create a SemanticSearchApp, including
                                            embedding type, LLM name, device, vector database directory, collection name,
                                            and data/metadata for initializing the search application.

    Returns:
        CreateAppResponse | Response: On success, returns a `CreateAppResponse` with the details of the created app.
                                      On failure, returns a `Response` with an error message.

    Workflow:
    1. Evaluate `data` and `metadata` if they are provided as strings.
    2. Instantiate a `SemanticSearchApp` with the provided parameters.
    3. Generate a unique UUID for the new app and store it in the `search_predictors` dictionary.
    4. Create a vector database within the app, handling any errors that occur during this process.
    5. Save the updated `search_predictors` dictionary to a JSON file.
    6. Return a `CreateAppResponse` with the app details on success, or a `Response` with an error message on failure.
    """

    try:
        # Convert `data` and `metadata` from strings to their respective Python objects if necessary
        args.data = eval(args.data) if args.data and isinstance(args.data, str) else args.data
        args.metadata = eval(args.metadata) if args.metadata and isinstance(args.metadata, str) else args.metadata
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"{e}")

    # Instantiate the SemanticSearchApp with the provided parameters
    semantic_search_app = SemanticSearchApp(
        embedding_type=args.embedding_type,
        llm_name=args.llm_name,
        device=args.device,
        persist_directory=args.vectordb_directory,
        collection_name=args.collection_name
    )
    
    # Generate a unique UUID for the new application
    uui = uuid4()

    # Store the new app in the search_predictors dictionary
    if args.all_query_and_info_file in shared_utils.search_predictors:
        shared_utils.search_predictors[args.all_query_and_info_file][uui] = semantic_search_app
    else:
        shared_utils.search_predictors[args.all_query_and_info_file] = {uui: semantic_search_app}

    # Try to create the vector database for the app, handling any errors
    try:
        response = shared_utils.search_predictors[args.all_query_and_info_file][uui].create_vector_db(data=args.data, metadatas=args.metadata)
        message = f"SemanticSearchApp created and {response}"
    except Exception as e:
        # Remove the app from search_predictors if the creation fails
        del shared_utils.search_predictors[args.all_query_and_info_file]
        return Response(content=f"{e}", status_code=500)

    # Try to save the updated search_predictors dictionary to a JSON file
    try:
        save_search_predictors_as_json(args.all_query_and_info_file, shared_utils.search_predictors)
    except Exception as e:
        return Response(content=f"{e}", status_code=500)

    # Return a CreateAppResponse with the details of the created app
    return CreateAppResponse(
        message=message,
        collection_name=args.collection_name,
        app_token=str(uui),
        vectordb_directory=args.vectordb_directory
    )

        
@app.post("/semantic_search/set_query_and_releations")
async def set_query_and_releations(args: SetQuery) -> ResponseQuery:
    """
    Sets queries and their relationships within a specific SemanticSearchApp instance and updates the session data.
    The endpoint validates the session, processes each query, and then saves the updated state to a JSON file.

    Args:
        args (SetQuery): The parameters required to set queries and their relationships, including the app token, 
                         list of queries, local prompt path, and prompt.

    Returns:
        ResponseQuery: An object containing the list of queries with their UUIDs, the app token, 
                       the expected number of queries, and the actual number of processed queries.

    Workflow:
    1. Validate the session using `check_session` to ensure the `all_query_and_info_file` and `app_token` are valid.
    2. Iterate through the list of queries, setting up relationships for each within the specified SemanticSearchApp instance.
    3. Save the updated `search_predictors` dictionary to the specified JSON file.
    4. Return a `ResponseQuery` object with details about the processed queries.

    Raises:
        HTTPException: If any error occurs during the processing of a query, an HTTPException with a 404 status code is raised.
    """

    # Validate the session for the provided file and app token
    check_session(args.all_query_and_info_file, args.app_token, key=None)
    
    i = 0  # Counter for successfully processed queries
    queries_and_uuids = []  # List to store queries and their UUIDs

    # Iterate through each query provided in the args
    for query in args.queries:
        try:
            query_uuid = uuid4()  # Generate a unique UUID for the query
            query_and_releation = shared_utils.search_predictors[args.all_query_and_info_file][UUID(args.app_token)].set_query_and_releations(
                query=query,
                local_prompt_path=args.local_prompt_path,
                prompt=args.prompt,
                query_uuid=query_uuid
            )
            
            i += 1  # Increment the counter for processed queries
            queries_and_uuids.append(query_and_releation)  # Add the query and its UUID to the list
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"{e}")

    # Save the updated search predictors to the JSON file
    save_search_predictors_as_json(args.all_query_and_info_file, shared_utils.search_predictors, queries_and_uuids)
    
    # Return a ResponseQuery object with details about the processed queries
    return ResponseQuery(
        queries_and_uuids=queries_and_uuids,
        app_token=args.app_token,
        expected_queries_len=len(args.queries),
        queries_len=i
    )


@app.post("/semantic_search/all_queries_and_keys/{sort_type}")
def all_queries_and_keys(sort_type: Literal["created_at", "query"], args: UserSearchParameters) -> List[ResponseQueryInfoListItem]:
    """
    Retrieves all queries and their associated keys for a given SemanticSearchApp, sorted by either creation date or query string.
    
    Args:
        sort_type (Literal["created_at", "query"]): The criterion by which the results should be sorted. 
                                                    Can be "created_at" for sorting by the creation date or "query" 
                                                    for sorting alphabetically by the query string.
        args (UserSearchParameters): Parameters necessary for identifying the specific SemanticSearchApp, 
                                     including the path to the query and info file and the app token.

    Returns:
        List[ResponseQueryInfoListItem]: A list of `ResponseQueryInfoListItem` objects, each containing a query and its associated metadata, sorted by the specified criterion.

    Raises:
        HTTPException: If any error occurs, such as the session not being found or an issue during sorting, an HTTPException with a 404 status code is raised.
    """

    # Validate the session for the provided file and app token
    check_session(args.all_query_and_info_file, args.app_token, key=None)
    
    try:
        # Retrieve the query and relations for the specified app token
        query_and_releations = shared_utils.search_predictors[args.all_query_and_info_file][UUID(args.app_token)].query_and_releations
        
        # Define a mapping for the sorting criteria
        sort_map = {
            "created_at": lambda x: list(x.values())[0]["created_at"],  # Sort by the "created_at" field
            "query": lambda x: list(x.values())[0]["query"]  # Sort by the "query" field
        }
        
        # Sort the queries based on the provided sort_type and convert them into ResponseQueryInfoListItem objects
        query_list = [
            ResponseQueryInfoListItem(query_and_releations=query_and_releation)  
            for query_and_releation in sorted(query_and_releations, key=sort_map[sort_type])
        ]
        
        return query_list  # Return the sorted list
    except Exception as e:
        # Raise an HTTPException with a 404 status code if any error occurs
        raise HTTPException(status_code=404, detail=f"{e}")


def _get_query_and_releations(all_query_and_info_file: str, app_token: str, key: str):
    """
    Retrieves the `search_predictor` associated with the given `app_token` and the specific `query_and_releations` 
    identified by the provided `key`.

    Args:
        all_query_and_info_file (str): The path to the file containing the query and session information.
        app_token (str): The UUID string of the app token that identifies the specific SemanticSearchApp instance.
        key (str): The unique identifier for the query and its relations within the SemanticSearchApp.

    Returns:
        Tuple[SemanticSearchApp, dict]: Returns a tuple containing the `SemanticSearchApp` instance and the 
                                        `query_and_releations` dictionary corresponding to the given key.

    Raises:
        HTTPException: If the `query_and_releations` cannot be found using the provided key, an HTTPException 
                       with a 404 status code is raised.
    """
    # Validate the session for the provided file, app token, and key
    check_session(all_query_and_info_file, app_token, key=key)
    # Retrieve the search predictor associated with the app token
    search_predictor = shared_utils.search_predictors[all_query_and_info_file][UUID(app_token)]
    
    # Flatten the query_and_releations list into a dictionary for easy access by key
    query_and_releations = {k: v for d in search_predictor.query_and_releations for k, v in d.items()}
    
    # Retrieve the query and its relations using the provided key
    query_and_releations = query_and_releations.get(key)

    if not query_and_releations:
        raise HTTPException(status_code=404, detail=f"Key {key} not found in query and relations.")

    return search_predictor, query_and_releations

#method 1
@app.post("/semantic_search/method/get_main_query_results/{key}")
async def get_main_query_results(args: GetQueryResults, key: str) -> SemanticSearchResponse:
    """
    Retrieves documents that match the main query associated with the given key in a SemanticSearchApp.

    Args:
        args (GetQueryResults): Parameters for retrieving the main query results, including the number 
                                of documents to retrieve, threshold score, and maximum number of results.
        key (str): The unique identifier for the query and its relations within the SemanticSearchApp.

    Returns:
        SemanticSearchResponse: An object containing the retrieved documents, the main query, and a success flag.

    Raises:
        HTTPException: If any error occurs during the retrieval process, an HTTPException with a 404 status code is raised.
    """


    try:
        # Use the helper function to retrieve the search predictor and query_and_releations
        search_predictor, query_and_releations = _get_query_and_releations(args.all_query_and_info_file, args.app_token, key)
        
        # Retrieve documents similar to the main query
        documents = search_predictor.get_similarity_text(
            db=search_predictor.main_vectordb,
            query=query_and_releations["query"],
            k=args.k,
            threshold_score=args.threshold_score,
            max_result_n=args.max_result_n
        )
        
        # Return a successful response with the retrieved documents and query
        return SemanticSearchResponse(documents=documents, queries=[query_and_releations["query"]], is_success=True)
    
    except Exception as e:
        # Raise an HTTPException with a 404 status code if any error occurs
        raise HTTPException(status_code=404, detail=f"{e}")


#method 2
@app.post("/semantic_search/method/get_sub_queries_results/{key}")
async def get_sub_queries_results(args: GetQueryResults, key: str) -> SemanticSearchResponse:
    """
    Retrieves documents that match the sub-queries associated with the given key in a SemanticSearchApp.

    Args:
        args (GetQueryResults): Parameters for retrieving the sub-query results, including the number 
                                of documents to retrieve, threshold score, and maximum number of results.
        key (str): The unique identifier for the query and its relations within the SemanticSearchApp.

    Returns:
        SemanticSearchResponse: An object containing the retrieved documents, the sub-queries, and a success flag.

    Raises:
        HTTPException: If any error occurs during the retrieval process, an HTTPException with a 404 status code is raised.
    """
    
    try:
        # Use the helper function to retrieve the search predictor and query_and_releations
        search_predictor, query_and_releations = _get_query_and_releations(args.all_query_and_info_file, args.app_token, key)
        
        # Retrieve documents similar to the sub-queries
        documents = search_predictor.get_similarity_texts(
            db=search_predictor.main_vectordb,
            queries=query_and_releations["sub_queries"],
            k=args.k,
            threshold_score=args.threshold_score,
            max_result_n=args.max_result_n
        )
        
        # Return a successful response with the retrieved documents and sub-queries
        return SemanticSearchResponse(documents=documents, queries=query_and_releations["sub_queries"], is_success=True)
    
    except Exception as e:
        # Raise an HTTPException with a 404 status code if any error occurs
        raise HTTPException(status_code=404, detail=f"{e}")

#method 3
@app.post("/semantic_search/method/get_keywords_query_results/{key}")
async def get_keywords_query_results(args: GetQueryResults, key: str) -> SemanticSearchResponse:
    """
    Retrieves documents that match the keywords associated with the given key in a SemanticSearchApp.

    Args:
        args (GetQueryResults): Parameters for retrieving the keyword query results, including the number 
                                of documents to retrieve, threshold score, and maximum number of results.
        key (str): The unique identifier for the query and its relations within the SemanticSearchApp.

    Returns:
        SemanticSearchResponse: An object containing the retrieved documents, the keywords, and a success flag.

    Raises:
        HTTPException: If any error occurs during the retrieval process, an HTTPException with a 404 status code is raised.
    """

    try:
        # Use the helper function to retrieve the search predictor and query_and_releations
        search_predictor, query_and_releations = _get_query_and_releations(args.all_query_and_info_file, args.app_token, key)
        
        # Retrieve documents similar to the keywords
        documents = search_predictor.get_similarity_texts(
            db=search_predictor.main_vectordb,
            queries=query_and_releations["keywords"],
            k=args.k,
            threshold_score=args.threshold_score,
            max_result_n=args.max_result_n
        )
        
        # Return a successful response with the retrieved documents and keywords
        return SemanticSearchResponse(documents=documents, queries=query_and_releations["keywords"], is_success=True)
    
    except Exception as e:
        # Raise an HTTPException with a 404 status code if any error occurs
        raise HTTPException(status_code=404, detail=f"{e}")


#method 4
@app.post("/semantic_search/method/get_keywords_join_query_results/{key}")
async def get_keywords_join_query_results(args: GetQueryResults, key: str) -> SemanticSearchResponse:
    """
    Retrieves documents that match the joined keywords associated with the given key in a SemanticSearchApp.

    Args:
        args (GetQueryResults): Parameters for retrieving the joined keyword query results, including the number 
                                of documents to retrieve, threshold score, and maximum number of results.
        key (str): The unique identifier for the query and its relations within the SemanticSearchApp.

    Returns:
        SemanticSearchResponse: An object containing the retrieved documents, the joined keywords, and a success flag.

    Raises:
        HTTPException: If any error occurs during the retrieval process, an HTTPException with a 404 status code is raised.
    """

    try:
        # Use the helper function to retrieve the search predictor and query_and_releations
        search_predictor, query_and_releations = _get_query_and_releations(args.all_query_and_info_file, args.app_token, key)
        
        # Retrieve documents similar to the joined keywords
        documents = search_predictor.get_similarity_texts(
            db=search_predictor.main_vectordb,
            queries=query_and_releations["keywords_join_list"],
            k=args.k,
            threshold_score=args.threshold_score,
            max_result_n=args.max_result_n
        )
        
        # Return a successful response with the retrieved documents and joined keywords
        return SemanticSearchResponse(documents=documents, queries=query_and_releations["keywords_join_list"], is_success=True)
    
    except Exception as e:
        # Raise an HTTPException with a 404 status code if any error occurs
        raise HTTPException(status_code=404, detail=f"{e}")

#method 5
@app.post("/semantic_search/method/get_merge_query_results/{key}")
async def get_merge_query_results(args: GetMergeQueryResults, key: str) -> SemanticSearchMergeResponse:
    """
    Retrieves merged query results based on various query states and merges them according to the specified parameters.

    Args:
        args (GetMergeQueryResults): Parameters for merging the query results, including various flags and thresholds 
                                     to control the merging process.
        key (str): The unique identifier for the query and its relations within the SemanticSearchApp.

    Returns:
        SemanticSearchMergeResponse: An object containing the merged documents and the associated queries.

    Raises:
        HTTPException: If any error occurs during the merging process, an HTTPException with a 404 status code is raised.
    """

    try:
        # Use the helper function to retrieve the search predictor and query_and_releations
        search_predictor, query_and_releations = _get_query_and_releations(args.all_query_and_info_file, args.app_token, key)
        
        # Perform the merging of results based on the provided parameters
        documents = search_predictor.get_merge_results(
            threshold_score=args.threshold_score,
            max_result_n=args.max_result_n,
            all_queries=args.all_queries,
            external_help_queries=args.external_help_queries,
            all_queries_state=args.all_queries_state,
            sub_questions_state=args.sub_questions_state,
            helper_keywords_state=args.helper_keywords_state,
            keywords_join_list_state=args.keywords_join_list_state,
            merge_type=args.merge_type,
            k=args.k,
            main_keywords_state=args.main_keywords_state,
            query_and_releations=query_and_releations,
            coefficient=args.coefficient,
            coefficient_helpers=args.coefficient_helpers
        )

        # Return a successful response with the merged documents and associated queries
        return SemanticSearchMergeResponse(documents=documents, queries=query_and_releations, is_success=True)
    
    except Exception as e:
        # Raise an HTTPException with a 404 status code if any error occurs
        raise HTTPException(status_code=404, detail=f"{e}")

@app.delete("/semantic_search/close_user_info_from_search_predictors")
async def close_user_info_from_search_predictors(args: UserAppDirectory):
    """
    Deletes all user information associated with the specified query and info file from the search predictors.

    Args:
        args (UserAppDirectory): The directory information necessary to identify which user's data 
                                 should be deleted from the search predictors.

    Returns:
        Response: A response object confirming the deletion of the user info from search predictors.

    Raises:
        HTTPException: If an error occurs during the deletion process, an HTTPException with a 404 status code is raised.
    """

    try:
        # Delete the entry corresponding to the specified all_query_and_info_file from search_predictors
        del shared_utils.search_predictors[args.all_query_and_info_file]
        
        # Return a confirmation message upon successful deletion
        return Response(message=f"{args.all_query_and_info_file} deleted from search predictors", status_code=200)
    
    except Exception as e:
        # Raise an HTTPException with a 404 status code if an error occurs during deletion
        raise HTTPException(status_code=404, detail=f"{e}")

@app.delete("/semantic_search/delete_app_from_search_predictors")
async def delete_app_from_search_predictors(args: UserSearchParameters):
    """
    Deletes a specific query and its associated relations from a SemanticSearchApp instance.

    Args:
        key (str): The unique identifier for the query and its relations within the SemanticSearchApp.
        args (UserSearchParameters): Parameters necessary to identify the specific SemanticSearchApp instance, 
                                     including the path to the query and info file and the app token.

    Returns:
        Response: A response object confirming the deletion of the app that given app_token

    Raises:
        HTTPException: If an error occurs during the deletion process, an HTTPException with a 404 status code is raised.
    """

    # Validate the session for the provided file, app token, and key
    check_session(args.all_query_and_info_file, args.app_token)
    
    try:

        # Delete the query and its relations from the list
        del shared_utils.search_predictors[args.all_query_and_info_file][UUID(args.app_token)]
        
        # Save the updated search predictors to the JSON file
        save_search_predictors_as_json(args.all_query_and_info_file, shared_utils.search_predictors)
        
        # Return a confirmation message upon successful deletion
        return Response(message=f"{tempt_query} query and relations deleted", status_code=200)
    
    except Exception as e:
        # Raise an HTTPException with a 404 status code if an error occurs during deletion
        raise HTTPException(status_code=404, detail=f"{e}")

@app.delete("/semantic_search/delete_query_and_releations/{key}")
async def delete_query_and_releations(key: str, args: UserSearchParameters):
    """
    Deletes a specific query and its associated relations from a SemanticSearchApp instance.

    Args:
        key (str): The unique identifier for the query and its relations within the SemanticSearchApp.
        args (UserSearchParameters): Parameters necessary to identify the specific SemanticSearchApp instance, 
                                     including the path to the query and info file and the app token.

    Returns:
        Response: A response object confirming the deletion of the query and its relations.

    Raises:
        HTTPException: If an error occurs during the deletion process, an HTTPException with a 404 status code is raised.
    """

    # Validate the session for the provided file, app token, and key
    check_session(args.all_query_and_info_file, args.app_token, key=key)
    
    try:
        # Find the index of the query and relations that match the provided key
        index = next((i for i, d in enumerate(shared_utils.search_predictors[args.all_query_and_info_file][UUID(args.app_token)].query_and_releations) if key in d), None)
        
        if index is None:
            raise HTTPException(status_code=404, detail=f"Query with key {key} not found")
        
        # Store the query for confirmation message
        tempt_query = shared_utils.search_predictors[args.all_query_and_info_file][UUID(args.app_token)].query_and_releations[index][key]
        
        # Delete the query and its relations from the list
        del shared_utils.search_predictors[args.all_query_and_info_file][UUID(args.app_token)].query_and_releations[index]
        
        # Save the updated search predictors to the JSON file
        save_search_predictors_as_json(args.all_query_and_info_file, shared_utils.search_predictors)
        
        # Return a confirmation message upon successful deletion
        return Response(message=f"{tempt_query} query and relations deleted", status_code=200)
    
    except Exception as e:
        # Raise an HTTPException with a 404 status code if an error occurs during deletion
        raise HTTPException(status_code=404, detail=f"{e}")

@app.delete("/semantic_search/kill_embedding")
async def kill_embedding(embedding_type: str) -> Response:
    """
    Deletes the specified embedding model from the `embeding_models` dictionary and clears the GPU cache.

    Args:
        embedding_type (str): The type of embedding model to be deleted from the `embeding_models` dictionary.

    Returns:
        Response: A response object confirming the deletion of the embedding model.

    Raises:
        HTTPException: If the embedding model specified by `embedding_type` is not found, an HTTPException 
                       with a 404 status code is raised.
    """

    try:
        # Check if the specified embedding type exists in the embeding_models dictionary
        if embedding_type not in shared_utils.embeding_models:
            message = f"Cannot kill {embedding_type}. It was not found in the available embedding models."
            raise HTTPException(status_code=404, detail=message)

        # Delete the embedding model from the dictionary
        del shared_utils.embeding_models[embedding_type]
        # Clear the GPU cache to free up memory
        torch.cuda.empty_cache()

        # Return a confirmation message upon successful deletion
        return Response(message=f"{embedding_type} has been successfully deleted", status_code=200)
    
    except Exception as e:
        # Raise an HTTPException with a 404 status code if an error occurs during the deletion process
        raise HTTPException(status_code=404, detail=f"{e}")


