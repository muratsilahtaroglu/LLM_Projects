
import requests,os
from fastapi import HTTPException
from typing import List, Tuple, Dict,Literal
try:
    import client.base_utils
   
except:
    import semantic_search.client.base_utils as base_utils
current_dir = os.path.dirname(os.path.abspath(__file__))
__all__ = ["SimilarityTextsClient"]
class SimilarityTextsClient:
    
    def __init__(self, host=None, port=8002) -> None:
        """
        The SimilarityTextsClient class creates a client to connect to a FastAPI server
        running on the specified host and port.

        Args:
            host (str, optional): The hostname or IP address of the FastAPI server.
                                  Defaults to "0.0.0.0".
            port (int, optional): The port number on which the FastAPI server is running.
                                  Defaults to port 8002.
        """
        # Define the headers for JSON content type.
        self.headers = {'Content-Type': 'application/json'}
        
        # If the host is not provided, use the default "0.0.0.0".
        if host is None:
            host = "0.0.0.0"
        
        # Assign the host and port for internal use.
        host = host
        port = port
        
        # Construct the main URL to access the FastAPI server's semantic search endpoint.
        self.main_url = f"http://{host}:{port}/semantic_search"
        
    def get_existing_app_tokens(self, collection_names:list, predictors_path: str):
    
        load_and_check_app_info_responses = self.load_and_check_app_info({"all_query_and_info_file":predictors_path})
        if not isinstance(load_and_check_app_info_responses, list):
            raise f"Error in {predictors_path}"
        app_tokens = []
        for i, collection_name in enumerate(collection_names):
            
            for check_app_info in load_and_check_app_info_responses:
                if  check_app_info.collection_name == collection_name:
                    app_tokens.append(check_app_info.app_token)
                    print(f"{i+1}. check_app_info is already exist collection_name:{check_app_info.collection_name} app_token:",check_app_info.app_token, flush=True)
                    return app_tokens
        return
                
    def create_app_and_get_app_token(self, collection_name:str, predictors_path: str, data:list, metadata:list, vectordb_directory: str=os.path.join(current_dir,"vectordb"), embedding_type:str="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"):
        
        assert len(data) == len(metadata), "Data and metadata must have the same length"
        create_app_args = {"all_query_and_info_file":predictors_path,
                "collection_name":collection_name,
                "embedding_type":embedding_type,
                "vectordb_directory":vectordb_directory,
                "data":data,
                "metadata":metadata
                }
    
        print(f"creating collection_name:{collection_name}", flush=True)
        create_app_response = self.create_app(create_app_args)
        print(f"create_app_response:\n{str(create_app_response)}",create_app_response.app_token, flush=True)
        app_token = create_app_response.app_token
        return app_token
    
 
    def create_app(self, tuned_args:dict) -> base_utils.CreateAppResponse:
        
        """
        Create a new semantic search application using the provided parameters.

        Args:
            args (base_utils.SemanticSearchAppParameters): The parameters required to configure
                                                        the semantic search application.
                                                        
            base_utils.UserAppDirectory is a model to define the user's application directory configuration.

        Returns:
            base_utils.CreateAppResponse: The response from the server containing information
                                        about the created application.
        """
        args= base_utils.SemanticSearchAppParameters()
        for key, value in tuned_args.items():
            setattr(args, key, value)
        url = f"{self.main_url}/create_app"
        response_args = self._get_response(url, args, base_utils.CreateAppResponse())
        
        return response_args
    
    def load_and_check_app_info(self, tuned_args:dict ) -> List[base_utils.CreateAppResponse]:
        
        """
        Load and check the application information based on the provided directory configuration.

        Args:
            args (base_utils.UserAppDirectory): The directory configuration required to locate
                                            and load the application information. This typically
                                            includes the path to the JSON file with query and info data.
            base_utils.UserAppDirectory is a model to define the user's application directory configuration.

        Returns:
            List[base_utils.CreateAppResponse]: A list of CreateAppResponse objects containing
                                            the details of each loaded application.
        """
        args= base_utils.UserAppDirectory()
        for key, value in tuned_args.items():
            setattr(args, key, value)
        # Construct the URL for the load and check application info endpoint.
        url = f"{self.main_url}/load_and_check_app_info"
        
        # Send a request to the server and receive the response.
        response_args = self._get_response(url, args, base_utils.CreateAppResponse())
        
        # Parse and return the server response as a list of CreateAppResponse objects.
        return response_args
    
    def set_queries_and_releations(self,  tuned_args:dict)->base_utils.ResponseQuery:
        """
        This method sends a request to set up queries and their relationships within the application.
        It interacts with a backend API endpoint to process the provided queries and returns a structured
        response.

        Args:
            args (base_utils.SetQuery): An instance of the SetQuery class that contains the queries
                                    and associated settings (like prompts and local prompt paths) to be processed.

        Returns:
            base_utils.ResponseQuery: A ResponseQuery object that contains the results of the operation,
                                    including the processed queries, their UUIDs, and metadata.
        """
        args=  base_utils.SetQuery()
        for key, value in tuned_args.items():
            setattr(args, key, value)
        # Construct the URL for the API endpoint that handles setting queries and their relationships.
        url = f"{self.main_url}/set_query_and_releations"
        
        # Send a request to the backend API with the `SetQuery` data provided in `args`.
        response_args = self._get_response(url, args, base_utils.ResponseQuery())
        
        # Parse the backend response into a `ResponseQuery` object and return it.
        return response_args
    
    def get_all_queries_and_keys(self, tuned_args:dict, sort_type:Literal["created_at","query"]="created_at"):
        """
        This method retrieves all queries and their associated keys from the backend, sorted by the specified sort type.
        It sends a request to an API endpoint that returns a list of queries and keys, which can be sorted by either
        the creation date or the query string.

        Args:
            args (base_utils.UserSearchParameters): An instance of the UserSearchParameters class containing
                                                user-specific search parameters, including the application token.
            sort_type (Literal["created_at", "query"], optional): The criteria by which the queries should be sorted.
                                                                Can be either "created_at" (default) to sort by
                                                                the creation date or "query" to sort alphabetically
                                                                by the query string.

        Returns:
            dict: The response from the backend containing all the queries and keys, sorted by the specified criteria.
        """
        args=  base_utils.UserSearchParameters()
        for key, value in tuned_args.items():
            setattr(args, key, value)
         # Construct the URL for the API endpoint that retrieves all queries and keys, sorted by the specified sort type.
        url = f"{self.main_url}/all_queries_and_keys/{sort_type}"
        
        # Send a request to the backend API with the `UserSearchParameters` provided in `args`.
        response = self._get_response(url, args)
        
        # Return the response from the backend, which contains the sorted list of queries and keys.
        return response
    
    def get_key_by_query(self,tuned_args:dict,  query:str):
        """
        This method retrieves the key associated with a specific query from a list of all queries and their keys.
        It first fetches all queries and keys using the `get_all_queries_and_keys` method and then iterates through
        the list to find the query that matches the provided query string. If a match is found, the corresponding key
        is returned.

        Args:
            args (base_utils.UserSearchParameters): An instance of the UserSearchParameters class containing
                                                user-specific search parameters, including the application token.
            query (str): The query string for which the associated key is being searched.

        Returns:
            str or list: The key associated with the specified query if found. If no matching query is found,
                        an empty list is returned.
        """
        # args=  base_utils.UserSearchParameters()
        # for key, value in tuned_args.items():
        #     setattr(args, key, value)
        # Retrieve all queries and their associated keys.
        all_queries_and_keys = self.get_all_queries_and_keys(tuned_args)
        
        # Iterate through the list of queries and keys.
        for q_k in all_queries_and_keys:
            # If the query matches the provided query string, return the associated key.
            for key,value in q_k["query_and_releations"].items():
                if value["query"] == query:
                    return key
        
        # If no matching query is found, return an empty list.
        #tuned_args["queries"] = [query]
        
        query_and_relations = self.set_queries_and_releations({**tuned_args,**{"queries":query}})
        key = list(query_and_relations.queries_and_uuids[0].keys())[0]
        return key
       
    def get_similarity_documents(self, method: str=None, query: str=None, tuned_args:dict= None, 
                                all_queries: list = None, external_help_queries: list = None):
        """
        This method retrieves documents similar to a given query by using the specified method. It supports
        different retrieval strategies, including automatic merging of results and direct querying via specific methods.

        Args:
            method (str): The method to use for retrieving similarity documents. Must be one of the predefined methods.
            key (str): The key associated with the query, used to identify the query in the database or search index.
            args (base_utils.GetQueryResults | base_utils.GetMergeQueryResults, optional): The parameters for the query or merge operation.
                                                                                        If not provided, defaults are used based on the method.
            all_queries (list, optional): A list of all previous queries used in the semantic search, potentially used in the merge_auto method.
            external_help_queries (list, optional): Additional helper queries that can be included to refine the search results, 
                                                    particularly used in the merge_auto method.

        Returns:
            list: A list of documents that match the query criteria. The number of returned documents is limited to 15.

        Raises:
            ValueError: If the specified method is not one of the predefined methods.
        
        Workflow:
        1. Check if the provided method is valid by comparing it against a list of predefined methods.
        2. If the method is "merge_auto", use the _get_auto_similarity_documents helper function.
        3. If args is not provided, initialize it based on whether the method involves merging or direct querying.
        4. Send a request to the backend using the specified method and query key.
        5. Retrieve and return the top 15 documents from the response, if any.
        """

        key = self.get_key_by_query(tuned_args, query)
        # Retrieve the list of valid methods for document retrieval
        methods = self._get_methods()
        
        # Check if the provided method is valid
        if method not in methods:
            raise ValueError(f"Your method is '{method}'. But your method must be one of the {methods}")
        
        # Handle the "merge_auto" method separately using the helper function
        elif method == "merge_auto":
            return self._get_auto_similarity_documents(key=key, tuned_args=tuned_args)
        
        # Initialize args based on the method if not provided
        args = base_utils.GetMergeQueryResults() if  "merge" in method else base_utils.GetQueryResults()
       
        for key, value in tuned_args.items():
            setattr(args, key, value)
        # Construct the URL for the API request based on the method and key
        url = f"{self.main_url}/method/{method}/{key}"
        
        # Send the request to the backend and retrieve the response
        response_json = self._get_response(url, args)
        
        # Extract the documents from the response and return the top 15
        documents = response_json["documents"]
        if len(documents) > 0:
            return [document for document in documents][:15]

    ### Delete Functions
    def delete_user_info_from_search_predictors(self, args: base_utils.UserAppDirectory):
        """
        Deletes user information from the search predictors. This method sends a request to the backend
        to remove all user-related data from the search predictor system.

        Args:
            args (base_utils.UserAppDirectory): An instance of UserAppDirectory that contains the necessary
                                            user-specific directory or application information for the deletion request.

        Returns:
            dict: The response from the backend, typically indicating the success or failure of the deletion operation.
        """
        url = f"{self.main_url}/delete_user_info_from_search_predictors"
        response_json = self._get_response(url, args)
        return response_json

    def delete_query_and_releations(self, key: str, args: base_utils.UserSearchParameters):
        """
        Deletes a specific query and its associated relations from the system. This method targets a particular
        query identified by its key and removes it along with any related data.

        Args:
            key (str): The key associated with the query that needs to be deleted.
            args (base_utils.UserSearchParameters): An instance of UserSearchParameters that contains the necessary
                                                parameters, such as the application token, for the deletion request.

        Returns:
            dict: The response from the backend, indicating whether the deletion was successful.
        """
        url = f"{self.main_url}/delete_query_and_releations/{key}"
        response_json = self._get_response(url, args)
        return response_json
    
    def delete_user_info_from_search_predictors(self, args: base_utils.UserAppDirectory):
        """
        Deletes user information from the search predictors. This method sends a request to the backend
        to remove all user-related data from the search predictor system.

        Args:
            args (base_utils.UserAppDirectory): An instance of UserAppDirectory that contains the necessary
                                            user-specific directory or application information for the deletion request.

        Returns:
            dict: The response from the backend, typically indicating the success or failure of the deletion operation.
        """
        # Construct the URL for the delete request.
        url = f"{self.main_url}/delete_user_info_from_search_predictors"
        
        # Send the delete request to the backend and capture the response.
        response_json = self._get_response(url, args)
        
        # Return the backend response, which typically includes status information.
        return response_json
    
    ### Helper Functions
    def _get_auto_similarity_documents(self,key, tuned_args):
        """
        This helper function retrieves documents that are similar to the given query by progressively
        broadening the search criteria. The function attempts to find relevant documents by using various
        query parameters and states, adjusting the search in several iterations if necessary.

        Args:
            query (str): The query string for which similar documents are being searched.

        Returns:
            list: A list of up to 15 unique documents that match the query criteria. If fewer than 5 unique
                documents are found, the search parameters are expanded in subsequent iterations.

        Workflow:
        1. Initialize search parameters using `GetMergeQueryResults` with specific states and thresholds.
        2. Attempt to retrieve documents from the API by querying with progressively broader criteria.
        3. Collect unique documents, ensuring that no duplicate texts are included.
        4. Return a list of up to 15 unique documents based on the search results.

        Note:
            The function makes multiple attempts (up to 8) to find relevant documents, adjusting the query parameters
            in each iteration. If enough documents are found before reaching the maximum attempts, the search stops early.
            This method is a heuristic approach to ensure relevant results are obtained even if the initial query parameters
            are too restrictive.
        """
        
        # Initialize search parameters
        args = base_utils.GetMergeQueryResults(
            app_token=tuned_args["app_token"],
            all_query_and_info_file=tuned_args["all_query_and_info_file"]
            
        )
        
        # TODO: This section will later be moved to the Insight Question API
        
        # Get the key associated with the query
        method = "get_merge_query_results"
        i = 0
        texts_docs = {}

        # Attempt to retrieve documents, broadening the search criteria in each iteration if necessary
        while i < 8:
            i += 1
            url = f"{self.main_url}/method/{method}/{key}"
            response = self._get_response(url, args)
            documents = response["documents"]

            if len(documents) > 0:
                for document in documents:
                    # Add unique documents to the texts_docs dictionary
                    if document["metadata"]["data.text"] not in list(texts_docs.keys()):
                        texts_docs[document["metadata"]["data.text"]] = document
            
            # If more than 5 unique documents are found, return the top 15 (or fewer if less than 15 found)
            if len(texts_docs) :
                return list(texts_docs.values())[:30]
            
            # Adjust search parameters in subsequent iterations to broaden the search
            if i == 1:
                args.main_keywords_state = True
                args.helper_keywords_state = True
                args.merge_type = "proud"
            elif i == 2:
                args.sub_questions_state = True
                args.merge_type = "square_sum2"
            elif i == 3:
                args.all_queries_state = True
                args.merge_type = "sum"
                args.threshold_score = 0.92
            elif i == 4:
                args.threshold_score = 0.95
                
            
            # TODO: Add additional steps here if necessary for further iterations
        
        # Return the list of unique documents found, limited to the top 15
        return list(texts_docs.values())[:30]

        
    def _get_methods(self):
        """
        This method returns a list of available methods that can be used for processing or retrieving
        query results. These methods likely correspond to different strategies or operations that can
        be performed on query data.

        Returns:
            list: A list of method names as strings, each representing a specific operation that can be
                performed within the context of the application.
        """
        
        # List of method names that correspond to various operations related to query processing
        methods = [
            "merge_auto",                # Method for automatically merging results
            "main_query_results",        # Method for retrieving results based on the main query
            "sub_queries_results",       # Method for retrieving results based on sub-queries
            "keywords_query_results",    # Method for retrieving results based on individual keywords
            "keywords_join_query_results", # Method for retrieving results based on joined keywords
            "merge_query_results"        # Method for merging results from multiple queries
        ]
        
        # Return the list of methods
        return methods
            
    def _is_response_ok(self, response: requests.Response, detail: str = "Sub process is not working."):
        """
        Checks the HTTP status code of the response and raises an HTTPException if the status indicates an error.
        This method is used to validate the response from an API call and handle errors appropriately.

        Args:
            response (requests.Response): The HTTP response object received from the API call.
            detail (str, optional): A custom error message to include in the exception if the response indicates a failure.
                                    Defaults to "Sub process is not working."

        Raises:
            HTTPException: Raises an HTTPException if the response status code is 400 or above.
                        The exception includes the status code and the response text, along with the custom detail message.
        """

        # Check if the status code is 500 or above, indicating a server-side error.
        if response.status_code >= 500:
            raise HTTPException(status_code=response.status_code, detail=f"{detail}\n{response.text}")
        
        # Check if the status code is 400 or above, indicating a client-side error.
        elif response.status_code >= 400:
            raise HTTPException(status_code=response.status_code + 100, detail=f"{detail}\n{response.text}")

    
    def _get_response(self, url, args, response_args=None):
        """
        Sends a POST request to the specified URL with the provided arguments, 
        checks the response status, and returns the response content as a JSON object.

        Args:
            url (str): The URL to which the POST request is sent.
            args: The arguments to be included in the POST request body. This is expected to be an object that
                can be transformed into a JSON-serializable dictionary.
            response_args: An optional argument specifying the response object type to map the response data.

        Returns:
            dict/list: The response content as an object of type response_args if specified, otherwise as a JSON object.

        Raises:
            HTTPException: If the response status code indicates an error, this exception is raised with
                        the appropriate status code and message.
        """

        # Transform the `args` object into a JSON-serializable dictionary using the helper method.
        args_json = base_utils.transform_base_model_args_json(args)
        
        # Send the POST request to the specified URL with headers and the JSON-encoded body.
        response = requests.post(url, headers=self.headers, json=args_json)
        
        # Check if the response status is OK; if not, raise an HTTPException.
        self._is_response_ok(response)
        response_json = response.json()
        # Return the response content as a JSON object.
       
        if not response_args:
            return response_json
        if isinstance(response_json,dict):
            for key, value in response_json.items():
                setattr(response_args, key, value)
            return response_args
        elif isinstance(response_json, list):
            response_args_list = []
            
            
            for r_j in response_json:
                new_response_args = response_args.__class__()
                for key, value in r_j.items():
                    setattr(new_response_args , key, value)
                response_args_list.append(new_response_args )
            return response_args_list
        

