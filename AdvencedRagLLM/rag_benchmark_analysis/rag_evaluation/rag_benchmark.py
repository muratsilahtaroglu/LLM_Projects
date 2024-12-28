from ragatouille import RAGPretrainedModel
from typing import List, Optional
from datasets import load_dataset
import datasets
import copy
import sys
import os
import queue
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from semantic_search.semantic_search import SemanticSearch
except:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from semantic_search.semantic_search import SemanticSearch

import llm_pre_processing.file_operations_utils as fo_utils


class RAGBenchmark:
    def __init__(self):
        pass

    def convert_json_to_dataset(self, json_file: str) -> datasets.Dataset:
        """
        Converts a JSON file to a Hugging Face Dataset object.

        Args:
            json_file (str): The path to the JSON file to convert.

        Returns:
            datasets.Dataset: The converted Dataset object.

        Raises:
            FileNotFoundError: If the file specified by json_file does not exist.
        """

        if not os.path.isfile(json_file):
            raise FileNotFoundError(f"The file {json_file} does not exist.")

        data = load_dataset("json", data_files=json_file,  split = 'train')
        
        return datasets.Dataset.from_list(data)
    

    def get_relevant_docs_with_semantic_search(
        self, 
        query: str, 
        knowledge_index: SemanticSearch,
        collection_name: List[str] | str,
        test_settings: Optional[dict] = None, 
        rerank_method: str = "evaluation"
        ) -> List[dict]:

        """
        Retrieves relevant documents for a given query using semantic search.

        Args:
            query (str): The query string to search for.
            knowledge_index (SemanticSearch): The semantic search object to use for the search.
            collection_name (List[str] | str): A list of collection names or a single collection name to search in.
            test_settings (Optional[dict], optional): Additional settings to refine the similarity document arguments. Defaults to None.
            rerank_method (str, optional): The reranking method to use. Defaults to "evaluation".

        Returns:
            List[dict]: A list of dictionaries containing the title and content of the retrieved documents.
        """
        
        all_query_and_info_file = knowledge_index.all_query_and_info_file
        contents = {}

        # If collection_name is a string, process it directly without threads
        if isinstance(collection_name, str):
            knowledge_index.collection_name = collection_name
            knowledge_index.set_app_token_by_collection_name()
            similarity_doc_args = {
                "all_query_and_info_file": all_query_and_info_file,
                "app_token": knowledge_index.app_token,
            }
            if test_settings:
                similarity_doc_args.update(test_settings)

            result = knowledge_index.set_query_and_relations_response_for_evaluation(
                query=query, 
                similarity_documents_args=similarity_doc_args, 
                rerank_method=rerank_method
            )
            contents[collection_name] = result
        else:
            # Handle the case where collection_name is a list
            thread_list:List[threading.Thread] = []
            result_queue = queue.Queue()

            def fetch_content(collection_name: str, test_settings: Optional[dict], knowledge_index_copy: SemanticSearch):
                """
                Fetches and processes content for a given collection name using semantic search.

                Args:
                    collection_name (str): The name of the collection to fetch content from.
                    test_settings (dict, optional): Additional settings to refine the similarity document arguments.

                Side Effects:
                    Updates the knowledge index's collection name and app token.
                    Performs a semantic search to retrieve relevant documents.
                    Stores the retrieved results in the result queue.
                """

                knowledge_index_copy.collection_name = collection_name # Use the copy
                knowledge_index_copy.set_app_token_by_collection_name()
                similarity_doc_args = {
                    "all_query_and_info_file": all_query_and_info_file,
                    "app_token": knowledge_index_copy.app_token,
                }
                if test_settings:
                    similarity_doc_args.update(test_settings)

                # Perform search
                result = knowledge_index_copy.set_query_and_relations_response_for_evaluation(
                    query=query, 
                    similarity_documents_args=similarity_doc_args, 
                    rerank_method=rerank_method
                )
                result_queue.put({collection_name: result})

            for cn in collection_name:
                knowledge_index_copy = copy.deepcopy(knowledge_index) 
                thread = threading.Thread(target=fetch_content, args=(cn, test_settings, knowledge_index_copy))
                thread_list.append(thread)

            for thread in thread_list:
                thread.start()

            for thread in thread_list:
                thread.join()

            # Collect results from the queue
            while not result_queue.empty():
                content = result_queue.get()
                contents.update(content)

        content_list = []
        for _, v in contents.items():
            if len(v) > 0:
                for i in range(len(v)):
                    content_list.append(f"""{v[i]}""")         

        return content_list


    def answer_with_rag(self,
        question: str,
        knowledge_index: SemanticSearch,
        collection_name: List[str] | str,
        reranker: Optional[RAGPretrainedModel] = None,
        test_settings: Optional[dict] = None
    ):

        """
        Answer a question using the RAG pipeline.

        Args:
            question (str): The question to answer.
            knowledge_index (SemanticSearch): The knowledge index to search.
            collection_name (str or List[str]): The name of the collection to search in.
            reranker (RAGPretrainedModel, optional): The reranking model to use. Defaults to None.
            test_settings (dict, optional): Additional settings to refine the similarity document arguments.

        Returns:
            List[str]: The relevant documents.
        """

        if test_settings:
            del test_settings["index"]

        # Gather documents with retriever
        relevant_docs = self.get_relevant_docs_with_semantic_search(query=question, knowledge_index=knowledge_index,rerank_method="evaluation", collection_name= collection_name,test_settings=test_settings)

        # Optionally rerank results
        if reranker:
            relevant_docs = reranker.rerank(question, relevant_docs, k=10)
            relevant_docs = [doc["content"] for doc in relevant_docs]

        return relevant_docs


    def run_rag_tests(
        self,
        eval_dataset: datasets.Dataset,
        knowledge_index: SemanticSearch,
        collection_name: List[str] | str,
        output_file: str,
        reranker: Optional[RAGPretrainedModel] = None,
        processing_column: str = "topic",
        test_settings: Optional[dict] = None,
        verbose: Optional[bool] = True,
    ):

        """
        Runs RAG tests on a given evaluation dataset.

        Args:
            eval_dataset (datasets.Dataset): The evaluation dataset to run tests on.
            knowledge_index (SemanticSearch): The knowledge index to use for retrieval.
            collection_name (List[str] | str): The collection name to use for retrieval.
            output_file (str): The file path to save results to.
            reranker (Optional[RAGPretrainedModel], optional): An optional reranker model to use. Defaults to None.
            processing_column (str, optional): The column name in the evaluation dataset to process. Defaults to "topic".
            test_settings (Optional[dict], optional): Additional test settings to use. Defaults to None.
            verbose (Optional[bool], optional): Whether to print progress updates. Defaults to True.

        Returns:
            None
        """

        if not isinstance(collection_name, (list, str)):
            raise ValueError("collection_name must be a string or a list of strings.")


        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_file_folder = os.path.join(current_dir,"results")
        output_file = os.path.join(output_file_folder,output_file)

        # Load existing results
        existing_results = eval(fo_utils.read_textual_file(file_path=output_file)) if os.path.exists(output_file) else []

        new_results = []
        for example in (eval_dataset):

            processing_column_result = example[processing_column]

            # Skip if topic and test index combination already exists
            if test_settings:
                test_index = test_settings.get("index")  # Index for the current test setting
                
                if any(
                    output[processing_column] == processing_column_result and output["test_settings"].get("index") == test_index
                    for output in existing_results
                ):
                    continue

            else:
                if any(
                    output[processing_column] == processing_column_result for output in existing_results
                ):
                    continue

            # Get relevant documents
            relevant_docs = self.answer_with_rag(
                question=processing_column_result,
                knowledge_index=knowledge_index,
                collection_name=collection_name,
                test_settings=test_settings,
                reranker=reranker,
            )

            if verbose:
                print(f"Processed Column Value: {processing_column_result}", flush=True)

            # Add the index key back for result storage
            if test_settings: test_settings["index"] = test_index
            example["retrieved_docs"] = relevant_docs
            example["test_settings"] = test_settings
            new_results.append(example)

        # Merge results and append to the file
        combined_results = existing_results + new_results
        fo_utils.write_textual_file(data=combined_results,path=output_file)
        print(f"Results for test index {test_index} saved to {output_file}", flush=True)
