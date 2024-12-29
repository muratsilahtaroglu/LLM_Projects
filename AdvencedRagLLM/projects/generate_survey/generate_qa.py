from queue  import Queue
from ai_utils import *


from generate_survey_data_utils import *
import time, copy, threading
import survey_semantic_search_creating as ssc
from client import semantic_search_client as ss_client, base_utils 

current_dir = os.path.dirname(os.path.abspath(__file__))
    
class SurveyQuestionAnswerGenerator:
    def __init__(self,queries_path:str,queries_sheet_name:str, user_name:str,repeat_count:int=1, source_column:str="Text", app_parameters:dict=None) -> None:
        self.queries_path = queries_path
        self.user_name = user_name
        self.repeat_count = repeat_count
        self.source_column = source_column 
        self.similarity_text_client = ss_client.SimilarityTextsClient(port=ssc.__SSC_PORT__)
        self.servey_response = AISurveyResponse()
        self.survey_data = base_utils.get_data(queries_path, queries_sheet_name)
        self.app_parameters = app_parameters
        
    def create_results(self ,model:str = "ollama",port="11437"):

        """
        Creates results for each query in survey_data using the specified model and port parameters.
        
        Parameters:
        model (str): The model to use for generating responses. Supported models are "ollama", "gemini", or any valid GPT model.
        port (str): The port to use for ollama model. Default is "11437".
        
        Returns:
        None
        
        """
        self.gpt_responses,self.gpt_reasons,self.gpt_scores, self.query_tweets = [],[],[],[]
        
        
        self.all_queries = self.survey_data["survey_questions_en"].to_list()
        self.set_query_and_relations(queries=self.all_queries)
        tweets=[]
        for  i,query in enumerate(self.all_queries):
            try:
                
                prompt,system_prompt,tweets = self.get_prompts_and_tweets(query)

                if model == "gemini":
                    temp_gpt_responses = self.servey_response.get_ai_response(system_prompt= system_prompt, user_prompt=prompt,max_tokens=70,repeat_count=self.repeat_count)
                    
                elif "gpt" in model:
                    temp_gpt_responses = self.servey_response.get_gpt_responses(model=model,system_promt= system_prompt, user_prompt=prompt,max_tokens=70,repeat_count=self.repeat_count)
                    
                elif "ollama" in model:
                    temp_gpt_responses = self.servey_response.get_ollama_response(system_prompt= system_prompt, user_prompt=prompt,max_tokens=70,repeat_count=self.repeat_count,port=port)
        
                else:
                    raise f"{model} model is not found"
     
            except Exception as e:
                print(f"Error App:{self.user_name}\nError Query:{query}\nError: {e}")
                temp_gpt_responses = ["[score: 3 (Neutral), reason:  From my tweets, it is not clearly that]"]
                
                time.sleep(2)
            self.set_results(temp_gpt_responses, tweets)
            print("user_name:",self.user_name,f"\n{i}.Query: ",query,"\nResponse: ",temp_gpt_responses)
            
           
            
    def set_query_and_relations(self, queries:list):

        """
        This function sets the given query and its relations in the semantic search client.
        
        Parameters:
        query (str): The query to be set.
        predictors_path (str): The path to the file that contains the predictors.
        app_token (str): The app token to be used for the semantic search client.
        
        Returns:
        result (dict): The result of the set queries and relations operation.
        """
        assert isinstance(queries, list)
        set_query_and_relations_args = copy.deepcopy(self.app_parameters)
        set_query_and_relations_args["queries"] = queries
        result = self.similarity_text_client.set_queries_and_releations(set_query_and_relations_args)
        #print(result,flush=True)
        
    def get_prompts_and_tweets(self,query:str):
            
        """
        Generates a prompt and system prompt based on the given query and retrieves related tweets.

        This function uses the semantic similarity client to obtain documents similar to the provided query.
        It then formats these documents as tweets and constructs a prompt and system prompt by replacing
        specific words in the query.

        Args:
            query (str): The query string used to find similar documents and construct prompts.

        Returns:
            tuple: A tuple containing:
                - prompt (str): The formatted prompt including the user's name, modified query, and related tweets.
                - system_prompt (str): The formatted system prompt including the user's name.
                - tweets (list): A list of tweet contents extracted from the similar documents.
        """

        similarity_documents_args = self.app_parameters  
        documents = self.similarity_text_client.get_similarity_documents(method="merge_auto",query=query,tuned_args=similarity_documents_args)
        tweets = []
        if documents:
            tweets = [document["page_content"] for document in documents] 
        query = query.replace("Is ","Could ")
        prompt = temp_promptt.format(user_name=self.user_name, question=query,tweets=tweets)
        system_prompt = temp_system_promptt.format(user_name=self.user_name)
        return prompt,system_prompt, tweets
    
    def set_results(self, temp_gpt_responses, tweets):
        """
        Sets the results for a given query by separating the GPT responses into reasons and scores.
        
        Parameters:
        temp_gpt_responses (list): A list of GPT responses for a given query.
        tweets (list): A list of related tweets for the given query.
        """
        temp_gpt_reasons,temp_gpt_scores = seperate_responses(temp_gpt_responses)
                
        len_tweets = len(tweets)
        if len_tweets == 0:
            len_tweets = 1
        self.gpt_responses.append(temp_gpt_responses)
        self.gpt_reasons.append(temp_gpt_reasons)
        self.gpt_scores.append(temp_gpt_scores)
        self.query_tweets.append(tweets)
        
    def save_results(self, main_folder_path:str,version: str="V1"):
        if len(self.gpt_responses)!= len(self.all_queries):
            print(f"{self.user_name} is len diffrent gpt_responses value and  queries")
        folder_path =  f"{main_folder_path}/{version}/{self.user_name}"
        os.makedirs(folder_path, exist_ok=True)
        
        print(self.user_name," response count is ", len(self.gpt_reasons))
        save_results(data=self.survey_data, user_name=self.user_name,folder_path=folder_path,queries=self.all_queries,gpt_responses=self.gpt_responses,
                     gpt_reasons=self.gpt_reasons,gpt_scores=self.gpt_scores,query_tweets=self.query_tweets,version=version)


def main(port: str, user_data_path: str,results_folder:str, data_path, predictors_path,version="V1",survey_flag=True):
    """
    Executes the main process for generating survey question answers for a given user.

    This function initializes the necessary parameters and generates survey responses
    using a specified model. It retrieves an application token, checks whether the
    survey for the user has been completed, and if not, processes the survey data
    and saves the results.

    Parameters:
        port (str): The port to use for the model.
        user_data_path (str): The file path to the user's data.
        results_folder (str): The directory where results will be saved.
        data_path (str): The path to the file containing the survey questions.
        predictors_path (str): The path to the predictors file.
        version (str, optional): The version of the survey. Defaults to "V1".
        survey_flag (bool, optional): Flag to determine if the survey should be processed. Defaults to True.

    Returns:
        None
    """

    try:
        print(port, "   ",user_data_path)
        #print(user_data_path)
        file_name = os.path.basename(user_data_path)
        user_name = os.path.splitext(file_name)[0]
        vectordb_directory = os.path.join(current_dir,"vectordb")
        app_token = ssc.get_app_token(collection_name=user_name,predictors_path=predictors_path,vectordb_directory=vectordb_directory,data_file_path=user_data_path)
        print(app_token,flush=True)
        time.sleep(1)
    
        completed_user_data_paths = glob.glob(f"{results_folder}/{version}/*")
        completed_users = {os.path.basename(folder) for folder in completed_user_data_paths}  
        if survey_flag and user_name not in completed_users:
            with CalculateTime(f"{user_name}'s survey questions completed"):
                app_parameters = {
                    "all_query_and_info_file":predictors_path,
                    "app_token":app_token
                }
                source_column="ENText"  
                survey_question_answer_genarator = SurveyQuestionAnswerGenerator(queries_path= data_path,queries_sheet_name="Survey",
                                                                            user_name=user_name ,repeat_count=1, source_column=source_column,app_parameters=app_parameters)
                print(f"Starting survey on :{user_data_path}")
                survey_question_answer_genarator.create_results(model="ollama",port=port)
                survey_question_answer_genarator.save_results(results_folder, version=version)
        
    except Exception as e:
        print(f"Error App:{user_data_path}\nError: {e}")
        

def worker(port, task_queue:Queue, results_folder, data_path, predictors_path, version, survey_flag):
    """
    Worker function that processes tasks in a queue.

    This function is intended to be run in a separate thread. It waits for tasks to
    be available in the queue, processes them, and marks them as done. If no tasks
    are available within the timeout, the function exits.

    Args:
        port (str): The port to use for the GPT model.
        task_queue (Queue): The queue containing the tasks to be processed.
        results_folder (str): The folder where the results will be saved.
        data_path (str): The path to the file containing the survey questions.
        predictors_path (str): The path to the file containing the predictors.
        version (str): The version of the survey.
        survey_flag (bool): Whether to use all ports or not.
    """
    while not task_queue.empty():
        try:
            user_data_path = task_queue.get(block=True, timeout=1)  # Wait until a task is available
            task_queue.task_done()  # Mark the task as done
        except Exception:
            break  # Exit if no more tasks are available within the timeout
        print(port, "   ",user_data_path)
        # Process the task
        main(port, user_data_path, results_folder, data_path, predictors_path, version, survey_flag)
                
def dynamic_thread_processing(user_data_paths, results_folder: str, data_path, predictors_path, version, survey_flag=True):
    """
    Processes the given list of user data paths in parallel using multiple threads.

    The function creates a queue and fills it with the given user data paths. It then
    starts multiple threads, each of which processes the tasks in the queue. The
    function waits for all threads to complete before returning.

    Args:
        user_data_paths (list): A list of paths to user data files.
        results_folder (str): The folder where the results will be saved.
        data_path (str): The path to the file containing the survey questions.
        predictors_path (str): The path to the file containing the predictors.
        version (str): The version of the survey.
        survey_flag (bool, optional): Whether to use all ports or not. Defaults to True.
    """
    ports = ["11437", "11438", "11439", "11440"] if survey_flag else ["11437"]
    task_queue = Queue()

    # Fill the queue with tasks
    for user_data_path in user_data_paths:
        task_queue.put(user_data_path)

    threads = []

    with CalculateTime("All users survey questions completed"):
        for port in ports:
            thread = threading.Thread(
                target=worker, 
                args=(port, task_queue, results_folder, data_path, predictors_path, version, survey_flag)
            )
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()



