import ast
import glob
import pandas as pd
from pydantic import BaseModel
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# try:
#     import ollama_client 
# except:
#     import projects.ollama_client as ollama_client
import ai_utils
#import semantic_search_client.semantic_search_utils as s_utils

from client import semantic_search_client, base_utils
__SSC_PORT__ = 8003 #8002, 8003
ss_client = semantic_search_client.SimilarityTextsClient(port=__SSC_PORT__)
current_dir = os.path.dirname(os.path.abspath(__file__))

class TweetResponse(BaseModel):
    tweets: list
    

def get_data_and_metadata(file_path:str, collection_name: str):
    
    """
    This function takes a file path and collection name as arguments and returns 
    cleaned text data and its metadata from the file. It reads the file, 
    concatenates the sheets, and removes duplicates. It also removes rows with nan 
    values in the TEXT column. The function returns a tuple of two items. The first 
    item is a list of cleaned text data. The second item is the metadata of the text 
    data in the form of a list of dictionaries. Each dictionary in the list has the 
    same keys as the columns of the dataframe. The function also adds a column called 
    'Data_Type' to the metadata and sets it to 'Tweet' for all rows. The function 
    returns these two items as a tuple.
    """
    print("Collecting data ....",flush=True)
    #sheet_names = base_utils.get_sheet_names(file_path)
    sheet_name = None
    data = base_utils.get_data(file_path,sheet_name=sheet_name)
    combined_data:pd.DataFrame = pd.concat(data.values(), ignore_index=True)
    file_path = "datasets/data.xlsx"
    suc_data_path = os.path.join(current_dir, file_path)
    try:
        suc_data = base_utils.get_data(suc_data_path, 
                                    collection_name)
        combined_data:pd.DataFrame = pd.concat([combined_data, suc_data], ignore_index=True)
    except:
       print("suc data not found")

    combined_data['CREATED_AT'] = combined_data['CREATED_AT'].astype(str)
    #data.rename(columns= {"Unnamed: 0":"csv_index"},inplace=True)
    combined_data.dropna(subset=["TEXT"], inplace=True)
    combined_data = combined_data.drop_duplicates(subset='TEXT', keep='last')
    combined_data.rename(columns= {"TEXT":"data.text"},inplace=True)
    combined_data = combined_data.reset_index(drop=True)
    #TODO: Buraya detected diline göre eğer tr değilse ve image analizi olanlar ingilizce dbye aktarlıcaktır
    print(combined_data.columns,flush=True)

    combined_data['Data_Type'] = 'Tweet'
    

    content_data = list(combined_data["TEXT_IN_ENGLISH"])
    metadata = combined_data.to_dict(orient="records")
    print( "getting just text data")
    print("text data len",len(list(combined_data["data.text"])))
        
    
    content_data = base_utils.clean_float_values(content_data)
    
    
    metadata = base_utils.clean_float_values(metadata)

    return content_data, metadata

def get_app_token(collection_name: str, predictors_path:str, vectordb_directory: str, data_file_path: str, add_text=False):
    """
    Retrieves or creates an application token for a specified collection.

    This function checks for existing application tokens for a given collection name.
    If a token does not exist, it collects data and metadata, and creates a new 
    semantic search application to obtain an application token.

    Args:
        collection_name (str): The name of the collection to retrieve or create an app token for.
        predictors_path (str): Path to the predictors configuration file.
        vectordb_directory (str): Directory path for the vector database.
        data_file_path (str): File path to the data file for collecting data and metadata.
        add_text (bool, optional): Flag indicating whether to add text. Defaults to False.

    Returns:
        str: The application token for the collection. Returns an error message with details 
        if an exception occurs.
    """

    orjinal_collection_name = collection_name
    collection_name = collection_name.strip("_")
    try:
        if not add_text:
            app_tokens = ss_client.get_existing_app_tokens([collection_name],predictors_path)
            if not app_tokens:
                app_tokens = ss_client.get_existing_app_tokens([orjinal_collection_name],predictors_path)
        if app_tokens:
            return app_tokens[0]
        with ai_utils.CalculateTime(f"{data_file_path} collecting data time: "):
            data,metadata = get_data_and_metadata(data_file_path, orjinal_collection_name)
        embedding_type = "hugging_face"
        
        with ai_utils.CalculateTime(f"{collection_name} created app done time: "):
            
            app_token = ss_client.create_app_and_get_app_token(collection_name=collection_name,predictors_path=predictors_path,vectordb_directory=vectordb_directory,data=data,metadata=metadata,embedding_type=embedding_type)
        return app_token
    except Exception as e:
        return "ERROR:",e, "collection_name: ",collection_name 
