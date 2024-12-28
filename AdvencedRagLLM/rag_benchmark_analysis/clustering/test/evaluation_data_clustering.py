
import re
import sys,os
from pathlib import Path
import pandas as pd
# Add the top-level directory to the system path
base_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(base_dir))
import  file_operations_utils as fo_utils
import rag_benchmark_analysis.clustering.dataset_clustering as dataset_clustering

def get_df(data_path:str):
    data = fo_utils.read_textual_file(data_path)
    if data_path.endswith(".json"):
            df = pd.DataFrame(eval(data))
    elif data_path.endswith(".csv") or data_path.endswith(".parquet"):
        df = data
    else:
        print("Unsupported file extension.") 
    return df
def remove_links(text):
        text = re.sub(r'http\S+|https\S+', '', text).strip()
         # Remove mentions
        text = re.sub(r'@\w+', '', text).strip()

        # Remove emojis
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # Emoticons
            u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
            u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
            u"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
            u"\U00002500-\U00002BEF"  # Chinese Characters
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642"
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # Dingbats
            u"\u3030"
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub(r'', text).strip()
        
        return re.sub(r'http\S+|https\S+', '', text).strip()

def generate_data_clustering_projects(task_name:str):
    version = "V"
    if task_name =="rag":
        data_path="/home/clone/clone/preparing_dataset/data/clone_ai_trainsetV6_updated.json"
        df = get_df(data_path= data_path)
        data = df["topic"].tolist()
 
    elif task_name=="tweet":
        data_path="/home/clone/clone/preparing_dataset/ml_tweets.parquet"
        df = get_df(data_path= data_path)
        df['TEXT'] = df['TEXT'].apply(remove_links)
        df.drop_duplicates(subset="TEXT",inplace=True)
        data = df["TEXT"].tolist()

    elif task_name == "pdf":
        data_path="/home/clone/clone/docs/edited_data/parsed_pdf_dataV4.json"
        df = get_df(data_path= data_path)
        df.drop_duplicates(subset="title",inplace=True)
        data = df["title"].tolist()

    elif task_name == "twitter":
        data_path="/home/clone/clone/docs/edited_data/parsed_tweet_data.json"
        df = get_df(data_path= data_path)
        df.drop_duplicates(subset="title",inplace=True)
        data = df["title"].tolist()
    
    elif task_name == "youtube":
        data_path="/home/clone/clone/docs/edited_data/parsed_youtube_dataV7.json"
        df = get_df(data_path= data_path)
        df.drop_duplicates(subset="title",inplace=True)
        data = df["title"].tolist()
    
    response_list = dataset_clustering.generate_data_clustring(task_name,version,data=data,start_best_metric_order=0,stop_best_metric_order=70,range_order=1)
        
    return response_list

if __name__ == "__main__":
    
    response_list = generate_data_clustering_projects(task_name="youtube") #rag, tweet, pdf, twitter, youtube
    for response in response_list:
        print(response)