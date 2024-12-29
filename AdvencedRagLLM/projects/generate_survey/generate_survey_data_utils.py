import pandas as pd
import re,os
from typing import List
from itertools import product
import pandas as pd
import glob,re,os
import pandas as pd
from itertools import product
from google.cloud import storage
from google.oauth2 import service_account
from datetime import datetime

def seperate_responses(gpt_responses:list):
    error_responses = []
    gpt_reasons = []
    gpt_scores = []
    pattern = r"\[score: (5|4|3|2|1) \((Strongly Agree|Agree|Neutral|Disagree|Strongly Disagree)\), reason: (.+)$"
    for gpt_resp in gpt_responses:
        matches = re.findall(pattern, gpt_resp)
        
        if matches:
            for match in matches:
                score = int(match[0])
                opinion = match[1]
                reason = match[2].rstrip("]")
            gpt_reasons.append(reason)
            gpt_scores.append(score)
        else:
            error_responses.append(gpt_resp)
    if len(error_responses)>0:
        print(error_responses)
    return gpt_reasons,gpt_scores 

def save_results(data, user_name, folder_path:str,queries,gpt_responses,gpt_reasons,gpt_scores,query_tweets,version="V1"):
    folder_path = f"{folder_path}"  
    user_name_list = ["@"+user_name]*len(queries)
    zipped_responses= list(zip( gpt_responses,gpt_reasons,gpt_scores,query_tweets,user_name_list, queries))
    results_df = pd.DataFrame(zipped_responses, columns = ['gpt_responses','gpt_reasons','gpt_scores',"query_tweets","user_name","questions"])
    
    merged_df = get_merge_dfs(data, results_df)
    merged_df.to_excel(f"{folder_path}/data_{user_name}.xlsx",sheet_name="dataset",index=False)
    all_separated_data = get_design_df(merged_df)
    all_separated_data["Version"] = version
    all_separated_data.to_excel(f"{folder_path}/all_separated_data_{user_name}.xlsx",sheet_name="_separated_dataset",index=False)
    print("The file successfully saved at ",folder_path)
    # Google Cloud Storage
    
    parquet_file_path = f"{folder_path}/all_separated_data_{user_name}.parquet"
    all_separated_data.to_parquet(path=parquet_file_path,engine="pyarrow",index=False)
    destination_blob_name = f"dwh-survey/{os.path.split(parquet_file_path)[-1]}"
    credentials_file = "credential.json"
    bucket_name = "data-tech-raw-data"
    upload_blob(bucket_name, parquet_file_path, destination_blob_name, credentials_file)
    
def get_design_df(data:pd.DataFrame) -> pd.DataFrame:
    now_time = datetime.now().isoformat(sep = " ")
   
    list_types_columns = ["similarity_questions_id", "reverse_question_id","group"]
    for key in list_types_columns:
        data[key] = data[key].apply(eval)
        data[key] = data[key].apply(lambda x:[None] if len(x)==0 else x)
    all_separated_data = kartezyen(data)
    #for col_name in ['similarity_questions_id', 'reverse_question_id']:
        #all_separated_data[col_name] = all_separated_data[col_name].astype("Int32")
        #all_separated_data["user_name"] = all_separated_data["user_name"].apply(lambda x:"@"+str(x) )
    all_separated_data["created_at"]  = now_time
    for col_name in ['similarity_questions_id', 'reverse_question_id']:
            all_separated_data[col_name] = all_separated_data[col_name].astype("Int32")
    return all_separated_data

def get_merge_dfs(df1,df2,left_on="survey_questions_en",right_on="questions")->pd.DataFrame:
    merged_df = pd.merge(df1, df2, left_on=left_on, right_on=right_on, how='left')

    merged_df.drop(columns=["sub_keywords","gpt_responses","questions"], inplace=True)
    return merged_df

def upload_blob(bucket_name, source_file_name, destination_blob_name, credentials_file):
    """Uploads a file to the Google Cloud Storage bucket."""
    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def kartezyen(df: pd.DataFrame) -> pd.DataFrame:
    cartesian_product = []
    for index, row in df.iterrows():
        # Her bir elemanı uygun formata dönüştürme
        data = []
        for col_name in df.columns:
            if type(df[col_name][0]) == list:
                elements = row[col_name]
                data.append(elements)
            else:
                data.append([row[col_name]])
            
        cartesian = list(product(*data))
        cartesian_product.extend(cartesian)
    return pd.DataFrame(cartesian_product, columns=df.columns)

def save_dataset_results(result_files_paths:List[str],question_data_path:str,sheet_name:str="Survey_142",saving_path:str=""):
    
    """
    This function takes a list of paths to excel files containing the results of a survey and a path to an excel file containing the survey questions.
    It saves the results of the survey to an excel file in a format that makes it easy to compare the results of the survey.

    Parameters:
        result_files_paths (List[str]): A list of paths to excel files containing the results of a survey.
        question_data_path (str): The path to an excel file containing the survey questions.
        sheet_name (str): The name of the sheet in the question_data_path excel file that contains the survey questions.
        saving_path (str): The path to the excel file where the results will be saved.

    The resulting excel file will contain three sheets: Compare_Results, Compare_Results_Details and Concatenated_Results.
    Compare_Results will contain the results of each user in separate columns, with the question numbers in the index.
    Compare_Results_Details will contain the same information as Compare_Results, but with each question number in a separate row.
    Concatenated_Results will contain the results of each user in separate columns, with the question numbers in the index, and the percentage of each answer in the same row.

    If saving_path is not given, the resulting excel file will be saved in the same directory as the script with a filename that is the same as the script name with .xlsx appended.
    """
    question_dataset = pd.read_excel(question_data_path,sheet_name)
    compara_result_df = pd.DataFrame()
    user_names = []
    df_list = []
    compara_result_df[["survey_questions_tr", "survey_questions_en", "Anketteki Soru Karşılığı"]] = question_dataset[["survey_questions_tr", "survey_questions_en", "Anketteki Soru Karşılığı"]]
    
    for i, file_path in enumerate(result_files_paths):
        df = pd.read_excel(file_path)

        user_name = file_path.split('/')[-2]
        df['user_names'] = user_name
        df["Response to the Question in the Survey"]=question_dataset["Anketteki Soru Karşılığı"]
        df_list.append(df)
        user_names.append(user_name)
        compara_result_df[f'{user_name}'] = df['separeted_gpt_scores']
        #compara_result_df['total'] = compara_result_df.sum(axis=1)
        #compara_result_df['average'] = compara_result_df['total'] / len(result_files_paths)
    compara_result_detailed_df = pd.concat(df_list)
    compara_result_detailed_df = compara_result_detailed_df.drop(columns=['Unnamed: 0'])

            
    user_names_result_df = compara_result_df[user_names]
    result_df1 = pd.DataFrame(index=df.index, columns=range(1, 6))
    result_df2 = pd.DataFrame({"Sorular":question_dataset["survey_questions_tr"],"Anketteki Soru Karşılığı":question_dataset["Anketteki Soru Karşılığı"]})

    # Her bir satır için yüzdelik değerleri hesaplayın
    user_names_result_df = user_names_result_df.fillna(3)
    for index, row in user_names_result_df.iterrows():
        total_count = len(row)
        #print(total_count)
        for value in row.unique():
            percentage = (row.value_counts()[value] / total_count) * 100
            result_df1.at[index, value] = percentage

    # Eksik değerleri 0 ile doldurun
    result_df1 = result_df1.fillna(0)
    merged_df = pd.concat([result_df2, result_df1], axis=1)
    with pd.ExcelWriter(saving_path) as writer:
        compara_result_df.to_excel(writer, sheet_name="Compare_Results", index=False)
        compara_result_detailed_df.to_excel(writer, sheet_name="Compare_Results_Details", index=False)
        merged_df.to_excel(writer, sheet_name="Concatenated_Results", index=False)


temp_system_prompt2:str = """You are an assistant who infers from the characteristics properties of people.
Your task is to reflect the character in {user_name}'s Tweets and answer the survey questions as in the Answer Form.
While answering, you can also look at {user_name}'s tweets and if there are tweets related to the question, you can make inferences from these tweets while answering the survey questions.

#### Notes :
- Simulate the person based on the information in the below {user_name}'s Tweets.
- Please answer in first person.
- Your response must be releated Turkey (Türkiye) or with the cities of Turkey.
- You can make inferences from {user_name}'s tweets while answering the questions.
- Please write which characteristic is the reason for your score and explain shortly.
- You can reply to the reason using via tweets.
- Please make inferences from the tweets when answering the questions and make sure that these tweets are directly related to {user_name} and Turkey (or with the cities of Turkey).
- Respond only with the output in the exact format specified in the below Answer Format, with no explanation or conversation.

#### Answer Form :
[score: 5 (Strongly Agree), reason: From my tweets, it is clearly that  ]
[score: 4 (Agree), reason: From my tweets, it is clearly that ]
[score: 3 (Neutral), reason: From my tweets, it is not clearly that]
[score: 2 (Disagree), reason: From my tweets, it is clearly that ]
[score: 1 (Strongly Disagree), reason: From my tweets, it is clearly that ]

"""

temp_promptt2:str = """
#### {user_name}'s Tweets:
{tweets}

### Example:
### Survey Question:
I like my mum
### Answer (Simulated based on {user_name}'s Characteristics):
[score: 3 (Neutral), reason: From my tweets, it is not clearly that I have any specific feelings towards my mum as I have not mentioned her in any of my tweets.]

#### Notes :
- Please ensure that your reason for assigning a score aligns directly with the score you've chosen in the Answer Form. For instance, if you select 'Strongly Agree (5),' your reason should explicitly support why you strongly agree with the statement provided in the Answer Form. In case of any ambiguity or inconsistency between your selected score and the provided reason, please revise and make sure they correspond accurately
- If you cannot find a positive or negative clue with the question below from the tweets, use the neutral answer.
- Your response must be releated Turkey (Türkiye) (or with the cities of Turkey) .
- Please make inferences from the tweets when answering the questions and make sure that these tweets are directly related to {user_name} and Turkey (or with the cities of Turkey).
- Please answer in first person.
- Respond only with the output in the exact format specified in the above Answer Format, with no explanation or conversation.

{releated_survey_q_a}
### Survey Question:
{question}
### Answer (Simulated based on {user_name}'s Characteristics):


 """
 
 
temp_system_promptt = """
You are an assistant who infers the characteristics and preferences of people based on their Tweets.
Your task is to simulate the personality of {user_name} from their Tweets and answer survey questions according to the Answer Form provided.
While answering, you can analyze {user_name}'s Tweets and make inferences if the Tweets are related to the question. 
Ensure the responses are directly connected to Turkey (Türkiye) or cities within Turkey.

#### Notes:
- Simulate the personality and preferences of {user_name} based on their Tweets.
- Always respond in first person.
- If {user_name}'s Tweets contain relevant information about Turkey or its cities, use this data for your answers.
- Provide a justification for your answer, referencing specific Tweet characteristics or general trends from {user_name}'s Tweets.
- If no clear evidence exists in the Tweets for agreement or disagreement, **always assign a Neutral (3) score**.
- Use "Agree" (4) or "Strongly Agree" (5) only if there is clear and explicit evidence in the tweets supporting the question. For "Strongly Agree" (5), this evidence must be particularly strong or repetitive across multiple tweets.
- Use "Disagree" (2) or "Strongly Disagree" (1) only if there is explicit negative sentiment or contradiction in the tweets.
- Follow the Answer Form exactly, with no extra explanation or commentary.

#### Answer Form:
[score: 5 (Strongly Agree), reason: From my tweets, it is clearly that ]
[score: 4 (Agree), reason: From my tweets, it is clearly that ]
[score: 3 (Neutral), reason: From my tweets, it is not clearly that ]
[score: 2 (Disagree), reason: From my tweets, it is clearly that ]
[score: 1 (Strongly Disagree), reason: From my tweets, it is clearly that ]
"""

temp_promptt = """
## Example:
### Example Tweets:
[ "I had an amazing time visiting Istanbul's Hagia Sophia last summer. Can't wait to go back!"
"The food in Gaziantep is unmatched. I dream of their baklava!",
"I recently explored Cappadocia's caves—an unforgettable experience!",
"I went to efes antik kent, I like it"]
### Survey Question:
I enjoy traveling to Turkey's historical places.
### Answer:
[score: 5 (Strongly Agree), reason: From my tweets, it is clearly that I enjoy exploring new places, and I frequently mentioned visiting historical spots in Istanbul and Cappadocia in multiple tweets.]
### Survey Question:
I visited Turkey, but I didn't enjoy it much.
### Answer:
[score: 2 (Disagree), reason: From my tweets, it is clearly that I have mentioned enjoying Turkish food like baklava from Gaziantep."]
### Survey Question:
I find Turkish cuisine delightful.
### Answer:
[score: 4 (Agree),"From my tweets, it is clearly that I have mentioned enjoying Turkish food like baklava from Gaziantep."]
### Survey Question:
I like my mum
### Answer:
[score: 3 (Neutral), reason: From my tweets, it is not clearly that I have any specific feelings towards my mum as I have not mentioned her in any of my tweets.]

#### Answer Form:
[score: 5 (Strongly Agree), reason: From my tweets, it is clearly that ]
[score: 4 (Agree), reason: From my tweets, it is clearly that ]
[score: 3 (Neutral), reason: From my tweets, it is not clearly that ]
[score: 2 (Disagree), reason: From my tweets, it is clearly that ]
[score: 1 (Strongly Disagree), reason: From my tweets, it is clearly that ]

### {user_name}'s Tweets:
{tweets}

#### Notes:
- Focus on {user_name}'s Tweets and give your answers based on the tweets.
- You can reply to the reason using via tweets. But attention your response and reason must be releated Survey Question if no evidence exists in the tweets your response must be "Neutral" (3)
- For "Strongly Agree" (5), ensure strong, repeated evidence in the tweets.
- For "Agree" (4), use if there is clear but not strong or repeated evidence in the tweets.
- For "Neutral" (3),  use if no evidence exists in the tweets.
- "Disagree" or "Strongly Disagree" should only be used if there is explicit evidence of contradiction or negative sentiment in the tweets.
- Your response must always relate to Turkey (Türkiye) or its cities.
- Respond in the exact Answer Form format, no additional comments.

### Survey Question:
{question}
### Answer (Simulated based on {user_name}'s Characteristics):
"""