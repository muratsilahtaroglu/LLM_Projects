import os, pandas as pd, glob, re
import swifter

class Modified_Tweet_Dataset_to_Train_Agent:
  def __init__(self, df_paths=None,main_path = None) -> None:
    self.updated_data_paths = []
    if main_path:
       df_paths = glob.glob(f"{main_path}/*/original/*.csv")
    for i,path in enumerate(df_paths):
      try:
        df = pd.read_csv(path)    
        df.drop_duplicates(subset='Text', inplace=True)
        df.reset_index(drop=True,inplace=True)
        self.df = df
        self.df["Modified_Text"]= df['Text'].swifter.apply(self.remove_unwanted_words_and_modified_words)
        self.df["Modified_Text2"]= df['Text2'].swifter.apply(self.remove_unwanted_words_and_modified_words)
        self.instructions = self.get_instructions()
        instruction_data_separatelly = self.df.swifter.apply(self.data_modified_separatelly,axis=1)
        self.df["instruction_data_separatelly"] = instruction_data_separatelly
        self.df["fine_tune_prompt"] = self.df["instruction_data_separatelly"]
        instruction_data = self.df.swifter.apply(self.data_modified,axis=1)
        self.df["instruction_data"] = instruction_data
        #self.save_df(self.df,path,"modified_all")
        selected_df = self.design_df_by_original_tweet_count(self.df)
        print("path:",path)
        path = self.save_df(selected_df, path,"updated_train_data")
        self.updated_data_paths.append(path)
      except Exception as e:
         print("Error: ",e,"\nPath: ",path)

  def get_instructions(self):
    instructions = ["You are a social person. You are active on Twitter. ### Tweet:n\{tweet}",
                      "You're very friedly and you are active on social media. You like tweets that you are fond of. ### Your liked Tweet:\n{tweet}",
                      "You're quite sociable, and you maintain an active presence on social platforms. You reply to some your friend tweets. ### Your friend tweet:\n{tweet1}\n### Your reply:\n{tweet2}",
                      "You're extremely sociable, and you maintain a lively presence on social platforms. You answer  the a few quoted tweet. ### Quoted tweet:\n{tweet1}\n### Your answer:\n{tweet2}",
                      "You're highly sociable and have an active presence on social media. You retweet some tweets. ### Your Retweeted:\n{tweet}"]
    return  instructions
    
  def remove_unwanted_words_and_modified_words(self, text:str) -> str:
      #part1 remove_unwanted_words
      if isinstance(text, str): 
          text_removed_http = re.sub(r"@\w+|https\S+|\n","",text).strip()
          modified_text = re.sub( "\t", '', text_removed_http)
          modified_text = re.sub(r'http\S+', '', modified_text)
          modified_text = re.sub(r'^\.', '', modified_text)
          modified_text = re.sub(r'^RT :', '', modified_text)
          return modified_text.strip()

      else:
          return "" 
  
  def data_modified_separatelly(self, tweet_data):
      if tweet_data["Type"] == "Original":
          prompt = self.instructions[0].format(tweet=tweet_data["Modified_Text"])
      elif tweet_data["Type"] == "Liked":
          prompt = self.instructions[1].format(tweet=tweet_data["Modified_Text"])
      elif tweet_data["Type"] == "Reply":
          prompt = self.instructions[2].format(tweet1=tweet_data["Modified_Text2"],tweet2=tweet_data["Modified_Text"])
      elif tweet_data["Type"] == "Quoted":
          prompt = self.instructions[3].format(tweet1=tweet_data["Modified_Text2"],tweet2=tweet_data["Modified_Text"])
      elif tweet_data["Type"] == "Retweeted":
          prompt = self.instructions[4].format(tweet=tweet_data["Modified_Text"])
      else:
          prompt = ""
      return prompt
  
  def data_modified(self, tweet_data):
    if tweet_data["Type"] == "Reply" or tweet_data["Type"] == "Quoted":
        prompt = self.instructions[2].format(tweet1=tweet_data["Modified_Text2"],tweet2=tweet_data["Modified_Text"])
    else:
        prompt = self.instructions[0].format(tweet=tweet_data["Modified_Text"])
    return prompt
  
  def save_df(self, df:pd.DataFrame, path,sub_fix:str):
     print("save_df")
     path_split = os.path.split(path)
     path_folder =f"files/{sub_fix}"
     os.makedirs(path_folder, exist_ok=True)
     path = path_folder + f"/{sub_fix}_{path_split[-1]}" 
     df.to_csv(path, index=False)
     return path
  
  def design_df_by_original_tweet_count(self, df:pd.DataFrame):
    exist_types = list(set(df["Type"]))
    original_count = df["Type"].value_counts()["Original"] if "Original" in exist_types else 0
    liked_count = df["Type"].value_counts()["Liked"] if "Liked" in exist_types else 0
    reply_count = df["Type"].value_counts()["Reply"] if "Reply" in exist_types else 0
    retweeted_count = df["Type"].value_counts()["Retweeted"]  if "Retweeted" in exist_types else 0
    quoted_count = df["Type"].value_counts()["Quoted"] if "Quoted" in exist_types else 0

    count_list = []
    dfs = []
    if liked_count>original_count:
        liked_items = df[df["Type"] == "Liked"]
        random_liked_items = liked_items.sample(n=original_count, random_state=42)  
        count_list.append("Liked")
        dfs.append(random_liked_items)

    if quoted_count>original_count:
        liked_items = df[df["Type"] == "Quoted"]
        random_liked_items = liked_items.sample(n=original_count, random_state=42) 
        count_list.append("Quoted")
        dfs.append(random_liked_items)

    if reply_count>original_count:
        reply_items = df[df["Type"] == "Reply"]
        random_reply_items = reply_items.sample(n=original_count, random_state=42) 
        count_list.append("Reply")
        dfs.append(random_reply_items)

    if retweeted_count>original_count:
        retweeted_items = df[df["Type"] == "Retweeted"]
        random_retweeted_items = retweeted_items.sample(n=original_count, random_state=42) 
        count_list.append("Retweeted")
        dfs.append(random_retweeted_items)

    # Seçilen öğeleri bir veri çerçevesine dönüştürün
    if len(count_list) >0:
        filtered_df = df[~df["Type"].isin(count_list)]
        dfs.append(filtered_df)
        selected_df = pd.concat(dfs)
        selected_df = selected_df.sample(frac=1, random_state=42).reset_index(drop=True)
        return selected_df
    else:
        return df
    