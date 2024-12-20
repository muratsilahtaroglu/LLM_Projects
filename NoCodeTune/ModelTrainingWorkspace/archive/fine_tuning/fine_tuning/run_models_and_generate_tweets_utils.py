
from transformers import AutoTokenizer,AutoModelForCausalLM, AutoModelForSeq2SeqLM, GenerationConfig, TrainingArguments
import torch
import time
import re, os
import pandas as pd
from datasets import load_dataset,DatasetDict
from pydantic import BaseModel
import random
from peft import LoraConfig, get_peft_model, TaskType
import subprocess


class FreeGpuDetector:
    def __init__(self) -> None:
        pass
    
    def get_first_free_cuda_index(self,wanted_memory_as_gb:int=5):
        gpu_status = self.get_gpu_status()
        gpus = self.get_free_gpus(gpu_status)
        for gpu in gpus:
            free_memory = gpu["free_memory"]
            free_memory = free_memory / 1024 #GB
            if free_memory >= wanted_memory_as_gb:
                return gpu["index"]
        return 4
    def get_gpu_status(self):
        try:
            result = subprocess.run(['gpustat', '--watch'], capture_output=True, text=True, check=True, shell=True)
            gpu_status = result.stdout
            return gpu_status
        except subprocess.CalledProcessError as e:
            print(f'Hata: {e}')

    def get_free_gpus(self,gpu_status):
        free_gpus = []

        if gpu_status:
            # gpustat çıktısını satır satır bölelim
            lines = gpu_status.split('\n')

            # Her bir satırı kontrol edelim
            for line in lines:
                # Satırda "0 % |" ifadesini arayalım (boş GPU'ları temsil eder)
                match = re.search(r'\b0 % \|', line)
                
                # Eğer eşleşme varsa, o GPU boştur
                if match:
                    # Satırdan GPU indeksini çıkartalım
                    gpu_index_match = re.search(r'\[([\d]+)\]', line)
                    if gpu_index_match:
                        if gpu_index_match:
                            gpu_index = int(gpu_index_match.group(1))

                            # Toplam bellek ve kullanılan belleği çıkartalım
                            total_memory_match = re.search(r'(\d+) \/ (\d+) MB', line)
                            if total_memory_match:
                                total_memory = int(total_memory_match.group(2))
                                used_memory = int(total_memory_match.group(1))

                                free_memory = total_memory - used_memory
                                free_gpus.append({"index": gpu_index, "free_memory": free_memory})


        return free_gpus

class MyTrainingArguments(BaseModel):
    max_steps:int = 12
    save_steps:int = 5
    num_train_epochs:int = 5
    base_model_name:str = "meta-llama/Llama-3.2-1B" #pythia-1.4b, T5-base, gemma2:2b
    dataset_path:str = "ModelTrainingWorkspace/DataLoad/Files"
    gpu_count = 2

    update_data_to_twitter_agent:bool=False
    learning_rate:float=5.0e-5
    metric_for_best_model:str="eval_loss"
    disable_tqdm:bool=True
    eval_steps:int=1# Number of update steps between two evaluations
    save_steps:int=10 # After # steps model is saved
    warmup_steps:int=1 # Number of warmup steps for learning rate scheduler
    per_device_eval_batch_size:int=8 # Batch size for evaluation
    evaluation_strategy:str="steps"
    logging_strategy:str="steps"
    optim:str="adafactor"
    gradient_accumulation_steps:int = 4
    gradient_checkpointing:bool=False

class MyPeftTrainingArguments(BaseModel):
    finetuned_model_paths:list = ["meta-llama/Llama-3.2-1B"]
    first_peft_train : bool = False
    existing_peft_model_path:str = "" # if first_peft_train is false, you must write peft model path
    first_train_step:int = 1
    update_model_step:int=1

"""
SEQ_CLS = "SEQ_CLS"
SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
CAUSAL_LM = "CAUSAL_LM"
TOKEN_CLS = "TOKEN_CLS"
QUESTION_ANS = "QUESTION_ANS"
FEATURE_EXTRACTION = "FEATURE_EXTRACTION"
# for pythia task_type = TaskType.CAUSAL_LM,   "target_modules": ["query_key_value"]
# for FLAN-T5 task_type = TaskType.SEQ_2_SEQ_LM  ["q", "v"]
bias = ['none', 'all' ,'lora_only']
"""
class MyLoraConfig(BaseModel):
 
    r:int=32 # Rank
    lora_alpha:int=64
    target_modules:list=["query_key_value"]
    lora_dropout:float=0.1
    bias:str="none"
    task_type:str|TaskType="CAUSAL_LM"  
        
    
tokenizer = ""
max_token_len = 0
def data_load_function(dataset_names, test_split_size=0.1,validation_split_size=0.2):
    """
    Loads a dataset from files and splits it into train, test, and validation datasets.
    
    Parameters:
    dataset_names (list): List of paths to the dataset files.
    test_split_size (float, optional): The proportion of the dataset to use for testing. Defaults to 0.1.
    validation_split_size (float, optional): The proportion of the dataset to use for validation. Defaults to 0.2.
    
    Returns:
    DatasetDict: A dictionary containing the train, test, and validation datasets.
    """
    file_names = [os.path.basename(dataset_name) for dataset_name in dataset_names]
    folder = os.path.dirname(dataset_names[0])
    train_end = (100-100*(test_split_size+validation_split_size))
    test_end = (100-100*(validation_split_size))
    splits = {0: train_end, train_end: test_end, test_end:100}
    try:
        data = [load_dataset(path=folder,data_files=file_names,split=f'train[{int(k)}:{int(l)}%]') for k,l in splits.items()]
    except:
        data = [load_dataset(path=folder,data_files=file_names,split=f'test[{int(k)}:{int(l)}%]') for k,l in splits.items()]
        
    seed_value = random.randint(1, 1000) 
    test_data = pd.DataFrame({"test":data[1].shuffle(seed=seed_value)})
    test_data = pd.json_normalize(test_data['test'])
    test_data.to_csv(f"{folder}/Test_{file_names[0]}")
    return DatasetDict({"train":data[0].shuffle(seed=seed_value),"test":data[1].shuffle(seed=seed_value),"validation":data[2].shuffle(seed=seed_value)})

class CalculateTime:
    def __init__(self, info:str) -> None:
        self.info = info
    
    def __enter__(self):
        self.st = time.time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.et = time.time()
        elapsed_time = self.et - self.st
        print(f'\n{self.info} runtime:', elapsed_time, 'seconds')
        print(f'\n{self.info} runtime:', elapsed_time/60, 'minute')
        print(f'\n{self.info} runtime:', elapsed_time/(60*60), 'hour\n')


def get_finetuned_models(base_model_name, model_paths,local_files_only):
    """
    Loads multiple pre-trained models from the provided model paths and returns them as a list.
    
    This function iterates over the provided model paths and loads each model using the AutoModelForCausalLM.from_pretrained() method.
    The models are loaded onto available GPUs, and the function returns a list of the loaded models. If all GPUs are exhausted,
    the function returns an empty list.
    
    Parameters:
    base_model_name (str): The name of the base model used for fine-tuning.
    model_paths (list[str]): A list of paths to the pre-trained models.
    local_files_only (bool): If True, the function will only load models from local files and will not attempt to download models from the internet.
    
    Returns:
    list[AutoModelForCausalLM]: A list of loaded models.
    """
    device_id = 0
    finetuned_models = []
    with CalculateTime("Models loaded"):
        for i,model_path in enumerate(model_paths):
            
                device_id = FreeGpuDetector().get_first_free_cuda_index()
                assert device_id<4 , "Device ID should be smaller  than 4. Check the your GPU. Check if you have free space "
                if device_id<4:

                    if  "t5" in base_model_name:
                        AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,local_files_only=local_files_only, device_map = {"": device_id})
                    else:
                        finetuned_slightly_model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=local_files_only,device_map={"":device_id})
                        finetuned_models.append(finetuned_slightly_model)
                else:
                    return finetuned_models
            
    return finetuned_models


def get_response_by_causal_peft_model(text, model, tokenizer, max_input_tokens, max_output_tokens):
  # Tokenize
  
  """
  Generates a response using the given causal PEFT model, tokenizer, and input text.
  
  This function takes in a string of text, a PEFT model, a tokenizer, and the maximum number of input and output tokens.
  It encodes the input text using the tokenizer, generates a response using the model, decodes the generated tokens
  back into text, and strips the input prompt from the output text. If the output text is shorter than 15 words,
  it increases the maximum output tokens by 20 and calls itself recursively.
  
  Parameters:
  text (str): The input text to generate a response from.
  model (PEFT model): The causal PEFT model to use for generation.
  tokenizer (tokenizer): The tokenizer to use for encoding the input text.
  max_input_tokens (int): The maximum number of input tokens to use.
  max_output_tokens (int): The maximum number of output tokens to generate.
  
  Returns:
  str: The generated response after stripping the input prompt.
  """
  input_ids = tokenizer.encode(
          text,
          return_tensors="pt",
          truncation=True,
          max_length=max_input_tokens,
  )
  #input_ids = tokenizer(text,return_tensors="pt",truncation=True,max_length=max_token_len, padding="max_length")
  # Generate
  device = model.device
  input_ids=input_ids.to(device)
  generation_config=GenerationConfig(max_new_tokens=max_output_tokens, num_beams=1,top_p=0.5,temperature=0.3,do_sample=True,repetition_penalty=1.3,pad_token_id=0 )
  #generated_tokens_with_prompt = model.generate(input_ids, max_length=max_output_tokens,top_p=0.5,temperature=0.3,do_sample=True,repetition_penalty=1.3)
  generated_tokens_with_prompt = model.generate(input_ids=input_ids, generation_config=generation_config)
  
  # Decode
  generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

  # Strip the prompt
  generated_text_answer = generated_text_with_prompt[0][len(text)+1:]
  generated_text_answer = ''.join(filter(str.isprintable, generated_text_answer))
  if max_output_tokens > 160:
    return generated_text_answer
  if len(generated_text_answer.split(" ")) < 15:
        max_output_tokens += 20
        return get_response_by_causal_peft_model(text, model, tokenizer, max_input_tokens, max_output_tokens)

  return generated_text_answer

def get_tokenizer(model_name):
    #TODO diğer model type larına göre güncelle
    """
    Returns a tokenizer for the given model name, with the bos token set to "" and a padding token added.

    Parameters:
    model_name (str): The name of the model to get a tokenizer for.

    Returns:
    tokenizer: A tokenizer with the bos token set to "" and a padding token added.
    """
    cuda_index = FreeGpuDetector().get_first_free_cuda_index()
    tokenizer = AutoTokenizer.from_pretrained(model_name,device=cuda_index)
    tokenizer.bos_token = "<|startoftext|>"
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = 0
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    return tokenizer



def get_max_output_tokens_function(prompt, max_output_tokens):
    if "Your friend tweet" in prompt:
        pattern = r"Your friend tweet:\n(.*?)\n###  Your reply:"
        
    elif "Quoted tweet" in prompt:
        pattern = r"Quoted tweet:\n(.*?)\n###  Your answer:"
        
    else: 
        return max_output_tokens
    is_there_any_text = is_there_any_text_function(pattern, prompt)
    
    if is_there_any_text:
        max_output_tokens += 40
        return max_output_tokens
    else:
        return False
    
 
"""import gc
for model in finetuned_models:
    del model
    gc.collect()"""   

def is_there_any_text_function(pattern, prompt):
    if re.findall(pattern, prompt):
        is_there_any_text = len(re.findall(pattern, prompt)[0])
        if  is_there_any_text:
            return True
    return False

#def is_there_any_text_function2(pattern, prompt):
    #match = re.search(pattern, prompt.replace("\n",""))
    #return bool(match)

instructions = ["You are a social person. You are active on Twitter. ### Tweet:n\{tweet}"]


def modified_test_results(test_data_result_df:pd.DataFrame)->pd.DataFrame:
    #This function is modified the result to eveluate from Character analysis (Postman)
    """
    This function is modified the result to eveluate from Character analysis (Postman)

    For each prompt in the test_data_result_df, it checks if the prompt contains any of the base_instructions.
    If it does, it replaces the instruction with an empty string and appends the expected response and evaluated response to modifed_expected_responses and modified_evaluate_responses respectively.
    Otherwise, it appends the original expected response and evaluated response to modifed_expected_responses and modified_evaluate_responses respectively.

    The function then creates a new dataframe, modified_data_result_df, with the modified expected responses and evaluated responses.
    It drops the original Expected Responses and Evaluate Responses columns and returns the modified dataframe.

    Parameters:
    test_data_result_df (pd.DataFrame): The dataframe containing the test results.

    Returns:
    modified_data_result_df (pd.DataFrame): The modified dataframe with the expected responses and evaluated responses modified according to the character analysis instructions.
    """
    base_instructions = ["You are a social person. You are active on Twitter.",
                    "You're very friedly and you are active on social media. You like tweets that you are fond of.",
                    "You're quite sociable, and you maintain an active presence on social platforms. You reply to some your friend tweets.",
                    "You're extremely sociable, and you maintain a lively presence on social platforms. You answer  the a few quoted tweet.",
                    "You're highly sociable and have an active presence on social media. You retweet some tweets."]
    modifed_expected_responses = []
    modified_evaluate_responses = []
    for i, prompt in enumerate(test_data_result_df['Prompts']):
        matched_instruction = None
        for instruction in base_instructions:
            if instruction in prompt:
                matched_instruction = instruction
                break

        if matched_instruction is not None:
            without_instruction_prompt = prompt.replace(matched_instruction,"")
            expexted_tweet = without_instruction_prompt + f" {test_data_result_df['Expected Responses'][i]}"
            evaluated_tweet = without_instruction_prompt + f" {test_data_result_df['Evaluate Responses'][i]}"
            
            modifed_expected_responses.append(expexted_tweet)
            modified_evaluate_responses.append(evaluated_tweet)
        else:
            modifed_expected_responses.append(test_data_result_df['Expected Responses'][i])
            modified_evaluate_responses.append(test_data_result_df['Evaluate Responses'][i])

    modified_data_result_df = test_data_result_df
    modified_data_result_df['Expected_Responses'] = modifed_expected_responses
    modified_data_result_df['Evaluated_Responses'] = modified_evaluate_responses
    modified_data_result_df.drop(columns=['Expected Responses', 'Evaluate Responses'], inplace=True)
    return modified_data_result_df


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

def tokenize_peft_function(tokenizer,example):
    
    """
    Tokenizes the prompt and output text in the example using the provided tokenizer.

    This function takes a tokenizer and an example containing a fine-tuning prompt and an output.
    It tokenizes both the prompt and the output text, padding and truncating to the maximum length,
    and returns a modified example with the tokenized input IDs and labels moved to GPU 0.

    Parameters:
    tokenizer: A tokenizer object with a method for encoding text into input IDs.
    example (dict): A dictionary containing "fine_tune_prompt" and "output" keys.

    Returns:
    dict: The input example updated with 'input_ids' and 'labels' keys containing
          tokenized tensors moved to GPU 0.
    """

    prompt = [p for p in example["fine_tune_prompt"]]
    print(prompt)
    # Tokenize prompt
    input_ids = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['input_ids'] = input_ids.cuda(0) 
    
    # Tokenize summary
    labels = tokenizer(example['output'], padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = labels.cuda(0)  # Move to GPU 0
    
    return example

def split_text(text: str, max_chars=2000):
    """
    Splits a given text into multiple chunks based on the maximum character length (max_chars).

    If the length of the text is less than or equal to max_chars, the text is returned as is.
    Otherwise, the function splits the text into chunks by words, adding each word to the current chunk
    until the length of the current chunk is greater than max_chars. The function then returns the
    current chunk, stripped of any trailing whitespace.

    Parameters:
    text (str): The text to split.
    max_chars (int): The maximum character length of each chunk.

    Returns:
    str: The split text, or the original text if it is shorter than or equal to max_chars.
    """
    if len(text) <= max_chars:
        return text
    words = text.split()
    current_chunk_length = 0

    for word in words:
        if current_chunk_length + len(word) <= max_chars:
            current_chunk_length += len(word) + 1  # Add 1 for the space between words
        else:
            break
    return text[:current_chunk_length].strip()
    
def tokenize_function(examples,key=None):
    #TODO#Max chars should be updated according to token
    
    keys = [key, "instruction_data_separatelly_en", "train_prompts","prompts_en","prompts_tr","prompts","instruction_data_separatelly"]
    
    for key in keys:
        if key:
            try:
                if len(examples[key])>5000:
                    token_len = tokenizer(examples[key],return_tensors="np",padding=True)["input_ids"].shape[1]
                text = split_text(examples[key], max_chars=4000)
                    
                break
            except KeyError:
                pass
            
    tokenized_inputs = tokenizer(text,return_tensors="np",padding=True)

    #tokenized_inputs = tokenizer(text,return_tensors="np",truncation=True,max_length=max_token_len, padding="max_length")
    tokenized_inputs = tokenizer(text,return_tensors="np",truncation=True,max_length=max_token_len)
    #TODO: Here label ids and input ids may need to be changed and the effect of added firm paddig should be observed.
    return {"input_ids": tokenized_inputs["input_ids"][0],"attention_mask": tokenized_inputs["attention_mask"][0],"labels": tokenized_inputs["input_ids"][0]  # Assuming you want labels to be the same as input_ids
            }

def convert_dataset_to_tokenized_datasets(dataset):
   
    tokenized_data = {}
    if "train" in dataset:
        tokenized_data["train"] = dataset["train"].map(tokenize_function)
    if "validation" in dataset:
        tokenized_data["validation"] = dataset["validation"].map(tokenize_function)
    if "test" in dataset:
        tokenized_data["test"] = dataset["test"].map(tokenize_function)
    if len(tokenized_data) > 0:
        return DatasetDict(tokenized_data)
    
    return dataset.map(tokenize_function)
    
def removing_unnecessary_columns_to_train_with_pythia_model(train_data:DatasetDict):
    if not isinstance(train_data,DatasetDict):
        raise "Data type must be DatasetDict"
    return train_data.remove_columns([col for col in train_data["train"].column_names if col not in ["input_ids", "attention_mask", "labels"]])

def get_max_len_token(train_data, key=None):
    max_token_len = 0
    keys = [key,"instruction_data_separatelly_en","train_prompts","prompts_en","prompts_tr","prompts","instruction_data_separatelly"]
    
    for key in keys:
        if key:
            try:
                text =  train_data["train"][key]
                break
            except KeyError:
                pass
    for text in train_data["train"][key]:
        
        token_len = tokenizer(text,return_tensors="np",padding=True)["input_ids"].shape[1]
        if token_len> max_token_len:
            max_token_len = token_len
    return 2048 if max_token_len > 2048 else max_token_len
"""
def tokenize_function(tokenizer, examples):
    propmt = examples["First_Words"]
    i= 1
    tokenized_propmt= tokenizer(propmt,return_tensors="np",padding=True)
    if i==1:
      
      max_length = min( tokenized_propmt["input_ids"].shape[1],2048 )
    i +=1
    tokenized_propmt = tokenizer(propmt,return_tensors="np",truncation=True,max_length=max_length,padding=True)
    #label = examples["Remaining_Text"]
    #tokenized_label = tokenizer(label,return_tensors="np",truncation=True,max_length=max_length,padding=True)
    return {"input_ids": tokenized_propmt["input_ids"][0],"attention_mask": tokenized_propmt["attention_mask"][0],"labels": tokenized_label["input_ids"][0]  # Assuming you want labels to be the same as input_ids
            }
"""