
from typing import Literal
from pydantic import BaseModel
from peft import TaskType

class ProcessArguments(BaseModel):
    max_steps:int | None 
    save_steps:int | None 
    num_train_epochs:int | None 
    base_model_name:str | None 
    dataset_path:str | None 

    update_data_to_twitter_agent:bool | None
    learning_rate:float | None 
    metric_for_best_model:str | None 
    disable_tqdm:bool | None 
    eval_steps:int | None 
    save_steps:int | None 
    warmup_steps:int | None
    per_device_eval_batch_size:int | None 
    evaluation_strategy:str | None 
    logging_strategy:str | None
    optim:str | None 
    gradient_accumulation_steps:int | None  
    gradient_checkpointing:bool | None 
class MyTrainingArguments(BaseModel):
    max_steps:int = 12
    save_steps:int = 5
    num_train_epochs:int = 5
    base_model_name:str = "EleutherAI/pythia-410m"
    dataset_path:str = "datatuneflow-dataset-service/DataLoad/Files"

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
    finetuned_model_paths:list = ["EleutherAI/pythia-410m"]
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

class ArgsModel(BaseModel):
    train_type: Literal["full_train", "peft_train"]
    training_args: dict
    peft_args: dict | None
    lora_configs: dict | None
  
