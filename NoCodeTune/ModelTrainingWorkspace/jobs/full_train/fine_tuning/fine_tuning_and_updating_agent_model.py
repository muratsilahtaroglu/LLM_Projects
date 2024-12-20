try :
    import run_models_and_generate_tweets_utils as utils
    import lora_fine_tuning
    from fine_tuning import FineTuning
except:
    import fine_tuning.run_models_and_generate_tweets_utils as utils
    import fine_tuning.lora_fine_tuning as lora_fine_tuning
    from fine_tuning.fine_tuning import FineTuning 
    
from peft import  TaskType,PeftModel
from transformers import AutoModelForSeq2SeqLM
import torch
import json
import time
import os
import glob
import re


class FineTuningAndTrainPeftModel:
    def __init__(self, training_args:utils.MyTrainingArguments, base_model=None, tokenizer=None, dataset=None, output_dir="") -> None:
        self.training_args = training_args
        self.base_model_name = training_args.base_model_name
        self.dataset_path = training_args.dataset_path
        self.output_dir = output_dir
        self.peft_model = None
        self.peft_fine_tuning : lora_fine_tuning.LoraFineTuning= lora_fine_tuning.LoraFineTuning(base_model, tokenizer, dataset, output_dir)
        self.fine_tuning = FineTuning(training_output_dir= None)
    
    def update_lora_tunings(self,base_model, tokenizer, dataset, output_dir):
        self.base_model= base_model
    
    def get_tuned_datasets(self,model_name:str, dataset_path:str,update_data_to_twitter_agent=False) :
        """
        Gets the tokenized dataset for the PEFT model. This method is used to get the tokenized dataset for the PEFT model.
        It takes the model name, dataset path and a boolean flag to update the data to Twitter agent format as arguments.
        It returns the tokenized dataset.

        Parameters:
        model_name (str): The name of the model to get the tokenized dataset for.
        dataset_path (str): The path to the dataset.
        update_data_to_twitter_agent (bool): A boolean flag to update the data to Twitter agent format. Defaults to False.
        """
        return self.peft_fine_tuning.get_tokenized_datasets(model_name,dataset_path,update_data_to_twitter_agent)
    
    def get_pretrained_peft_model(self,base_model, model_path="", torch_dtype=torch.bfloat16, device_id=0, is_trainable=False):
        
        """
        Loads a pre-trained PEFT model from the specified base model and model path.

        This method initializes a PEFT model using the provided base model and loads the weights
        from the specified model path. The model can be configured to be trainable and can be 
        loaded onto a specific device.

        Parameters:
        base_model (PreTrainedModel): The base model to be used for initializing the PEFT model.
        model_path (str): The path to the pre-trained model weights. Defaults to an empty string.
        torch_dtype (torch.dtype): The data type for the model's tensors. Defaults to torch.bfloat16.
        device_id (int): The GPU device ID where the model should be loaded. Defaults to 0.
        is_trainable (bool): If True, the model parameters will be set to require gradients.
        
        Returns:
        PeftModel: The initialized PEFT model with pre-trained weights.
        """

        peft_model = PeftModel.from_pretrained(base_model,
                                               model_path,
                                                is_trainable=is_trainable, 
                                                #torch_dtype=torch_dtype,
                                                device_map = {"": device_id})
        return peft_model
    
    def get_base_models(self, finetuned_model_paths):
        fine_tuned_models = utils.get_finetuned_models(self.base_model_name, finetuned_model_paths,True)
        return fine_tuned_models
    
    def set_peft_fine_tuning(self, base_model, tokenizer, dataset, peft_model_dir="", peft_model=None,lora_configs:utils.MyLoraConfig=None) :
        """
        This function is used to set the PEFT fine tuning model. It takes the base model, tokenizer, dataset, PEFT model directory, PEFT model, and lora configuration as arguments.

        It first sets the PEFT model and the output directory. It then updates the PEFT fine tuning model with the given base model, tokenizer, dataset, and output directory.

        If a PEFT model is given, it sets the PEFT model by the updated PEFT model. If not, it sets the PEFT model by the original base model and the lora configuration.

        Parameters:
        base_model (PreTrainedModel): The base model to be used for fine tuning.
        tokenizer (PreTrainedTokenizerFast): The tokenizer to be used for tokenizing the dataset.
        dataset (DatasetDict): The dataset to be used for fine tuning.
        peft_model_dir (str): The directory of the PEFT model to be used for fine tuning. If not given, it will be set to the output directory with the given base model name.
        peft_model (PeftModel): The PEFT model to be used for fine tuning. If not given, it will be set to None.
        lora_configs (MyLoraConfig): The lora configuration to be used for fine tuning. If not given, it will be set to the default configuration.

        Returns:
        None
        """
        self.peft_model = peft_model
        self.output_dir = self.fine_tuning.edit_output_dir(peft_model_dir)
        self.peft_fine_tuning.update_peft_tunigs(base_model, tokenizer, dataset, self.output_dir)
        
        self.tuning_lora_config(peft_model_dir,lora_configs)
        
        if peft_model:
            self.peft_fine_tuning.set_peft_model_by_updated_peft_model(peft_model)
            
        else:
            self.peft_fine_tuning.set_peft_model_by_orginal_model_and_lora_config(base_model=self.peft_fine_tuning.base_model, 
                                                lora_config=self.peft_fine_tuning.lora_config)
    def tuning_lora_config(self,peft_model_dir,lora_configs:utils.MyLoraConfig=None):
        """
        This function is used to set the adapter configuration of the PEFT model.

        It first checks if the lora_configs is None or the PEFT model is not None. If so, it loads the adapter configuration from the json file in the PEFT model directory.
        If not, it sets the adapter configuration based on the base model name and the given lora configuration.

        Parameters:
        peft_model_dir (str): The directory of the PEFT model.
        lora_configs (MyLoraConfig): The adapter configuration.

        Returns:
        None
        """
        if lora_configs is None or self.peft_model:
            lora_configs = self.get_adapter_config_kwards(peft_model_dir)
            #biases = ['none', 'all' ,'lora_only']
        if not isinstance(lora_configs,dict):
            lora_configs = lora_configs.dict()
        self.peft_fine_tuning.set_lora_config(rank=lora_configs["r"],lora_alpha=lora_configs["lora_alpha"] ,lora_dropout=lora_configs["lora_dropout"], 
                                                target_modules=lora_configs["target_modules"] ,task_type=lora_configs["task_type"],bias=lora_configs["bias"])

    def get_adapter_config_kwards(self,peft_model_dir):
       #TODO: task type and target_modules should be added according to other types.
  
        """
        This function is used to get the adapter configuration from a PEFT model directory.

        It loads the adapter configuration from a json file in the PEFT model directory and returns it as a dictionary.

        If the PEFT model is not None, it loads the adapter configuration from the json file.
        If the PEFT model is None, it sets the adapter configuration based on the base model name.
        If the base model name contains  causal models such as "llama", "gemma" or"pythia" , it sets the target modules to ['query_key_value'] and the task type to TaskType.CAUSAL_LM.
        If the base model name contains "t5", it sets the target modules to ["q", "v"] and the task type to TaskType.SEQ_2_SEQ_LM.

        Parameters:
        peft_model_dir (str): The directory of the PEFT model.

        Returns:
        dict: The adapter configuration as a dictionary.
    """
        kwards = {"target_modules": [], "task_type": None, "r": 32,"lora_alpha": 64,"lora_dropout": 0.1,"bias" : "lora_only"}

        if self.peft_model:
            with open(f'{peft_model_dir}/adapter_config.json', 'r') as file:
                adapter_config = json.load(file)
            kwards["target_modules"] = adapter_config.get('target_modules', [])
            kwards["task_type"] = adapter_config.get('task_type')
            kwards["r"] = adapter_config.get('r', 32)
            kwards["lora_alpha"] = adapter_config.get('lora_alpha', 64)
            kwards["lora_dropout"] = adapter_config.get('lora_dropout', 0.1)
            kwards["bias"] = adapter_config.get('bias', "lora_only")
        
        elif "t5" in self.base_model_name:
            kwards["target_modules"] = ["q", "v"]
            kwards["task_type"] = TaskType.SEQ_2_SEQ_LM
        else:
            #TODO will be added in the future other traget modules
            kwards["target_modules"] = ['query_key_value']
            kwards["task_type"] = TaskType.CAUSAL_LM

        return kwards

    def train_peft_model(self, peft_fine_tuning:lora_fine_tuning.LoraFineTuning):
        peft_fine_tuning.set_traning_args(self.training_args)#TODO EDIT
        peft_fine_tuning.train_the_model_with_lora(peft_fine_tuning.peft_model, peft_fine_tuning.tokenizer, 
                                                peft_fine_tuning.dataset, peft_fine_tuning.training_args)
        self.peft_fine_tuning = peft_fine_tuning
        self.peft_model = peft_fine_tuning.trainer.model

def run_peft_train(training_args:utils.MyTrainingArguments=None,peft_args:utils.MyPeftTrainingArguments=None,lora_configs:utils.MyLoraConfig=None,output_dir:str="outputs"):
    """
    This function is used to train the Lora Model for a given task. It should be called after the full fine-tuning model is trained and saved.
    
    The function takes the following arguments:
    - training_args: a MyTrainingArguments object which contains the training arguments for the Lora model.
    - peft_args: a MyPeftTrainingArguments object which contains the arguments for the Lora training.
    - lora_configs: a MyLoraConfig object which contains the arguments for the Lora model.
    - output_dir: the directory where the Lora model will be saved.
    
    The function does the following:
    1. If first_peft_train is True, it loads the full fine-tuning model and does the following steps: 
        -Step 1: Loads the full fine-tuning model.
        -Step 2: Peft Tuning.
        -Step 3: Train Peft Model.
    2. If first_peft_train is False, it loads the pre-trained Lora model and does the following steps:
        -Step 1: Loads the pre-trained Lora model.
        -Step 2: Update Peft Tuning.
        -Step 3: Train Peft Model.
    
    The function returns the trained Lora model.
    """
    fine_tuning_and_train_peft_model = FineTuningAndTrainPeftModel(training_args)
   
    ####Chapter 1: If first peft train
    peft_model_dir = peft_args.existing_peft_model_path
    if peft_args.first_peft_train:
        #Step 1: Load Full Fine Tuning Model
        i=0
        
        while i<peft_args.first_train_step:
            tokenized_datasets = fine_tuning_and_train_peft_model.get_tuned_datasets(training_args.base_model_name, training_args.dataset_path,training_args.update_data_to_twitter_agent)
            base_model = fine_tuning_and_train_peft_model.get_base_models(peft_args.finetuned_model_paths)[0]
            #Step 2 : Peft Tuning
            first_peft_train_model_dir =f"{output_dir}/peft_models"
            fine_tuning_and_train_peft_model.set_peft_fine_tuning(base_model, utils.tokenizer, tokenized_datasets, first_peft_train_model_dir,peft_model=None,lora_configs=lora_configs)
            
            #Step 3 : Train Peft Model
            fine_tuning_and_train_peft_model.train_peft_model(fine_tuning_and_train_peft_model.peft_fine_tuning)
            peft_model = fine_tuning_and_train_peft_model.peft_model
            peft_model_dir = fine_tuning_and_train_peft_model.output_dir
            i += 1
    
    ####Chapter 2: 
    #Step 1  Load Full Fine Tuning Model
    i=0
    while i < peft_args.update_model_step:
        if not peft_model_dir or len(peft_model_dir)<1:
            raise "write peft_model_dir"
        tokenized_datasets = fine_tuning_and_train_peft_model.get_tuned_datasets(training_args.base_model_name, training_args.dataset_path,training_args.update_data_to_twitter_agent)
        base_model = fine_tuning_and_train_peft_model.get_base_models(peft_args.finetuned_model_paths)[0]
        
        device_id = utils.FreeGpuDetector().get_first_free_cuda_index()
        assert device_id<4 , "Device ID should be smaller than 4. Check the your GPU. Check if you have free space"
        #TODO:base model eğer verilebilirse peft_model_dir pathinden alınsın
        peft_model = fine_tuning_and_train_peft_model.get_pretrained_peft_model(base_model=base_model, model_path = peft_model_dir,
                                                                                device_id = device_id, is_trainable=True)
        #Step 2 : Uptade Peft Tuning
        fine_tuning_and_train_peft_model.set_peft_fine_tuning(base_model, utils.tokenizer, tokenized_datasets, peft_model_dir, peft_model=peft_model,lora_configs=lora_configs)

        #Step 3 : Train Peft Model
        fine_tuning_and_train_peft_model.train_peft_model(fine_tuning_and_train_peft_model.peft_fine_tuning)
        peft_model_dir = fine_tuning_and_train_peft_model.output_dir
        i += 1

def run_train(training_args:utils.MyTrainingArguments ,output_dir ):
    
    """
    This function is used to train a model. It takes MyTrainingArguments and output_dir as arguments.

    It first sets the train output directory. Then it creates a FineTuning object and sets the training arguments.
    It loads the tokenized dataset and loads the data. Then it trains the model and prints a message to show that the model has been created.

    Parameters:
    training_args (MyTrainingArguments): The arguments for training the model.
    output_dir (str): The directory to save the trained model.

    Returns:
    None
    """
    train_output_dir = f"{output_dir}/models"
    model_fine_tuning = FineTuning(train_output_dir)
    print("-----------------",model_fine_tuning.training_output_dir)
    model = utils.get_finetuned_models(training_args.base_model_name, [training_args.base_model_name],True)[0]
    model_fine_tuning.set_traning_args(training_args)
    tokenized_datasets = model_fine_tuning.get_tokenized_datasets(training_args.base_model_name, training_args.dataset_path, training_args.update_data_to_twitter_agent)
    data = model_fine_tuning.get_data()
    
    model_fine_tuning.train_the_model( model, utils.tokenizer, tokenized_datasets, model_fine_tuning.training_args)
    print(f"Created: {train_output_dir} model")

