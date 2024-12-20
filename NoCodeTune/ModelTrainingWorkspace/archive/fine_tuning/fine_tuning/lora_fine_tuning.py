from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, Trainer
import time
from datasets import load_dataset,DatasetDict
from dataclasses import asdict, dataclass, field
try:
    from fine_tuning.fine_tuning import FineTuning
except:
    from fine_tuning import FineTuning
    
import torch


class LoraFineTuning(FineTuning):

    def __init__(self, base_model, tokenizer, dataset:DatasetDict, peft_training_output_dir= f'./peft-training-output_dir-{str(int(time.time()))}') -> None:
        super().__init__(peft_training_output_dir)
        
        self.tokenizer = tokenizer
        self.base_model =base_model
        self.dataset = dataset
        self.check_dataset_format(dataset)
    
    def check_dataset_format(self, dataset)->None:
        if dataset:
            if not isinstance(dataset, DatasetDict):
                raise "Dataset format must be DatasetDict"
            elif not ("train" in dataset and "validation" in dataset):
                raise "Dataset format must include train and validation"
        
    
    def set_initial_parameters(self):
        self.lora_config : LoraConfig = None
        self.peft_model = None
        self.training_args:TrainingArguments = None

    def set_lora_config(self, rank=32, lora_alpha=64, target_modules=["q", "v"], lora_dropout=0.1, bias="none", 
                                task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
                                ) -> LoraConfig:
        """
        # for pythia task_type = TaskType.CAUSAL_LM,   "target_modules": ["query_key_value"]
        # for FLAN-T5 task_type = TaskType.SEQ_2_SEQ_LM  ["q", "v"]
        bias = ['none', 'all' ,'lora_only']
        """
        self.lora_config = LoraConfig(
            r=rank, # Rank
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=task_type  
        )
        x = 1
    
    def set_peft_model_by_orginal_model_and_lora_config(self, base_model, lora_config):
     
        self.peft_model = get_peft_model(base_model,
                                    lora_config)

    def set_peft_model_by_updated_peft_model(self, peft_model= None):
        self.peft_model = peft_model

    def update_peft_tunigs(self, base_model, tokenizer, dataset, peft_training_output_dir):
        self.tokenizer = tokenizer
        self.base_model =base_model
        self.dataset = dataset
        self.update_output_dir(peft_training_output_dir)
        
    def train_the_model_with_lora(self, peft_model, tokenizer, dataset, training_args ):
        
        self.train_the_model(peft_model, tokenizer, dataset, training_args)

    def save_trained_model_and_tokenizer(self):

        self.trainer.model.save_pretrained( self.peft_training_output_dir)
        self.tokenizer.save_pretrained(self.peft_training_output_dir)
    
