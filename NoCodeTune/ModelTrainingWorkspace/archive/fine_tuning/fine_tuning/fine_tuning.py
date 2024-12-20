from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, Trainer,DataCollatorForSeq2Seq
import os
import torch
import glob
import re
from datasets import DatasetDict
try :
    import fine_tuning.run_models_and_generate_tweets_utils as Utils
    from fine_tuning.modified_tweet_dataset_to_train import Modified_Tweet_Dataset_to_Train_Agent
except:
    import run_models_and_generate_tweets_utils as Utils
    from modified_tweet_dataset_to_train import Modified_Tweet_Dataset_to_Train_Agent
    

class FineTuning:

    def __init__(self, training_output_dir:str=None) -> None:
        if training_output_dir:
            self.training_output_dir = self.edit_output_dir(training_output_dir)
        self.utils :Utils= Utils
        self.data = None
    

    def set_traning_args(self,tuned_args:Utils.MyTrainingArguments|Utils.MyPeftTrainingArguments):
        """
        Sets the training arguments for model fine-tuning.

        This function initializes a `TrainingArguments` object with a specified 
        output directory and updates its attributes based on the provided 
        `tuned_args`. The attributes are dynamically set using the keys and 
        values from the `tuned_args` dictionary. Additionally, it configures 
        the number of GPUs to be used during training.

        Args:
            tuned_args: An instance of either `MyTrainingArguments` or 
                        `MyPeftTrainingArguments` containing the training 
                        parameters to be applied.
        """

        args = TrainingArguments(output_dir=self.training_output_dir,local_rank=-1) # local_rank: Rank of the process during distributed training.
        #args.__dict__.update(tuned_args.dict())
        for key, value in tuned_args.dict().items():
            setattr(args, key, value)
       
        self.training_args = args
        self.training_args._n_gpu = tuned_args.gpu_count
        #self.training_args._n_gpu = 1 if wanted fine tune use just one gpu
    
    def train_the_model(self, model, tokenizer, datasets, training_args:TrainingArguments):
        """
        Trains the model using the provided datasets and training arguments.

        This function sets up a Trainer from the Hugging Face Transformers library 
        to train a specified model. It prepares the data collator needed for 
        sequence-to-sequence tasks, initializes the Trainer with model, datasets, 
        and training arguments, and starts the training process. The trained model, 
        tokenizer, and configurations are saved to the specified output directory.

        Args:
            model: The model to be trained.
            tokenizer: The tokenizer corresponding to the model.
            datasets: A dictionary containing 'train' and 'validation' datasets.
            training_args (TrainingArguments): Configuration arguments for training.

        Saves:
            The trained model, tokenizer, and configurations to the output directory 
            specified in training_args.
        """

        os.makedirs(training_args.output_dir, exist_ok=True)
        data_collator = None
       
            
        data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset = datasets["validation"],
            data_collator=data_collator,
            tokenizer= tokenizer )
       
        self.trainer.train()
        tokenizer.save_pretrained(training_args.output_dir)
        self.trainer.model.save_pretrained(training_args.output_dir)
        self.trainer.model.config.save_pretrained(training_args.output_dir)
        self.trainer.model.generation_config.save_pretrained(training_args.output_dir)


    def update_output_dir(self, training_output_dir):
        self.training_output_dir = training_output_dir
    
    def get_tokenized_datasets(self,model_name:str, dataset_path:str, update_data_to_twitter_agent = False):
        
        self.data = self.get_dataset( model_name, dataset_path, update_data_to_twitter_agent)
        tokenized_datasets = self.utils.convert_dataset_to_tokenized_datasets( self.data)
        tokenized_datasets = self.utils.removing_unnecessary_columns_to_train_with_pythia_model(tokenized_datasets)
        return tokenized_datasets
    
    def get_dataset(self, base_model_name:str, dataset_path:str, update_data_to_twitter_agent=False):
        if update_data_to_twitter_agent:
            print(f"--------{dataset_path}")
            modified_tweet_dataset_to_train_agent = Modified_Tweet_Dataset_to_Train_Agent(df_paths=[dataset_path])
            dataset_path = modified_tweet_dataset_to_train_agent.updated_data_paths[0]
        data = self.utils.data_load_function([dataset_path])
        self.utils.tokenizer =  self.utils.get_tokenizer(model_name = base_model_name)
        self.utils.max_token_len = self.utils.get_max_len_token(data)
        return data
    
    def get_data(self):
        return self.data
    
    def edit_output_dir(self, output_dir:str)->str:
        """
        Modifies the specified output directory path to include a versioning scheme.

        This function checks if the given `output_dir` contains a versioning component
        starting with '/V'. If not, it appends '/V1_1' to the path and creates the 
        directory if it does not exist. If the path includes '/checkpoint', it removes 
        that part and updates the directory path to reflect the next available version 
        by analyzing existing subfolders.

        Args:
            output_dir (str): The original output directory path.

        Returns:
            str: The modified output directory path.
        """
        if "/V" not in output_dir:
             print("*******",output_dir)
             output_dir =  output_dir +"/V1_1"
             if not os.path.isdir(output_dir):
                 
                os.makedirs(output_dir, exist_ok=True)
                return output_dir
        output_dir = output_dir.split("checkpoint")[0]
       #TODO: ~ test deleted file
        match = re.search(r'(.+?)(?:\/checkpoint-\d+|\/~?V\d+(?:_?\d+)?)?\/?$', output_dir)
        main_path = match.group(1)
        sub_folders_count = len(glob.glob(main_path+"/*"))
        sub_folder_version = re.search(r'~?V(\d+)(?:_\d+)?', output_dir).group(1) if re.search(r'~?V(\d+)(?:_\d+)?', output_dir) else 1 
        output_dir = f"{main_path}/V{sub_folder_version}_{sub_folders_count+1}"
        return output_dir
