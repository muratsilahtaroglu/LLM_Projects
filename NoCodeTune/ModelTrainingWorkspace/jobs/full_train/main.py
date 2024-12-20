
from validate import ArgsModel,MyTrainingArguments,MyPeftTrainingArguments,MyLoraConfig,ProcessArguments
from pydantic import BaseModel
import pydantic_argparse
import sys
from server_utils import *
import argparse
import os
from transformers.modeling_utils import PreTrainedModel

def arg_parser():
    parser = argparse.ArgumentParser(description='Full train parameters')
    
    parser.add_argument('--max_steps', type=int, help='Max train step')
    parser.add_argument('--num_train_epochs', type=int, help='Number train epochs')

    return parser.parse_args()

def args_to_dict(args):
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    return args_dict

def update_args(obj, args:BaseModel,key:str=None):
    #args BaseModel den kalıtım alınan argümanlardır sözlük yapısı taklit edildi.
    try:
        value = args if key is None else args.model_dump()[key]
    except:
        value = args if key is None else args.dict()[key]
    tuned_args = value
    if tuned_args:
        obj.__dict__.update(tuned_args)
    return obj

if __name__ =="__main__":
    """
    log_out_fs = open("logs/stdout.log", "w", encoding="utf-8")
    
    def write(__s):
        # global log_out_fs
        result = log_out_fs.write(__s)
        log_out_fs.flush()
        return result
    # log_err_fs = open("logs/stderr.log", "w", encoding="utf-8")
    sys.stdout.write = write
    sys.stderr.write = write
    """
    import  fine_tuning.fine_tuning_and_updating_agent_model as agent
    parser = pydantic_argparse.ArgumentParser(
        model=ProcessArguments,
        
        prog="Training Arguments",
        description="Training Arguments",
        version="0.0.1",
        epilog="Example Training Arguments",
    )
    process_args = parser.parse_typed_args()
    
    
    
    job_args = get_job_args()
    args = job_args[0].args
    args = ArgsModel(**args)
    args.train_type
    train_type = args.train_type
    #process_args = arg_parser()
    process_args_dict = args_to_dict(process_args)
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    if train_type.lower() == "full_train":
        training_args = update_args(MyTrainingArguments(), args,"training_args")
        if process_args_dict:
            training_args.__dict__.update(process_args_dict)
        print(training_args)
        agent.run_train(training_args = training_args,output_dir=OUTPUTS_DIRNAME)
        
    elif train_type.lower() == "peft_train":
        training_args = update_args(MyTrainingArguments(), args,"training_args")
        peft_args = update_args(MyPeftTrainingArguments(), args,"peft_args")
        lora_configs = update_args(MyLoraConfig(), args,"lora_configs")

        agent.run_peft_train(
            training_args=training_args,
            peft_args=peft_args,
            lora_configs=lora_configs,
            output_dir=OUTPUTS_DIRNAME
        )
    else:
        raise f"Undefined train_type. Check job_args path"
    
    #log_out_fs.close()



