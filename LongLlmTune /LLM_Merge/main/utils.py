import os
import gc
import tempfile

import torch
import random
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any

from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, is_torch_npu_available


def load_llm(model_name:str, trust_remote_code:bool):
    """
    Load a language model from Hugging Face Hub.

    Args:
    - model_name: The name of the model on Hugging Face Hub.
    - trust_remote_code: Whether to trust the remote code on the Hub.

    Returns:
    - An instance of `AutoModelForCausalLM` that is loaded from the Hub.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code, device_map = {"": "cpu"})
    return model


def load_embedder(model_name:str, trust_remote_code:bool):
    """
    Load a sentence embedder from Hugging Face Hub.

    Args:
    - model_name: The name of the model on Hugging Face Hub.
    - trust_remote_code: Whether to trust the remote code on the Hub.

    Returns:
    - An instance of `AutoModel` that is loaded from the Hub.
    """
    model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code, device_map = {"": "cpu"})
    return model


def load_reranker(model_name:str, trust_remote_code:bool):
    """
    Load a reranker model from Hugging Face Hub.

    Args:
    - model_name: The name of the model on Hugging Face Hub.
    - trust_remote_code: Whether to trust the remote code on the Hub.

    Returns:
    - An instance of `AutoModelForSequenceClassification` that is loaded from the Hub.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=trust_remote_code, device_map = {"": "cpu"})
    return model


def load_seq2seq_model(model_name:str, trust_remote_code:bool):
    """
    Load a sequence-to-sequence model from Hugging Face Hub.

    Args:
    - model_name: The name of the model on Hugging Face Hub.
    - trust_remote_code: Whether to trust the remote code on the Hub.

    Returns:
    - An instance of `AutoModelForSeq2SeqLM` that is loaded from the Hub.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    return model


def load_model(model_name:str, model_type:str, trust_remote_code:bool=True):
    if model_type == 'decoder':
        model = load_llm(model_name, trust_remote_code=trust_remote_code)
    elif model_type == 'encoder':
        model = load_embedder(model_name, trust_remote_code=trust_remote_code)
    elif model_type == 'reranker':
        model = load_reranker(model_name, trust_remote_code=trust_remote_code)
    elif model_type == 'encoder-decoder':      
        model = load_seq2seq_model(model_name, trust_remote_code=trust_remote_code)
    else:
        raise NotImplementedError(f"not support this model_type: {model_type}")
    return model


def get_model_param_list(model_names: List[str], model_type:str):
    """
    Load a list of models and return their parameters.

    Args:
    - model_names: A list of model names on Hugging Face Hub.
    - model_type: The type of the models, which is one of 'decoder', 'encoder', 'reranker', or 'encoder-decoder'.

    Returns:
    - A list of dictionaries, where each dictionary contains the parameters of a model.
    """
    model_param_list = []
    for name in model_names:
        print(f"loading {name} -----------------")
        model = load_model(name, model_type=model_type)
        model_param_list.append(model.state_dict())
    return model_param_list


def merge_param(model_param_list: List[Dict], weights: List[float]):
    """
    Merge the parameters of a list of models with given weights.

    Args:
    - model_param_list: A list of dictionaries, where each dictionary contains the parameters of a model.
    - weights: A list of floats, where each float is the weight of the corresponding model in the list.

    Returns:
    - A dictionary, where each key-value pair is the merged parameter of the input models.
    """
    new_param = {}
    for k in model_param_list[0].keys():
        for w, param in zip(weights, model_param_list):
            if param[k].dtype == torch.int64 or param[k].dtype == torch.int32:
                new_param[k] = param[k]
            elif k not in new_param:
                new_param[k] = w * param[k]
            else:
                new_param[k] += w * param[k]
    return new_param


def get_model_param_dirs(model_names: List[str], model_type:str):
    """
    Load the parameters of a list of models and save them to temporary directories.

    The function loads each model in the list, saves its parameters to a temporary directory,
    and returns a list of paths to the temporary directories and the path to the root temporary directory.

    Args:
    - model_names: A list of model names on Hugging Face Hub.
    - model_type: The type of the models, which is one of 'decoder', 'encoder', 'reranker', or 'encoder-decoder'.

    Returns:
    - A tuple, where the first element is a list of paths to the temporary directories,
      and the second element is the path to the root temporary directory.
    """
    param_dirs = []
    temp_dir = tempfile.mkdtemp()
    print(f"create a temporary directory: {temp_dir}")

    for idx, name in enumerate(model_names):
        print(f"loading {name} -----------------")
        model = load_model(name, model_type=model_type)
        model_params = model.state_dict()

        model_temp_dir = os.path.join(temp_dir, f"model_{idx+1}")
        os.makedirs(model_temp_dir, exist_ok=True)
        param_dirs.append(model_temp_dir)

        for k, v in model_params.items():
            temp_param_file = os.path.join(model_temp_dir, f"{k}.ckpt")
            torch.save(v, temp_param_file)

        model = model.to("meta")
        del model_params
        gc.collect()

    return param_dirs, temp_dir


def merge_param_by_layer(model_param_dirs: List[str], weights: List[float]):
    """
    Merge model parameters across multiple directories using specified weights.

    This function combines model parameters stored in temporary directories, 
    weighting each parameter by the provided weights, to generate a new set 
    of merged parameters. The resulting merged parameters are saved in a 
    temporary file.

    Args:
    - model_param_dirs (List[str]): A list of paths to directories containing 
      model parameters for each model to be merged.
    - weights (List[float]): A list of weights corresponding to each model, 
      used to scale their parameters during merging. The length must match 
      the number of directories in `model_param_dirs`.

    Returns:
    - str: The file path to the temporary file containing the merged 
      parameters.
    """

    new_param = {}
    model_params = os.listdir(model_param_dirs[0])

    for param_file in tqdm(model_params, desc="Merging models"):
        param_name = param_file.replace(".ckpt", "")

        for w, model_dir in tqdm(zip(weights, model_param_dirs), total=len(weights), desc=f"Processing {param_name}", leave=False):            
            file_path = os.path.join(model_dir, param_file)
            param = torch.load(file_path)

            if param.dtype in [torch.int64, torch.int32]:
                new_param[param_name] = param
            elif param_name not in new_param:
                new_param[param_name] = w * param
            else:
                new_param[param_name] += w * param

            del param
            gc.collect()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".ckpt") as tmp_file:
        print(f"create a temporary file to store mixed weights: {tmp_file.name}")
        torch.save(new_param, tmp_file.name)
        temp_file_path = tmp_file.name

    del new_param
    gc.collect()

    return temp_file_path


def compute_weights(base_model, tokenizer, param_list: List[Dict], model_type: str, example_data: List[Any], temperature: float=5.0, batch_size:int=2, max_input_length:int=2048, neg_number:int=7):
    """
    Compute weights for model merging.

    Parameters:
    - base_model: The base model used to compute the loss for the given example data.
    - tokenizer: The tokenizer used to preprocess the example data.
    - param_list: A list of dictionaries, each containing the parameters of a model.
    - model_type: The type of model: decoder, encoder, or encoder-decoder.
    - example_data: A list of example data.
    - temperature: The temperature used to compute the weights.
    - batch_size: The batch size used to preprocess the example data.
    - max_input_length: The maximum input length used to preprocess the example data.
    - neg_number: The number of negative examples used to preprocess the example data.

    Returns:
    - weights: A list of weights, each corresponding to the weight of a model in the merged model.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif is_torch_npu_available():
        device = torch.device("npu")
    else:
        device = torch.device("cpu")
    base_model = base_model.to(device)
    
    if model_type == 'decoder':
        input_data = preprocess_data_for_llm(example_data=example_data, tokenizer=tokenizer, device=device, batch_size=batch_size, max_input_length=max_input_length)
        loss_func = llm_loss
    elif model_type == 'encoder':
        input_data = preprocess_data_for_embedder(example_data=example_data, tokenizer=tokenizer, device=device, batch_size=batch_size, max_input_length=max_input_length, neg_number=neg_number)
        loss_func = embedder_loss
    elif model_type == 'encoder-decoder':     
        input_data = preprocess_data_for_seq2seq(example_data=example_data, tokenizer=tokenizer, device=device, batch_size=batch_size, max_input_length=max_input_length)
        loss_func = seq2seq_loss

    example_loss = [] 
    with torch.no_grad():
        for params in param_list:
            base_model.load_state_dict(params)
            loss = loss_func(base_model=base_model, input_data=input_data)
            example_loss.append(loss)
    
    weights = torch.softmax(-torch.FloatTensor(example_loss)/temperature, -1).numpy().tolist()
    return weights



def preprocess_data_for_seq2seq(example_data, tokenizer, device, batch_size:int=2, max_input_length:int=512):       # Added Reimer
    """
    Preprocess example data for seq2seq tasks.

    Args:
        example_data: A list of example data, each containing an 'input' and an 'output' key.
        tokenizer: The tokenizer used to preprocess the data.
        device: The device on which the data will be processed.
        batch_size: The batch size used to preprocess the example data.
        max_input_length: The maximum input length used to preprocess the example data.

    Returns:
        A list of preprocessed data, each containing an 'input_ids', an 'attention_mask', and a 'labels' key.
    """
    batch_data = []
    for i in range(0, len(example_data), batch_size):
        batch_examples = example_data[i:i+batch_size]
        input_texts = [ex['input'] for ex in batch_examples]
        target_texts = [ex['output'] for ex in batch_examples]

        input_encodings = tokenizer(input_texts, text_target=target_texts, max_length=max_input_length, padding=True, truncation=True, return_tensors="pt")

        input_ids = input_encodings.input_ids.to(device)
        attention_mask = input_encodings.attention_mask.to(device)
        labels = input_encodings.labels.to(device)

        labels[labels == tokenizer.pad_token_id] = -100
        batch_data.append({
            "input_ids": input_ids, 
            "attention_mask": attention_mask,
            "labels": labels
        })
    return batch_data



def preprocess_data_for_embedder(example_data, tokenizer, device, batch_size:int=64, max_input_length:int=512, neg_number:int=7):
    """
    Preprocess example data for embedding tasks.

    This function tokenizes and processes query and passage data for embedding
    models, preparing batches for input to the model. It includes handling of
    both positive and negative passages for contrastive learning.

    Args:
        example_data: A list of example data, each containing a 'query', 'pos' (positive passages),
                      and 'neg' (negative passages) key.
        tokenizer: The tokenizer used to preprocess the data.
        device: The device on which the data will be processed.
        batch_size: The batch size used to preprocess the example data.
        max_input_length: The maximum input length used to preprocess the example data.
        neg_number: The number of negative examples to sample for each query.

    Returns:
        A list of preprocessed data batches, each containing tokenized queries
        and passages, ready for input to the embedding model.
    """

    input_data = []
    quries = []
    passages = []
    # max_input_length = min(512, max_input_length)
    for e in example_data:
        quries.append(e['query'])
        passages.append(e['pos'][0])
        passages.extend(random.sample(e['neg'], neg_number))

        if len(quries) == batch_size:
            q_tokens = tokenizer(quries, padding=True, truncation=True, max_length=max_input_length, return_tensors="pt")
            p_tokens =  tokenizer(passages, padding=True, truncation=True, max_length=max_input_length, return_tensors="pt")
            q_tokens, p_tokens = q_tokens.to(device), p_tokens.to(device)
            input_data.append([q_tokens, p_tokens])
            quries, passages = [], []

    if len(quries) > 0:
        q_tokens = tokenizer(quries, padding=True, truncation=True, max_length=max_input_length, return_tensors="pt")
        p_tokens =  tokenizer(passages, padding=True, truncation=True, max_length=max_input_length, return_tensors="pt")
        q_tokens, p_tokens = q_tokens.to(device), p_tokens.to(device)
        input_data.append([q_tokens, p_tokens])
        
    return input_data


def seq2seq_loss(base_model, input_data):
    """
    Calculate the average loss for a sequence-to-sequence model.

    This function computes the average loss over a dataset for a given
    sequence-to-sequence model. It iterates through batches of input data,
    performs a forward pass using the model, and accumulates the loss
    for each batch. The final average loss is returned as a float.

    Args:
        base_model: The sequence-to-sequence model used for computing the loss.
        input_data: A list of dictionaries, each containing 'input_ids',
                    'attention_mask', and 'labels' keys, representing batches
                    of data.

    Returns:
        float: The average loss across all batches in the input data.
    """

    total_loss = 0
    with torch.no_grad():
        for batch in input_data:
            outputs = base_model(input_ids=batch["input_ids"], 
                            attention_mask=batch["attention_mask"], 
                            labels=batch["labels"])
            total_loss += outputs.loss.cpu()
    average_loss = total_loss / len(input_data)
    return float(average_loss)


def embedder_loss(base_model, input_data):
    """
    Calculate the average loss for a sequence-to-sequence model with an embedder head.

    This function computes the average loss over a dataset for a given
    sequence-to-sequence model with an embedder head. It iterates through batches of
    input data, performs a forward pass using the model, and accumulates the loss
    for each batch. The final average loss is returned as a float.

    Args:
        base_model: The sequence-to-sequence model used for computing the loss.
        input_data: A list of dictionaries, each containing 'input_ids',
                    'attention_mask', and 'labels' keys, representing batches
                    of data.

    Returns:
        float: The average loss across all batches in the input data.
    """
    
    def generate_embeddings(model, inputs):
        embeddings = model(**inputs, return_dict=True).last_hidden_state[:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)        
        return embeddings

    with torch.no_grad():
        loss = 0
        for q_inputs, p_inputs in input_data:
            q_embeddings = generate_embeddings(base_model, q_inputs)
            p_embeddings = generate_embeddings(base_model, p_inputs)
            scores = torch.matmul(q_embeddings, p_embeddings.transpose(0, 1)) / 0.05
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_embeddings.size(0) // q_embeddings.size(0))
            batch_loss = torch.nn.CrossEntropyLoss(reduction='mean')(scores, target)
            loss += batch_loss.cpu()
    loss = float(loss / len(input_data))
    return float(loss)


def preprocess_data_for_llm(example_data, tokenizer, device, batch_size:int=2, max_input_length:int=2048):
    """
    Preprocesses example data for a language model, so that it can be used for loss computation.

    Args:
        example_data: A list of dictionaries, each containing 'input' and 'output' keys, representing examples of data.
        tokenizer: The tokenizer used to encode the input and output strings.
        device: The device to put the resulting tensors on.
        batch_size: The size of each batch of data.
        max_input_length: The maximum length of the input sequence.

    Returns:
        A list of dictionaries, each containing 'input_ids', 'labels', and 'attention_mask' keys, representing batches of preprocessed data.
    """
    batch_input_ids = []
    batch_labels = []
    batch_max_length = max_input_length
    for data in example_data:
        input, output = data['input'], data['output']
            
        input_ids = tokenizer.encode(input+' '+output)
        input_ids.append(tokenizer.eos_token_id)
            
        prompt_ids = tokenizer.encode(input)
        labels = [-100]*len(prompt_ids) + input_ids[len(prompt_ids):]
    
        input_ids = input_ids[:batch_max_length]
        input_ids += [tokenizer.pad_token_id] * (batch_max_length - len(input_ids))
        batch_input_ids.append(input_ids)
        
        labels = labels[:batch_max_length]
        labels += [-100] * (batch_max_length - len(labels))
        batch_labels.append(labels)
        
    batch_input_ids = torch.LongTensor(batch_input_ids).to(device)
    batch_labels = torch.LongTensor(batch_labels).to(device)
    attention_mask = batch_input_ids.ne(tokenizer.pad_token_id).to(device)
    
    batch_data = []
    for i in range(0, len(batch_input_ids), batch_size):
        batch_data.append(dict(
            input_ids=batch_input_ids[i:i+batch_size],
            labels=batch_labels[i:i+batch_size],
            attention_mask=attention_mask[i:i+batch_size],
            ))
    return batch_data



def llm_loss(base_model, input_data):
    """
    Computes the average loss for a language model over a dataset.

    This function takes a language model and a list of preprocessed input data,
    and returns the average loss across all batches in the input data.

    Args:
        base_model: The language model used for computing the loss.
        input_data: A list of dictionaries, each containing 'input_ids',
                    'labels', and 'attention_mask' keys, representing batches
                    of data.

    Returns:
        float: The average loss across all batches in the input data.
    """
    loss = 0
    with torch.no_grad():
        for data in input_data:
            output = base_model(**data)
            loss += output.loss.cpu()
    loss = float(loss / len(input_data))
    return loss




