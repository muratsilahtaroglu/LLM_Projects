import math
import torch
from tqdm import tqdm
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Mapping, Optional, Tuple
from accelerate import Accelerator
from collections import defaultdict
from transformers.modeling_outputs import BaseModelOutputWithPast


def optional_grad_ctx(with_grad=False):
    if with_grad:
        return nullcontext()
    else:
        return torch.no_grad()

def move_to_device(data, device):
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: move_to_device(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(move_to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        return data.to(**kwargs)
    else:
        return data

def compute_loss(logits, labels, shift=False):
 
    """
    Compute the loss for a batch of sequences.

    Args:
        logits: the output of the model, a tensor of shape (batch_size, seq_len, vocab_size)
        labels: the labels, a tensor of shape (batch_size, seq_len)
        shift: whether to shift the logits and labels by one token to compute the loss

    Returns:
        loss: the total loss, a scalar
        batch_loss: the loss for each element in the batch, a tensor of shape (batch_size)
        valid_token_num: the number of valid tokens in the batch, a tensor of shape (batch_size)
    """
    if shift:
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

    labels = labels.to(logits.device)
    batch_size = logits.shape[0]

    # TODO: the loss on -100 labels is 0 by default
    token_loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), 
        labels.reshape(-1), 
        reduction="none"
    ).reshape(batch_size, -1)   # batch_size, seq_len
    
    valid_token_num = (labels != -100).sum(-1)  # batch_size
    all_valid_token_num = valid_token_num.sum()
    
    if all_valid_token_num > 0:
        loss = token_loss.sum() / valid_token_num.sum()
    else:
        loss = token_loss.sum()

    batch_loss = token_loss.sum(-1) / valid_token_num
    # prevent nan
    if (valid_token_num == 0).any():
        batch_loss = batch_loss.masked_fill(valid_token_num == 0, 0.)

    return loss, batch_loss, valid_token_num


@torch.no_grad()
def evaluate_perplexity(model, dataloader, accelerator:Optional[Accelerator]=None):
    """
    Evaluate the perplexity of a model on a dataset.

    Args:
        model: The model to evaluate.
        dataloader: The dataloader to use for evaluation.
        accelerator: Optional; the accelerator to use for distributed evaluation.

    Returns:
        The perplexity score of the model over the dataset.
    """

    if accelerator is not None and type(dataloader) == torch.utils.data.DataLoader:
        # if the dataloader has been prepared, I shall not prepare it twice, especially in case of deepspeed
        dataloader = accelerator.prepare(dataloader)

    # if accelerator.process_index == 0:
    #     for name, x in model.named_parameters():
    #         print(f"{name: ^80} {x.dtype}")

    all_loss = defaultdict(list)
    for i, x in enumerate(tqdm(dataloader, desc="Computing Perplexity")):
        # TODO: important to reset memory for every batch
        if hasattr(model, "memory"):
            model.memory.reset()

        # the seq id
        index = x.pop("index")
        # length is used to group training data, no use here
        length = x.pop("length", None)

        output = model(**x)

        # TODO: I need the loss for each element in the batch for accurate computation, because the number of valid tokens may differ among elements
        if hasattr(output, "batch_loss"):
            # output from our model has batch_loss by default
            batch_loss = output.batch_loss
            valid_token_num = output.valid_token_num
        else:
            # output from other models does not
            loss, batch_loss, valid_token_num = compute_loss(output.logits, x["labels"], shift=True)

        if accelerator is not None and accelerator.num_processes > 1:
            # num_device * batch_size
            index = accelerator.gather_for_metrics(index)
            batch_loss = accelerator.gather_for_metrics(batch_loss)
            valid_token_num = accelerator.gather_for_metrics(valid_token_num)

        for _id, _loss, _num in zip(index.tolist(), batch_loss.tolist(), valid_token_num.tolist()):
            # loss times num is the total loss of all valid tokens
            all_loss[_id].append((_loss * _num, _num))

    for _id, loss_and_num in all_loss.items():
        # sum up the loss for all valid tokens in the entire sequence, and divide the number of valid tokens
        all_loss[_id] = sum([x[0] for x in loss_and_num]) / sum(x[1] for x in loss_and_num)
    
    # average across then take exp
    perplexity = math.exp(sum(all_loss.values()) / len(all_loss))
    return perplexity


@torch.no_grad()
def evaluate_generation(model, dataloader, accelerator:Optional[Accelerator]=None, tokenizer=None, return_new_tokens_only=True, return_decoded=True, **generation_config):
    """
    Evaluate a model on a generation task.

    Args:
        model: The model to evaluate. Should be a torch.nn.Module.
        dataloader: The dataloader to use for evaluation.
        accelerator: The accelerator to use for distributed evaluation.
        tokenizer: The tokenizer to use for decoding.
        return_new_tokens_only: Whether to return only the newly generated tokens.
        return_decoded: Whether to decode the output tokens to strings.
        **generation_config: Additional configuration for the generation.

    Returns:
        A tuple of two lists: indices and outputs. indices contains the sequence ids in the order of the dataloader,
        and outputs contains the generated tokens in the same order.
    """
    if accelerator is not None and type(dataloader) == torch.utils.data.DataLoader:
        # if the dataloader has been prepared, I shall not prepare it twice, especially in case of deepspeed
        dataloader = accelerator.prepare(dataloader)

    all_indices = []
    all_outputs = []
    
    for i, x in enumerate(tqdm(dataloader, desc="Computing Generation")):
        # if i > 3:
        #     break
        
        # TODO: important to reset memory for every batch
        if hasattr(model, "memory"):
            model.memory.reset()

        indices = x.pop("index")
        # length is used to group training data, no use here
        length = x.pop("length", None)

        outputs = model.generate(**x, **generation_config)
        if return_new_tokens_only:
            start_idx = x["input_ids"].shape[1]
            outputs = outputs[:, start_idx:]

        if accelerator is not None and accelerator.num_processes > 1:
            # must be contiguous
            outputs = accelerator.pad_across_processes(outputs.contiguous(), pad_index=tokenizer.pad_token_id, dim=1)
            outputs = accelerator.gather_for_metrics(outputs)
            indices = accelerator.gather_for_metrics(indices)

        outputs = outputs.tolist()
        indices = indices.tolist()
        if return_decoded:
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_indices.extend(indices)
        all_outputs.extend(outputs)

    return all_indices, all_outputs


@torch.no_grad()
def evaluate_nll(model, dataloader, accelerator:Optional[Accelerator]=None):
    """
    Evaluate the negative log likelihood of a model on a dataset.

    Args:
        model: the model to evaluate
        dataloader: the dataloader to use
        accelerator: the accelerator to use for distributed evaluation

    Returns:
        A dictionary mapping sequence ids to their corresponding negative log likelihood
    """
    if accelerator is not None and type(dataloader) == torch.utils.data.DataLoader:
        # if the dataloader has been prepared, I shall not prepare it twice, especially in case of deepspeed
        dataloader = accelerator.prepare(dataloader)

    # if accelerator.process_index == 0:
    #     for name, x in model.named_parameters():
    #         print(f"{name: ^80} {x.dtype}")

    all_loss = defaultdict(list)
    for i, x in enumerate(tqdm(dataloader, desc="Computing Perplexity")):
        # TODO: important to reset memory for every batch
        if hasattr(model, "memory"):
            model.memory.reset()

        # the seq id
        index = x.pop("index")
        # length is used to group training data, no use here
        length = x.pop("length", None)

        output = model(**x)

        # TODO: I need the loss for each element in the batch for accurate computation, because the number of valid tokens may differ among elements
        if hasattr(output, "batch_loss"):
            # output from our model has batch_loss by default
            batch_loss = output.batch_loss
            valid_token_num = output.valid_token_num
        else:
            # output from other models does not
            loss, batch_loss, valid_token_num = compute_loss(output.logits, x["labels"], shift=True)

        if accelerator is not None and accelerator.num_processes > 1:
            # num_device * batch_size
            index = accelerator.gather_for_metrics(index)
            batch_loss = accelerator.gather_for_metrics(batch_loss)
            valid_token_num = accelerator.gather_for_metrics(valid_token_num)

        for _id, _loss in zip(index.tolist(), batch_loss.tolist()):
            # loss times num is the total loss of all valid tokens
            all_loss[_id].append(_loss)

    return all_loss



@dataclass
class BeaconModelOutput(BaseModelOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    batch_loss: Optional[torch.FloatTensor] = None
    valid_token_num: Optional[torch.LongTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
