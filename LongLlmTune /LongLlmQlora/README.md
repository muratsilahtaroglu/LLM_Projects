# LongLlmQlora

## Overview

The **LongLlmQlora** project is designed to fine-tune large language models for long-context tasks efficiently. It utilizes techniques such as LoRA, DeepSpeed, gradient checkpointing, and 4-bit quantization to optimize GPU memory and enable scalable, multi-GPU training. The repository is structured into modules for data preparation, training, evaluation, and inference.

---

## Main Purpose

This project aims to enhance large language models (LLMs) for long-context scenarios, including tasks such as analyzing multi-chapter books, lengthy academic papers, and longitudinal datasets, all while minimizing computational and memory demands.

Using QLoRA fine-tuning, we extend the context length of **Llama-3.1-8B** from **8K to 80K** tokens. The training process is remarkably efficient, completing in just **10x24 hours on an 4xA6000 GPU machine (48G)** The fine-tuned model excels in long-context tasks like NIHS, topic retrieval, and extended language comprehension, while retaining strong performance in short-context applications.  

Notably, this context extension relies on only 3.5K synthetic data generated by GPT-4, highlighting the untapped potential of LLMs to handle extended contexts. With additional resources, the model's context length could surpass 80K tokens.
---

## Workflow

### Data Preparation

- Generate structured datasets for various tasks (QA, summarization, etc.) using raw input files.
- Use pre-built pipelines to handle "one-detail," "multi-detail," and biography-based tasks.

### Model Training

- Fine-tune large models with efficient memory optimization techniques (LoRA, 4-bit quantization, and gradient checkpointing).
- Leverage multi-GPU support through DeepSpeed for scaling.

### Evaluation

- Assess model performance using a variety of benchmarks and tasks, such as language modeling, topic retrieval, and needle-in-a-haystack scenarios.
- Compute metrics like ROUGE, perplexity, accuracy, and more.

### Inference

- Utilize chat templates and message-handling utilities for conversational and task-specific applications.
- Support for tokenization and efficient processing of long sequences.

---

## Key Features

- **Long-Context Fine-Tuning**: Handles extensive datasets with specialized techniques.
- **GPU Optimization**: Reduces memory usage with 4-bit quantization, mixed precision training, and LoRA.
- **Multi-GPU Scaling**: Employs DeepSpeed ZeRO3 to distribute training loads.
- **Custom Evaluation**: Includes metrics and benchmarks tailored for long-context tasks.

---

## How to Run

### Data Preparation

Use the scripts in the `data_pipeline` module to generate task-specific datasets.


# Environment
```bash
conda create -n unsloth python=3.10
conda activate unsloth

conda install pytorch==2.2.2 pytorch-cuda=12.1 cudatoolkit xformers -c pytorch -c nvidia -c xformers
pip install transformers==4.39.3 deepspeed accelerate datasets==2.18.0 peft bitsandbytes
pip install flash-attn --no-build-isolation
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# these packages are used in evaluation

pip install rouge fuzzywuzzy jieba pandas seaborn python-Levenshtein
```


**NOTE**: you must modify the source code of `unsloth` so that you can set the `rope_theta` correctly in training. Go to `$ENC_LOCATION$/lib/python3.10/site-packages/unsloth/models/llama.py`, comment all lines from `1080-1088`. The results should be like:
```python
if (rope_scaling is None) and (max_seq_length > model_max_seq_length):
    rope_scaling = max_seq_length / model_max_seq_length
    logger.warning_once(
        f"Unsloth: {model_name} can only handle sequence lengths of at most "\
        f"{model_max_seq_length}.\nBut with kaiokendev's RoPE scaling of "\
        f"{round(rope_scaling, 3)}, it can be magically be extended to "\
        f"{max_seq_length}!"
    )
    rope_scaling = {"type": "linear", "factor": rope_scaling,}
```


Full-attention models cannot run with more than 60K context length on a single A800 GPU. Parallel strategies are required. I use [`tensor_parallel`](https://github.com/BlackSamorez/tensor_parallel). However, `tensor_parallel` does not support `transformers>=4.36`. You should create another environment while downgrade to `transformers==4.35.1` and install `tensor_parallel`:
```bash
conda create -n full --clone unsloth
conda activate full

pip install transformers==4.35.1 datasets==2.14.5 tensor_parallel
```


**IMPORTANT NOTE**

For any path specified for `train_data` and `eval_data`: if it is prefixed with `long-llm:`, it will be solved to the relative path against `data_root`. 
  - for example, `long-llm:redpajama/train.json` -> `${data_root}/redpajama/train.json`
  - you can modify the default value of [`data_root`](src/args.py), so that you don't need to type it for each command.


### Training

Run the training script to fine-tune the model:
**Examples**
1. 
```bash
torchrun --nproc_per_node 4 -m main.train \
--data_root /data/long-llm \
--output_dir outputs/llama3.1-8B/lora_model \
--model_name_or_path meta-llama/Meta-Llama-3.1-8B \
--train_data long-llm:gpt/one_detail_book.train.64K.json long-llm:gpt/one_detail_paper.train.64K.json long-llm:gpt/multi_detail_book.train.json long-llm:gpt/multi_detail_paper_short.train.json long-llm:gpt/multi_detail_paper_long.train.json long-llm:gpt/bio_book.train.json long-llm:longalpaca/train.json long-llm:redpajama/train.json[5000] \
--max_length 80000 \
--rope_theta 200e6 \
--gradient_checkpointing \
--attn_impl flash_attention_2 \
--learning_rate 5e-5 \
--num_train_epochs 1 \
--save_strategy epoch \
--bf16 \
--lora_tune \
--lora_extra_params embed_tokens \
--load_in_4_bit

```

2. 
```bash
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3 /home/muratsilahtoroglu/.conda/envs/unsloth2_v2/bin/torchrun --master_port 29503 --nproc_per_node 4  -m my_train --data_root /data/long-llm --output_dir outputs/llama3.1-8B/lora_model --model_name_or_path outputs/llama3.1-8B/full_model_checkpoint-282546_merged_Llama-3-8B --train_data data/r_data/r_all_and_q_a.json --max_length 40000 --group_by_length --rope_theta 200e6 --attn_impl flash_attention_2 --gradient_checkpointing --use_reentrant True --learning_rate 5e-5 --num_train_epochs 50 --save_strategy epoch --logging_steps 5 --bf16 --lora_tune --lora_extra_params embed_tokens --load_in_4_bit --chat_template llama-3 > logs/llama3.1-8B.out 2>&1 & 
```


**waching gpu command**

```bash 
nvidia-smi 
#or 
gpustat --watch
``` 


# Evaluation

All evaluation results will be saved at `data/results/`.

## LoRA Model
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# base model id
model=meta-llama/Meta-Llama-3.1-8B
# lora model id
lora=outputs/llama3.1-8B/lora_model

COMMAND="--data_root /data/long-llm --model_name_or_path $model --lora $lora --rope_theta 200e6 --attn_impl flash_attention_2 --chat_template llama-3"

source /opt/conda/bin/activate unsloth

torchrun --nproc_per_node 4 -m main.eval_longbench --max_length 80000 $COMMAND
torchrun --nproc_per_node 4 -m main.eval_topic $COMMAND
torchrun --nproc_per_node 4 -m main.eval_mmlu $COMMAND

source /opt/conda/bin/activate full

python -m main.eval.eval_needle $COMMAND --min_length 8000 --max_length 80000 --enable_tp
python -m main.eval.eval_infbench $COMMAND --max_length 80000 --enable_tp

# you can use GPT4-o as the scorer with the following command:
# export OPENAI_API_KEY="sk-xxxx"
# python -m main.eval_needle $COMMAND --min_length 8000 --max_length 80000 --enable_tp --gpt_eval
```


## Full Model
```python
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "outputs/llama3.1-8B/lora_model-Merged"

torch_dtype = torch.bfloat16
# place the model on GPU
device_map = {"": "cuda"}

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  torch_dtype=torch.bfloat16,
  device_map=device_map,
  attn_implementation="flash_attention_2",
).eval()

with torch.no_grad():
  # short context
  messages = [{"role": "user", "content": "Tell me about yourself."}]
  inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
  outputs = model.generate(**inputs, max_new_tokens=50)[:, inputs["input_ids"].shape[1]:]
  print(f"Input Length: {inputs['input_ids'].shape[1]}")
  print(f"Output:       {tokenizer.decode(outputs[0])}")

  # long context
  with open("data/narrativeqa.json", encoding="utf-8") as f:
    example = json.load(f)
  messages = [{"role": "user", "content": example["context"]}]
  inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
  outputs = model.generate(**inputs, do_sample=False, top_p=1, temperature=1, max_new_tokens=20)[:, inputs["input_ids"].shape[1]:]
  print("*"*20)
  print(f"Input Length: {inputs['input_ids'].shape[1]}")
  print(f"Answers:      {example['answer']}")
  print(f"Prediction:   {tokenizer.decode(outputs[0])}")
```
