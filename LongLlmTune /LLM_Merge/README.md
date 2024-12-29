# LLM_Merge

**Make fine-tuning of language models akin to crafting a nuanced merge.**

Model merging can improve the performance of a single model. `LLM_Merge` is designed to help enhance large language models (LLMs) and dense embedding models. This strategy enables automatic merging of fine-tuned models and base models by computing optimal merging weights. `LLM_Merge` can boost performance on a target domain without degrading general capabilities, and it can even generate new task-specific models without requiring additional fine-tuning.

---

## Applications

### 1. Mitigating Catastrophic Forgetting
Fine-tuning a base language model can sometimes lead to a decline in general capabilities. By merging the fine-tuned model and the base model using `mix_models`, `LLM_Merge` ensures improved performance on downstream tasks while retaining the model's overall versatility.

If you have multiple fine-tuned models, you can:
1. Provide example data from your task.
2. Use `mix_models_with_data` to compute weights and merge the models intelligently.
3. Finally, use `mix_models` to merge the resulting model with your fine-tuned model.

---

### 2. Improving Performance for New Tasks Without Fine-Tuning
`LLM_Merge` allows the creation of models tailored to new tasks without training. By providing a few examples and pre-existing models, `mix_models_with_data` computes task-specific merging weights and creates a model that fits the new task.

---

### 3. Approximating Multitask Learning
Merge several task-specific models into a single model using `mix_models`. This approach approximates multitask learning, enabling the resulting model to handle multiple tasks effectively.

---

## Installation

Install the latest version from source:
```bash
git clone https://github.com/your-repo/LLM_Merge.git
cd LLM_Merge
pip install -e .
pip install -U LLM_Merge

```

There are there key functions in LLM_Merge: `mix_models`, `mix_models_with_data` and `mix_models_by_layers`.

### 1. Mix models

`mix_models` can merge models based on the given merging weights.
An example is merging the fine-tuned model and 
the base model to mitigate Catastrophic Forgetting after fine-tuning:

```python
from LLM_Merge.main import mix_models, mix_models_with_data

# mix LLMs and save it to output_path: ./mixed_model_1
model = mix_models(
    model_names_or_paths=["meta-llama/Llama-3.1-8b", "llama3-ag-news"], 
    model_type='decoder', 
    weights=[0.7, 0.3], 
    output_path='./mixed_llm')
# you can select a weight for your models to get a trade-off between generality and expertise.

# Mix Embedding Models
model = mix_models(
    model_names_or_paths=["BAAI/bge-base-en-v1.5", "bge-hotpotqa"], 
    model_type='encoder', 
    weights=[0.5, 0.5],
    output_path='./mixed_embedder')

# Mix reranker Models
model = mix_models(
    model_names_or_paths=["BAAI/bge-reranker-base", "BAAI/bge-reranker-base"], 
    model_type='reranker', 
    weights=[0.5, 0.5],
    output_path="./mixed_reranker")
```
Note that the sum of weights should be equal to 1.

You also can merge multiple models:
```python
from LLM_Merge.main import mix_models, mix_models_with_data

model = mix_models(
    model_names_or_paths=["BAAI/bge-base-en-v1.5", "bge-hotpotqa", "bge-quora", "bge-msmarco"], 
    model_type='encoder', 
    weights=[0.3, 0.2, 0.2, 0.3],
    output_path='./mixed_embedder_2')
# The sum of weights should be equal to 1.
```

### 2. Mix models with weights computed based on a few examples

`mix_models_with_data` can compute merging weights based on given data and merge models.
It can be used to produce a model for a new task without training, 
or boost the performance for the downstream task by leveraging the knowledge in others models.

- For LLMs

The format of `example_data` for LLMs is a list, where each item is a dict like:
```
{"input": str, "output": str}
```
LLM_Merge will compute the loss of the output. 

You can use the example data to merge models as following:

```python
from LLM_Merge.main import mix_models, mix_models_with_data

example_data = [
    {"input": "Question: when was the last time anyone was on the moon? Answer:\n", "output": "14 December 1972 UTC"},
    {"input": "Review: \"it 's a charming and often affecting journey . \" Is this movie review sentence negative or positive?\n", "output": "Positive"}
]

model = mix_models_with_data(
    model_names_or_paths=["meta-llama/Llama-3.2-1B", "llama3-ag-news", "llama3-nq"], 
    model_type='decoder', 
    example_data=example_data, 
    temperature=5.0)
# you can set the temperature argument to adjust the distribution of mixing weights
```


- For Embedder

The format of `example_data` for LLMs is a list, where each item is a dict like:
```
{"query": str, "pos": List[str], 'neg': List[str]}
```
where pos is a list of positive text and neg is a list of negative text. LLM_Merge will compute the contrastive loss. 

You can use the example data to merge models as following:
```python
from LLM_Merge.main import mix_models, mix_models_with_data

example_data = [
    {"query": "How does one become an actor in the Telugu Film Industry?", "pos": [" How do I become an actor in Telugu film industry?"], "neg": [" What is the story of Moses and Ramesses?", " Does caste system affect economic growth of India?"]}, 
    {"query": "Why do some computer programmers develop amazing software or new concepts, while some are stuck with basic programming work?", "pos": [" Why do some computer programmers develops amazing softwares or new concepts, while some are stuck with basics programming works?"], "neg": [" When visiting a friend, do you ever think about what would happen if you did something wildly inappropriate like punch them or destroy their furniture?", " What is the difference between a compliment and flirting?"]}
]

model = mix_models_with_data(
    model_names_or_paths=["BAAI/bge-base-en-v1.5", "bge-hotpotqa", "bge-quora"], 
    model_type='encoder', 
    example_data=example_data,
    temperature=5.0,
    max_input_length=512,
    neg_number=2)
```

### 3. Mix models layer by layer for reducing memory cost
The function `mix_models_by_layers` creates temporary directories to store weights of individual models and then merges them layer by layer.

This approach helps in reducing the memory consumption.

Once the merging process is completed, the temporary directories and files will be automatically removed.


```python
from LLM_Merge.main import mix_models_by_layers

# Mix Large Language Models (LLMs) and save the combined model to the path: ./mixed_llm
model = mix_models_by_layers(
    model_names_or_paths=["meta-llama/Llama-3.1-8b", "llama3.1-ag-news"], 
    model_type='decoder', 
    weights=[0.7, 0.3], 
    output_path='./mixed_llm')
```
