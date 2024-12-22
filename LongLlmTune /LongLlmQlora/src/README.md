# Long Context Fine-Tuning and Long LLM QLoRA Project

This repository focuses on fine-tuning models for long-context tasks, enabling effective GPU utilization and multi-GPU training while reducing memory requirements. The repository is structured to streamline training, evaluation, and inference workflows for large-scale models with efficient memory management.

## Folder Structure

```
src/
├── args.py           # Handles argument parsing and model configuration.
├── chat.py           # Implements chat templates and conversation utilities.
├── data.py           # Data preparation and preprocessing utilities.
├── __init__.py       # Initialization file for the src module.
├── metrics.py        # Metrics computation and evaluation.
├── modeling_utils.py # Utilities for model evaluation and generation.
├── trainer.py        # Custom Trainer for efficient model training.
├── utils.py          # General utilities for file handling, logging, and padding.
```

## File Explanations

#### `args.py`
Defines configurations and arguments for the model, training, and evaluation processes. Key features include:
- GPU memory optimization settings like LoRA and 4-bit loading.
- Parameters for managing long-context fine-tuning such as `max_length`, `beacon_window`, and `retrieval_method`.
- Device-specific settings for single or multi-GPU training.

#### `chat.py`
Implements conversation templates and message handling. Key features include:
- Tokenization and formatting for various model architectures.
- Efficient handling of nested conversations and padding for GPU processing.
- Attention to memory management during message encoding.

#### `data.py`
Handles data preparation for training and evaluation. Key features include:
- Tokenization and chunking of long text sequences to fit GPU memory.
- Sliding window and overlapping techniques to manage long contexts.
- Parallel processing for efficient dataset preparation.

#### `metrics.py`
Computes evaluation metrics such as ROUGE and saves results. Key features include:
- Normalization of predictions and labels.
- Support for various evaluation configurations.
- Integration with fine-tuning to log GPU-efficient metrics.

#### `modeling_utils.py`
Provides utilities for model operations like gradient checkpointing and LoRA integration. Key features include:
- Optimized memory management during forward and backward passes.
- Supports dynamic beacon tokens to handle long contexts.
- Flexible for multi-GPU setups with mixed precision.

#### `trainer.py`
Extends the Hugging Face `Trainer` class for custom fine-tuning. Key features include:
- GPU memory-efficient input preparation.
- Automatic memory resetting between batches.
- Support for perplexity and generation-based evaluation.

#### `utils.py`
Utility functions for file handling, logging, and padding. Key features include:
- Efficient batching and nested list padding for long sequences.
- Functions to mix model parameters across checkpoints.
- Tools to normalize text and manage data logging.

**Key Features:**
- Padding utilities to manage nested sequences efficiently.
- Optional gradient context manager for memory optimization.


## Key GPU and Memory Optimization Techniques

1. **Gradient Checkpointing**:
   - Reduces memory usage by storing only intermediate activations.

2. **LoRA (Low-Rank Adaptation)**:
   - Fine-tunes specific model layers, reducing the number of trainable parameters.

3. **Memory Resetting**:
   - Frees up memory between batches during training and evaluation.

4. **Efficient Tokenization**:
   - Chunking and padding ensure that sequences fit within GPU memory limits.

5. **Mixed Precision Training**:
   - Uses `fp16` and `bf16` for faster computations and lower memory usage.

6. **Multi-GPU Support**:
   - Distributes computations across multiple GPUs for faster training and larger batch sizes.

