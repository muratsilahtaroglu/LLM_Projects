# ModelTrainingWorkspace

The `ModelTrainingWorkspace` serves as the core directory for managing and executing fine-tuning tasks. It is part of the **NoCodeTune** system, designed to simplify the model fine-tuning process by providing an organized, modular structure. This directory integrates tools and scripts for data preparation, job management, and model training, forming a robust workspace for machine learning workflows.

---

## Main Features

- **Modular Design**: Clearly structured directories and scripts for easy navigation.
- **Fine-Tuning Framework**: Tools for training and tuning AI models with or without additional user coding.
- **Dynamic Job Management**: Support for creating and running jobs dynamically through the FineTuneFlow interface.
- **Logging and Monitoring**: Detailed logs for debugging and tracking training progress.
- **Server Integration**: Pre-configured scripts for managing data and fine-tuning servers.

## Features
- **Model Support**: Supports LLaMA, Gemma, and T5 for fine-tuning.
- **Multi-GPU Training**: Scale up training by specifying gpu_count.
- **Fine-Tuning**: Easy-to-use tools for full and LoRA-based fine-tuning.
- **NoCode Integration**: Configure workflows without extensive coding.
---

## Directory Structure

```plaintext
ModelTrainingWorkspace/
├── archive/                 # Stores reusable scripts and utilities.
│   ├── data_load/           # Contains utilities for data handling.
│   │   ├── data_load_server.py  # FastAPI server for dataset management.
│   │   └── file_operations_utils.py # Utilities for file operations.
│   ├── full_train/          # Base job structure for inheritance.
│   │   ├── files/           # Input files specific to the base jobs.
│   │   ├── fine_tuning/     # Fine-tuning utilities and scripts.
│   │   ├── main.py          # Main entry point for base job execution.
│   │   ├── outputs/         # Outputs generated by base jobs.
│   │   ├── server_utils.py  # Server management utilities.
│   │   └── validate.py      # Configuration validation scripts.
│   ├── ai_server.sh         # Script for managing AI servers.
│   └── check_requirements.py # Dependency verification script.
├── files/                   # Shared data files.
├── jobs/                    # Houses all fine-tuning job configurations.
│   ├── demo_job/            # Example job for demonstration purposes.
│   └── FineTuneFlow Jobs/   # Jobs dynamically generated by FineTuneFlow.
├── logs/                    # Logs for debugging and tracking.
```

---

## Detailed Folder Overview

### `archive/`

This directory contains reusable components and scripts for model training and fine-tuning tasks.

- **`data_load/`**:
  - `data_load_server.py`: FastAPI server for dataset upload and retrieval.
  - `file_operations_utils.py`: Utilities for operations like uploading, deleting, and reading files.

- **`full_train/`**:
  - `files/`: Input data for base jobs.
  - `fine_tuning/`: Scripts for fine-tuning models.
  - `main.py`: Entry script for running base jobs.
  - `outputs/`: Directory for storing model outputs.
  - `server_utils.py`: Server management utilities for job-specific tasks.
  - `validate.py`: Validation scripts for ensuring proper configurations.

- **`ai_server.sh`**: Script to set up and run the AI server with dependency checks.
- **`check_requirements.py`**: Verifies that all necessary dependencies are installed.

### `files/`

- **Purpose**: Shared data files accessible across multiple jobs.
- **Examples**: CSV files, datasets, or tokenized outputs.

### `jobs/`

- **Purpose**: Contains all job configurations for fine-tuning.
- **Contents**:
  - **`demo_job/`**: Example job for understanding the workflow.
  - **`FineTuneFlow Jobs/`**: Fine-tuning jobs dynamically created via the FineTuneFlow system.

### `logs/`

- **Purpose**: Logs generated during training and server operations for debugging and tracking.
- **Use Case**: Monitor system performance and troubleshoot issues.

---

## Workflow

### Step 1: Data Preparation
- Use `data_load_server.py` to upload and manage datasets.
- Validate uploaded data using the utilities in `file_operations_utils.py`.

### Step 2: Job Creation
- Create fine-tuning jobs dynamically via the **FineTuneFlow** interface.
- Jobs are saved in the `jobs/` directory for execution.

### Step 3: Model Training
- Execute training tasks using `main.py` in the `full_train/` directory.
- Monitor outputs in the `outputs/` directory.

### Step 4: Logging and Debugging
- Access detailed logs in the `logs/` directory.
- Use logs to optimize workflows and resolve issues.

### Step 5: Model Deployment
- Deploy trained models using pre-configured FastAPI servers.
- Integrate the models into production pipelines or use for further fine-tuning.

---

## Contribution Guidelines

- **Code Standards**: Follow PEP8 and include detailed docstrings.
- **Documentation**: Ensure proper documentation for any added components.
- **Testing**: Validate changes in a local environment before submission.
- **Pull Requests**: Submit PRs to the **NoCodeTune** repository with a clear description of changes.

---

## Contact
For further details or contributions:
- **Contact**: [Email](mailto:contact@nocodetune.com)
- **Contributors**: Murat Silahtaroğlu and the NoCodeTune Community
