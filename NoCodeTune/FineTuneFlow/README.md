# FineTuneFlow: Datatune Flow and Finetuning Service

FineTuneFlow is a project designed to streamline the process of managing datasets, fine-tuning machine learning models, and creating reproducible workflows. It consists of several components that work together to handle data loading, model management, and process execution.

## Folder Structure

```bash
FineTuneFlow/
├── __monkey/
│   ├── config.yaml      # YAML configuration file defining scripts, parameters, and other configurations used by `monkey.py`.
│   ├── monkey.py        # A CLI tool that interprets and executes scripts defined in `config.yaml` using Jinja templates and subprocesses.
│   └── output.json      # JSON file generated as output by executing defined scripts, used to store results or logs.
├── data_load/
│   ├── crud.py          # Contains CRUD operations for managing datasets, including retrieving and updating dataset versions.
│   ├── file_operations_utils.py  # Utility functions for handling various file formats like Excel, JSON, PDF, YAML, and more. Includes parsers for reading and writing files.
│   └── server.py        # Implements a FastAPI-based server providing endpoints for dataset operations like upload, retrieval, and version management.
├── fine_tune_ops/
│   ├── crud_io.py       # Handles file and job management, including creating, copying, and validating job configurations.
│   ├── crud.py          # CRUD operations for managing fine-tuning jobs, processes, and related metadata.
│   ├── process_management.py  # Manages background processes related to fine-tuning tasks, including monitoring and handling task states.
│   └── server.py        # FastAPI server providing APIs for managing fine-tuning workflows, job creation, and process execution.
├── static_files/
│   └── server_utils.py.jinja  # A Jinja template for dynamically generating utility Python scripts based on job-specific configurations.
├── workspace.code-workspace.json  # VS Code workspace configuration file, enabling easy project navigation and setup.
├── README.md            # Documentation file explaining the project, setup instructions, and usage.
├── requirements.txt     # Lists all Python dependencies required to run the project, including libraries for FastAPI, SQLAlchemy, and others.
├── logs.py              # Logging utility module providing structured logging for tracking application events and errors.
├── schemas.py           # Defines Pydantic models and schemas for validating and serializing data for API endpoints.
├── settings.py          # Configuration settings for the project, including paths, database connection strings, and directory structure.
└── entities.py          # Database entity definitions using SQLAlchemy, representing tables and relationships for datasets, jobs, and processes.
```

## Key Features

- **Data Load Management**: APIs for managing datasets including CRUD operations, file uploads, and versioning.
- **Fine-Tuning Management**: APIs for creating and running fine-tuning jobs, tracking processes, and managing dependencies.
- **Process Automation**: YAML-configured scripts and CLI tools to execute defined workflows.
- **Extensive File Support**: Utilities for parsing various file formats including Excel, csv, parquet, PDF, JSON, and more. This will support more file types in the future.

## Installation

### Prerequisites
- Python 3.10 or later is recommended.
- Ensure you have `virtualenv` installed:

```bash
pip install virtualenv
```

### Steps
1. Clone this repository to a local directory:
```bash
git clone <repository_url>
cd FineTuneFlow
```
2. Create a virtual environment and activate it:
```bash
virtualenv venv
source venv/bin/activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Use cu118, cu121, or cu124 based on your GPU type and the compatible PyTorch version. This affects transformers and GPU compatibility.
```

## Running the Services

### Activating the Virtual Environment
```bash
source venv/bin/activate
```

## Starting the Servers
### 1.Running via `uvicorn`
#### Data Load Server
```bash
uvicorn data_load.server:app --host 127.0.0.1 --port port_id1
```

#### Fine-Tuning Management Server
```bash
uvicorn fine_tune_ops.server:app --host 127.0.0.1 --port port_id2
```

### 2.Running via `ai_server.sh`
Set environment variables and run the script:
create your own ai_server.sh
```bash
export SERVER_TYPE=fine_tuning_server NOHUP=False && ./ai_server.sh
```

Example for Data Load Server:
```bash
export SERVER_TYPE=data_load_server NOHUP=False && ./ai_server.sh
```

#### Exports

- **`NOHUP`**: Set to `True` or `False` for background process handling.
- **`SERVER_TYPE`**: Define the type of server (`data_load_server`, `fine_tuning_server`, or `model_release_server`).

## Deployment

Access the services via:
- **Web Interface**: `http://<host>:<port>/docs`
- **API Endpoints**:
    - Data Load API: `http://<host>:<port>/files/` (GET, POST, DELETE)
    - Fine-Tuning API: `http://<host>:<port>/jobs/`

Example body for POST requests:
```json
{
  "path": "/home/user/FineTuneFlow/requirements.txt"
}
```

## Notable Files

### `config.yaml`
Defines scripts and configurations used in `monkey.py`. Example:
- Define CLI commands using Jinja templates.
- Configure script parameters for automating tasks.

### `file_operations_utils.py`
Contains utilities for reading and processing various file formats such as PDF, Excel, and JSON.

### `process_management.py`
Manages background processes using threading and `psutil`. Handles task execution and logging.

### `server.py`
Provides RESTful APIs for data loading and fine-tuning management using FastAPI.

### `schemas.py`
Defines Pydantic models for request/response validation.

### `entities.py`
Database models for jobs, datasets, and their versions. Uses SQLAlchemy for ORM.

---
For more details, explore the code and inline documentation.

## Contributors
Developed by: Murat Silahtaroğlu  
Contact: [muratsilahtaroglu13@gmail.com](mailto:muratsilahtaroglu13@gmail.com)
