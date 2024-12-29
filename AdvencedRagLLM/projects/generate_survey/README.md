# Generate Survey Data

## Description
The `Generate Survey Data` project is designed to automate the generation of survey question answers using various AI models. It processes user-specific data files in parallel using multithreading, integrates semantic similarity search for contextual understanding, and supports advanced AI models like GPT, Gemini, and Ollama.

---

## Features
- **Semantic Similarity Search**: Retrieves relevant contextual data for accurate and insightful responses.
- **Multithreading**: Efficiently processes multiple users concurrently.
- **AI Model Integration**: Supports GPT, Gemini, and Ollama AI models for flexibility and accuracy.
- **Configurable and Extensible**: Easily adapts to new datasets, configurations, or AI models.
- **Detailed Logging**: Tracks progress and errors for debugging and auditing.

---
## Directory Structure
```bash
generate_survey_data/
├── datasets/
│   └── dataset.xlsx                  # Main dataset file
├── generate_survey/
│   ├── v2/                           # Folder for user-specific data files
│   ├── vectordb/                     # Semantic vector database directory
├── survey_answers/
│   ├── V2/                           # Folder for generated survey results
├── logs/
│   └── generate_qa_v2.out            # Log file for progress and error tracking
├── ai_utils.py                       # Utility functions for API integration and logging
├── generate_qa.py                    # Core logic for survey question processing
├── survey_main.py                    # Main script for running the survey generation
├── survey_semantic_search_creating.py # Handles semantic search operations
├── _predictors.json                  # Configuration for semantic search and embeddings
├── requirements.txt                  # List of required Python packages
└── README.md                         # Project documentation
```

---


## Setup and Installation

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd generate_survey_data
```
### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv generate_survey_data_venv
source generate_survey_data_venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
### Step 4: Run the Main Script

```bash
nohup python3 survey_main.py >> logs/generate_qa.out 2>&1 &
```