## Client Library: Ollama Client
This repository provides a Python-based client library for working with AI models using the **Ollama API**. The library simplifies interactions with Ollama by providing utilities for generating responses, checking response compatibility, and parsing various data formats. It also **supports multithreading** for efficient handling of multiple requests.

## Folder Structure
```bash
m_libs/
├── client/
│   ├── ollama_client/
│   │   ├── __init__.py
│   │   ├── ollama_client.py
│   ├── setup.py
```



### File Descriptions

- **`__init__.py`**: Marks the `ollama_client` directory as a Python module.
- **`ollama_client.py`**: Contains the core functionality of the Ollama client, including AI response generation, response validation, and utility methods for content parsing.
- **`setup.py`**: Script to configure the library as a Python package with required dependencies.

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Install the necessary dependencies defined in `setup.py`.



## Dependencies
The library requires the following Python packages:
```bash
pandas
ollama
rouge_score
sentence_transformers
```
These dependencies will be installed automatically when running pip install ..


## Features
### Ollama Client
**AI Response Generation:**
Generates responses from specified AI models using the ollama client. Supports both streaming and non-streaming modes.

**Response Validation:**
Provides utilities to validate and parse AI responses, ensuring compatibility with input data.

**Multi-Threaded Processing:**
Supports handling multiple requests concurrently using Python threading.

**Content Scoring:**
Computes ROUGE scores and cosine similarity between AI responses and input content.



---


## Contributors

For further details or contributions: 

**Contact:** [muratsilahtaroglu13@gmail.com](mailto:muratsilahtaroglu13@gmail.com)

**Developed by:**  Murat Silahtaroğlu 
