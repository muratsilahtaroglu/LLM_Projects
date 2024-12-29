# Semantic Search Client Library

This repository provides a Python-based library for semantic search operations. It includes utilities for handling queries, managing vector databases, and interacting with AI models for semantic similarity and information retrieval.

---

## Folder Structure
```bash
m_libs/
├── client/
├── logicalai/
├── semantic_search/
│   ├── client/
│   │   ├── vectordb/
│   │   ├── __init__.py
│   │   ├── base_utils.py
│   │   ├── logging.py
│   │   ├── semantic_search_client.py

```

---

## Features

### Core Functionalities

- **Semantic Search Client**: Provides utilities for querying a vector database and retrieving similar documents.
- **Query Management**: Tools for creating, managing, and merging queries with a focus on semantic relationships.
- **Vector Database Handling**: Includes classes and methods for managing vector databases, embeddings, and associated metadata.
- **Customizable Embedding Models**: Supports various embeddings, including HuggingFace and Sentence Transformers.
- **Multi-Method Search**: Implements multiple retrieval strategies like `merge_auto`, `sub_queries`, and more.

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Required libraries are listed in the code, including:
  - `pydantic`
  - `requests`
  - `fastapi`
  - `pandas`
  - `openpyxl`


## File Descriptions
### base_utils.py
- Contains utility classes for managing queries, responses, and configurations.
- Provides Pydantic models for structured validation.
- Includes helper methods for reading data files and cleaning float values.
### semantic_search_client.py
- Defines the SimilarityTextsClient class for interacting with a FastAPI backend.
- Provides methods for creating apps, managing queries, and retrieving similar documents.
- Supports multiple retrieval methods and query processing strategies.
### logging.py
- Placeholder for logging functionalities.



---


## Contributors

For further details or contributions: 

**Contact:** [muratsilahtaroglu13@gmail.com](mailto:muratsilahtaroglu13@gmail.com)

**Developed by:**  Murat Silahtaroğlu 
