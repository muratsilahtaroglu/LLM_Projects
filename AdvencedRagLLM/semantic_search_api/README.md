# Advanced Hybrid Re-Ranker Framework for Semantic Search

---

## Introduction

In modern **semantic search systems**, embedding models are extensively used to optimize query and document matching. However, these models face inherent challenges such as:

- Expression variability.
- Contextual difficulties with short or long queries.
- Noise from irrelevant information.

This project addresses these issues by introducing an innovative **Hybrid Re-Ranker System** for semantic search. The framework enhances the performance of **Large Language Model (LLM)**-based **Retrieval-Augmented Generation (RAG)** systems, offering a reliable, scalable, and context-sensitive infrastructure.

---

## Key Features

### Hybrid Re-Ranker Methodology

The framework uses a multi-stage **re-ranking mechanism** to:
- Contextualize queries.
- Decompose them into sub-queries and keywords.
- Assign weights to query components.
- Re-rank documents using advanced merging strategies (e.g., `sum`, `product`, `square`).

This effectively filters noise and delivers high-quality, context-relevant information to LLMs.

### Core Capabilities

1. **Sub-query Decomposition**:
   - Splits complex or composite queries into sub-queries.
   - Extracts key and helper keywords for optimized search.

2. **Multi-Stage Re-Ranking**:
   - Combines document scores using advanced strategies (`sum`, `proud`, `square`, `auto`).

3. **Retrieval-Augmented Generation (RAG)**:
   - Integrates with RAG pipelines for enhancing search and generation workflows.

4. **Scalable and Domain-Specific Vector Databases**:
   - Language-specific and domain-specific embeddings.
   - Fine-grained control over search precision for industries like healthcare, tourism, and e-commerce.

---

## Folder Structure

```bash
project/
├── __main__.py                  # Main entry point for the FastAPI application.
├── advanced_semantic_search_method.py  # Core semantic search and re-ranking logic.
├── schemas.py                   # Data validation and API schema definitions.
├── shared_utils.py              # Utility functions and shared state.
├── logs.py                      # Custom logging system for application events.
├── data/
│   ├── demo_data.csv            # Sample dataset for semantic search.
│   ├── demo_predictors.json     # Pre-configured demo predictors.
│   └── all_query_and_relations.json # Sample queries and relations data.
├── vectordb/                    # Directory for vector database storage.
└── README.md                    # Project documentation.
```

## Objectives of the Framework

This project aims to address the following goals:

### Context-Aware Query Optimization:
- Break down user queries into smaller, manageable sub-queries.
- Generate helper keywords for improved document retrieval.

### Enhanced Retrieval Accuracy:
- Address noisy datasets by focusing on embedding-based re-ranking.

### Robust RAG Systems:
- Provide accurate, context-sensitive information to LLM-based systems.

### Scalable Solutions for Multiple Industries:
- Support applications in healthcare, tourism, e-commerce, education, etc.

---

## Workflow

### 1. Data Preparation and Structuring

#### Sources
- **Text-Based Data**: PDFs, articles, books, transcriptions.
- **Visual Data**: Tables and figures converted to text using vision models (e.g., LLava).

#### Steps
1. Extract content from diverse sources using tools like LangChain.
2. Organize data into hierarchical JSON structures:
   - **Title and Subtitles**: Metadata for document organization.
   - **Paragraphs and Sentences**: Smaller units for embedding.
   - **Summaries and Topics**: Extracted key points.
   - **Metadata**: Source details (e.g., page, paragraph numbers).
3. Prepare the data for embeddings.

---

### 2. Embedding and Vector Database Creation

#### Steps
1. Use contextual embedding models (e.g., Hugging Face, OpenAI) to generate vector representations.
2. Store vectors in databases optimized for:
   - **Language-Specific Data**: Separate databases for each language.
   - **Domain-Specific Data**: Databases for specific fields like healthcare or tourism.

---

### 3. Query Processing and Optimization

#### Steps
1. **Decompose Query**:
   - Main query, sub-queries, and keywords.
2. **Embed Queries**:
   - Generate embeddings for efficient matching.

#### Example

**Query**:
> "Plan my trip from Berlin to Istanbul on 10/11/2024 and suggest tickets and travel plans."

**Decomposed Output**:
```json
{
    "query": "Plan my trip from Berlin to Istanbul on 10/11/2024 and suggest tickets and travel plans.",
    "sub_queries": ["Find flights from Berlin to Istanbul", "Suggest travel plans in Istanbul"],
    "keywords": ["Berlin", "Istanbul", "travel plans"],
    "helper_keywords": ["flights Berlin Istanbul", "travel in Istanbul"]
}
```
---

# Re-Ranking Mechanism

## Steps
1. **Retrieve documents for**:
   - Main query.
   - Sub-queries.
   - Keywords and helper keywords.
   
2. **Merge and re-rank using methods like**:
   - **Sum**: Combine scores from different queries.
   - **Proud**: Multiply scores for higher precision.
   - **Square**: Prioritize dominant scores.

3. **Filter and prioritize**:
   - Identify and present relevant results for LLM consumption.

---

## Re-Ranker Strategies

| **Method** | **Description**                |
|------------|--------------------------------|
| **Sum**    | Aggregate scores.             |
| **Proud**  | Multiply scores.              |
| **Square** | Square-root based merging.    |
| **Auto**   | Adaptive strategy using all methods. |

---

# API Endpoints

## `/semantic_search/create_app`
- **Description**: Create a new semantic search application.
- **Inputs**:
  - `embedding_type`, `llm_name`, `vectordb_directory`.
- **Outputs**:
  - App token, collection name, vector DB directory.

## `/semantic_search/method/get_main_query_results`
- **Description**: Retrieve results for the main query.
- **Inputs**:
  - Query, threshold, vector DB.
- **Outputs**:
  - Ranked documents.

## `/semantic_search/method/get_merge_query_results`
- **Description**: Retrieve re-ranked results using multiple strategies.
- **Inputs**:
  - Sub-queries, keywords, helper keywords.
- **Outputs**:
  - Optimized documents.

---

# Installation

## Prerequisites
- **Python 3.8+**
- **GPU (optional)** for faster embedding and inference.

## Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/repo/semantic-search.git
   cd semantic-search

   ```
2. **Install dependencies:**

```bash
pip install -r requirements.txt
```