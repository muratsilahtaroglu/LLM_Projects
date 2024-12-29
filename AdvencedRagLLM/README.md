# AdvancedRagLLM

## Project Overview

**AdvancedRagLLM** is a comprehensive framework designed to build, optimize, and benchmark Retrieval-Augmented Generation (RAG) systems. It brings together advanced semantic search techniques, AI-driven data processing, hybrid ranking mechanisms, and robust benchmarking tools. The project is tailored to handle various applications, including personalized AI response systems, survey generation, and more.

This framework is modular and scalable, enabling use cases across industries such as healthcare, tourism, education, and e-commerce.

---

## Sub-Projects Overview

### 1. **Semantic Search**
Implements state-of-the-art semantic search techniques using embedding models and vector databases. It includes advanced query decomposition and multi-method search for context-sensitive information retrieval.

### 2. **Ollama Client Library**
Provides a Python client library for efficient interaction with AI models through the Ollama API. It simplifies AI response generation and validation while supporting multi-threaded operations for handling large-scale queries.

### 3. **Hybrid Re-Ranker Framework**
Introduces a robust re-ranking system for semantic search, enabling contextual query decomposition, multi-stage ranking, and enhanced document filtering for RAG pipelines.

### 4. **LLM Pre-Processing**
Handles the parsing and processing of diverse datasets such as PDFs, tweets, and YouTube transcripts. It integrates advanced AI models for structuring and organizing data dynamically for downstream tasks.

### 5. **RAG Benchmark Analysis**
Provides tools for evaluating and optimizing RAG systems through clustering, ablation experiments, and benchmarking with metrics like nDCG, precision, and recall.

## Example Projects
### 1. **CloneAI Platform**
Simulates personalized AI responses based on user-specific data and semantic search results. CloneAI mimics unique communication styles for context-aware interactions.

**Example Use Case**:  
A user queries, "How would John respond to this situation?" The system retrieves relevant data, analyzes the query context, and generates a response in John’s communication style.

### 2. **Generate Survey Data**
Automates the generation of survey answers using various AI models. It uses semantic similarity search for context, multithreading for efficiency, and advanced AI models like GPT, Gemini, and Ollama for accuracy.

**Example Use Case**:  
Given a dataset of survey questions, the system retrieves relevant contextual information, processes the data, and generates accurate, context-aware responses in bulk.

---

## Workflow Overview

1. **Data Pre-Processing**
   - Extract and organize data from various sources (e.g., PDFs, tweets, YouTube transcripts).
   - Generate embeddings and prepare structured content for semantic search.

2. **Semantic Search and Query Processing**
   - Execute semantic search queries using advanced embedding models.
   - Optimize queries by decomposing them into sub-queries and extracting keywords.

3. **Hybrid Re-Ranking**
   - Apply multi-stage ranking techniques to refine search results.
   - Use advanced merging strategies (e.g., sum, product, square) for context-sensitive filtering.

4. **Retrieval-Augmented Generation (RAG)**
   - Feed re-ranked search results into RAG pipelines to generate accurate, domain-specific answers.
   - Optimize LLMs to handle noisy datasets and short/long query contexts.

5. **Benchmarking and Evaluation**
   - Use tools for clustering, ablation experiments, and benchmarking to evaluate system performance.
   - Generate metrics like cosine similarity, precision, recall, and nDCG.

---

## Example Use Cases

1. **CloneAI Personalized Responses**
   - A user interacts with CloneAI to simulate a specific individual's communication style based on personal data and contextual search.
   - **Example Query**: "How would Sarah respond to a career change?"  
     - **Output**: "Based on her style, Sarah would say, 'Take the leap if it aligns with your goals and values.'"

2. **Generate Survey Data**
   - Automates the creation of survey responses by analyzing contextual data and integrating with advanced AI models.
   - **Example Input**: "How do employees feel about remote work?"  
     - **Output**: Context-aware survey responses reflecting user sentiments and patterns.

3. **Hybrid Re-Ranker for Context-Aware Retrieval**
   - A healthcare professional queries: "Summarize recent research on diabetes management."  
   - **Output**: Re-ranked documents based on query decomposition, yielding concise, accurate results for the LLM to process.

4. **RAG Benchmark Analysis**
   - Evaluate and optimize RAG system performance for a dataset containing customer support queries.  
   - **Output**: Benchmark results highlighting optimal configurations for precision and recall.

---

## Contributors

**Developed by:** Murat Silahtaroğlu  
**Contact:** [muratsilahtaroglu13@gmail.com](mailto:muratsilahtaroglu13@gmail.com)  

For inquiries or contributions, feel free to reach out.
