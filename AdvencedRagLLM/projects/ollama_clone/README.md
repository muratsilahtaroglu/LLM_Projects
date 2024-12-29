# Writing the README content to a file

readme_content = """
# CloneAI Platform

CloneAI is a simulation platform that mimics individuals' unique communication styles and provides insightful responses based on personal information, semantic search results, and context-driven AI generation.

## Features

- **Personalized AI Responses:** Simulates specific individuals based on personal data from `personal_info.txt`.
- **Semantic Search Integration:** Retrieves relevant documents (YouTube transcripts, PDFs, tweets) from a pre-built vector database for context-aware answers.
- **Dynamic Query Handling:** Differentiates between casual questions and queries requiring document-based information.
- **Multi-threading:** Uses parallel processing to improve response generation efficiency.
- **Customizable Deployment:** Includes a FastAPI backend and a Streamlit-based user interface.

---

## How It Works
1. User submits a query through the Streamlit UI or API.
2. CloneAI:
* Extracts the topic and task from the query.
* Identifies whether semantic search is needed.
* Fetches relevant content from a pre-built vector database if required.
* Combines personal information and content to simulate a response in the style of the specified individual.
* The response is delivered back to the user.
---

## Project Structure

```bash
.
├── clone_ai.py              # Core logic for CloneAI
├── clone_prompts.py         # Prompt templates for query handling and response generation
├── clone_ai_api.py          # FastAPI backend for serving CloneAI responses
├── streamlit_ui.py          # Streamlit-based user interface for querying CloneAI
├── personal_info.txt        # Text file containing personal information for AI simulation
├── topic_task_extractor_ai.py # Module for extracting topic and task from user queries
├── .env                     # Configuration file for environment variables

```
## Setup Instructions
1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Set Up Environment Variables
Create a .env file in the project root directory with the following configuration:

```bash
DATA_PATH=personal_info.txt
CLONE_NAME=CloneBot
MODEL_NAME=gemma2:27b
TOPIC=creating_clone_text
COUNT=2
PORT=8000
API_URL=http://127.0.0.1:8000/api/get_clone_response
APP_TITLE=CloneAI Query Interface
APP_SUBTITLE=Messaging with the CloneAI model
PLACEHOLDER_TEXT=Hello, I'm CloneAI. How can I help you?
SEND_BUTTON_TEXT=Send
SPINNER_TEXT=Please wait, this might take some time. Retrieving response from CloneAI...
WARNING_TEXT=Please enter a valid query.
ERROR_TEXT_TEMPLATE=An error occurred: {error}
RESPONSE_ERROR_TEXT_TEMPLATE=Error {status_code}: {error_detail}
```
4. Start the FastAPI Backend
```bash
python clone_ai_api.py
```
5. Start the Streamlit Frontend
```bash
streamlit run streamlit_ui.py
```

---


## Contributors

For further details or contributions: 

**Contact:** [muratsilahtaroglu13@gmail.com](mailto:muratsilahtaroglu13@gmail.com)

**Developed by:**  Murat Silahtaroğlu
