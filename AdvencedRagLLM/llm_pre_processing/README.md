# Pre-Processing Data with AI

This project provides tools for parsing and processing various types of data, including PDFs, tweets, and YouTube transcripts, using advanced AI models like **gemma2** and **llama**. Designed with modularity and scalability in mind, this framework dynamically handles diverse data parsing tasks based on user input, enabling efficient data extraction and cleaning.

---

## Features

- **PDF Parsing**: Extract and process text from PDF documents for AI-based insights.  
- **Tweet Analysis**: Clean and extract valuable information from tweet datasets.  
- **YouTube Transcripts**: Parse and analyze video transcripts for content processing.  
- **Dynamic AI Integration**: Supports AI models like gemma2 and llama with customizable prompt templates.  
- **Modular Design**: Reusable components allow for easy extension and integration.  

---

## Folder Structure

```bash
pre_processing_data/
├── main.py                # Entry point for running parsing tasks.
├── parse_pdf_ai.py        # Contains logic for parsing PDF documents.
├── parse_tweet_ai.py      # Processes tweets and extracts relevant data.
├── parse_youtube_ai.py    # Handles parsing of YouTube transcripts.
├── parse_prompts.py       # Stores customizable prompts for AI interactions.
├── requirements.txt       # List of Python dependencies for the project.
├── logs/                  # (Optional) Directory for storing log files.
Requirements
Python Version: Ensure Python 3.7+ is installed.
Dependencies: Install required libraries using the instructions below.
Installation
Clone the repository:

bash
Kodu kopyala
git clone <repository-url>
cd pre_processing_data
Install dependencies:

bash
Kodu kopyala
pip install -r requirements.txt
Prepare your data files in the data/ directory.

Usage
Command-Line Interface (CLI)
Run main.py with appropriate arguments to specify the task type and parameters.

Example Commands
PDF Parsing
bash
Kodu kopyala
python main.py --task pdf \
               --model_name "gemma2:27b" \
               --data_path "data/demo_pdf.json" \
               --output_path "edited_data/parsed_pdf_data.json" \
               --clone_name "Demo" \
               --topic "parsing_pdf_data" \
               --count 2
Tweet Parsing
bash
Kodu kopyala
python main.py --task tweet \
               --model_name "gemma2:27b" \
               --data_path "data/demo_tweet.json" \
               --output_path "edited_data/parsed_tweet_data.json" \
               --clone_name "Demo" \
               --topic "parsing_tweet_data" \
               --count 1
YouTube Transcript Parsing
bash
Kodu kopyala
```bash
python main.py --task youtube \
               --model_name "llama3.1:70b" \
               --data_path "data/demo_youtube.json" \
               --output_path "edited_data/parsed_youtube_data.json" \
               --clone_name "Demo" \
               --topic "parsing_youtube_data" \
               --count 2
```
Arguments
Argument	Description	Required
```bash
--task	The task to perform: pdf, tweet, or youtube.	Yes
--model_name	AI model to use, e.g., gemma2:27b, llama3.1.	Yes
--data_path	Path to the input data file.	Yes
--output_path	Path to save the output results.	Yes
--clone_name	Clone name for prompts.	Yes
--topic	Topic for the parsing task.	Yes
--count	Number of iterations for parsing.	No
--repeat_count	Number of times to repeat the task (default: 1).	No
```
Logs
Logs are stored in the logs/ directory (if implemented) for debugging and tracking progress.
Includes details of task execution, errors, and performance.
Future Improvements
Support for More Data Formats: Add parsing capabilities for CSV, JSONL, and other formats.
Advanced AI Models: Integrate new AI models to improve accuracy and expand capabilities.
GUI Integration: Develop a graphical user interface for non-technical users.
Asynchronous Processing: Implement faster processing through asynchronous tasks.
Long-Context Handling: Introduce frameworks for processing documents with 100k+ tokens.
QLoRA Fine-Tuning: Include efficient fine-tuning workflows for quantized LoRA-based training.
Real-Time Monitoring: Add dashboards for live tracking of parsing tasks and model performance.
Contribution Guidelines
Ensure all contributions follow PEP8 standards.
Add meaningful comments and documentation for new components.
Validate changes locally before submitting pull requests.
Open issues for any bugs or feature suggestions.
Contact
For questions or contributions, reach out to:
Email: contact@preprocessingai.com

Developed by: [Your Name]
Part of the Pre-Processing AI Tools Project