# Pre-Processing Data with AI

This project provides tools for parsing and processing various types of data, including PDFs, tweets, and YouTube transcripts, using advanced AI models like **phi-4**, **gemma2**, and **llama**. Built on the Ollama platform, it ensures ease of use and flexibility, with B-models quantized to levels such as 4-bit or 8-bit based on specific needs. This quantization approach significantly reduces GPU memory usage while maintaining high performance, making the framework both efficient and scalable for diverse data parsing tasks dynamically tailored to user inputs.


## Features

- **PDF Parsing**: Extract and process text from PDF documents for AI-based insights.  
- **Tweet Analysis**: Clean and extract valuable information from tweet datasets.  
- **YouTube Transcripts**: Parse and analyze video transcripts for content processing.  
- **Dynamic AI Integration**: Supports AI models like gemma2 and llama with customizable prompt templates.  
- **Modular Design**: Reusable components allow for easy extension and integration.  

---

## Folder Structure

```bash
llm_pre_processing/
├── main.py                # Entry point for running parsing tasks.
├── parse_pdf_ai.py        # Contains logic for parsing PDF documents.
├── parse_tweet_ai.py      # Processes tweets and extracts relevant data.
├── parse_youtube_ai.py    # Handles parsing of YouTube transcripts.
├── parse_prompts.py       # Stores customizable prompts for AI interactions.
├── requirements.txt       # List of Python dependencies for the project.
├── logs/                  # (Optional) Directory for storing log files.
├── data/                  # Directory for storing input data.
└── edited_data/           # Directory for storing edited data.
```
## Requirements
Python Version: Ensure Python 3.10 or higher is installed.
Dependencies: Install required libraries using the instructions below.


## Clone the repository:

```bash
git clone <repository-url>
cd llm_pre_processing
```

## Install dependencies:

```bash
pip install -r requirements.txt
```
Prepare your data files in the data/ directory.

Usage
Command-Line Interface (CLI)
Run main.py with appropriate arguments to specify the task type and parameters.

## Example Commands
### PDF Parsing
```bash
python main.py --task pdf \
               --model_name "gemma2:27b" \
               --data_path "data/demo_pdf.json" \
               --output_path "edited_data/parsed_pdf_data.json" \
               --clone_name "Demo" \
               --topic "parsing_pdf_data" \
               --count 2
```

### Tweet Parsing
```bash
python main.py --task tweet \
               --model_name "phi-4:14b" \
               --data_path "data/demo_tweet.json" \
               --output_path "edited_data/parsed_tweet_data.json" \
               --clone_name "Demo" \
               --topic "parsing_tweet_data" \
               --count 1
```
### YouTube Transcript Parsing

```bash
python main.py --task youtube \
               --model_name "llama3.3:70b" \
               --data_path "data/demo_youtube.json" \
               --output_path "edited_data/parsed_youtube_data.json" \
               --clone_name "Demo" \
               --topic "parsing_youtube_data" \
               --count 2
```

## Arguments
```bash
--task	        The task to perform: pdf, tweet, or youtube.	
--model_name	You can utilize any AI model on your local machine via the Ollama platform, such as gemma2:27b or llama3.3:70b or phi-4:14b
--data_path	    Path to the input data file.	
--output_path	Path to save the output results.	
--clone_name	Clone name for prompts.	
--topic	        Topic for the parsing task.	
--count	        Number of iterations for parsing.	
--repeat_count	Number of times to repeat the task (default: 1).	
```

## Contributors

For further details or contributions: 

Contact: [muratsilahtaroglu13@gmail.com](mailto:muratsilahtaroglu13@gmail.com)

Developed by:  Murat Silahtaroğlu