
import os
import argparse
from parse_pdf_ai import UpdatePDFText
from parse_tweet_ai import UpdateTweetText
from parse_youtube_ai import UpdateYoutubeTranscript

def main():
    # Setting up argparse for command-line arguments
    parser = argparse.ArgumentParser(description="Data Parsing Script")
    parser.add_argument('--task', required=True, choices=['pdf', 'tweet', 'youtube'], help="Task to perform: pdf, tweet, youtube")
    parser.add_argument('--model_name', required=True, help="Name of the AI model to use (e.g., gemma2:27b, llama3.3)")
    parser.add_argument('--data_path', required=True, help="Path to the input data file")
    parser.add_argument('--output_path', required=True, help="Path to the output file where results will be saved")
    parser.add_argument('--clone_name', required=True, help="Clone name used for prompts")
    parser.add_argument('--repeat_count', type=int, default=1, help="Number of times to repeat the task (default: 1)")
    parser.add_argument('--topic', required=True, help="Topic for parsing task")
    parser.add_argument('--count', type=int, default=1, help="Number of iterations for parsing (default: 1)")
    
    args = parser.parse_args()
    
    # Get the task and parameters
    task = args.task
    model_name = args.model_name
    data_path = args.data_path
    output_path = args.output_path
    clone_name = args.clone_name
    repeat_count = args.repeat_count
    topic = args.topic
    count = args.count
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    if task == "pdf":
        print("Starting PDF parsing...")
        update_pdf_text = UpdatePDFText(data_path, clone_name, repeat_count)
        parsed_data = update_pdf_text.get_parsing_pdf_data(model_name=model_name, topic=topic, count=count)
        update_pdf_text.save_pdf_text(parsed_data, output_path)
        print(f"PDF parsing completed. Results saved to {output_path}")
    
    elif task == "tweet":
        print("Starting Tweet parsing...")
        update_tweet_text = UpdateTweetText(data_path, clone_name, repeat_count)
        parsed_data = update_tweet_text.get_parsing_tweet_data(model_name=model_name, topic=topic, count=count)
        update_tweet_text.save_tweet_text(parsed_data, output_path)
        print(f"Tweet parsing completed. Results saved to {output_path}")
        
    elif task == "youtube":
        print("Starting YouTube transcript parsing...")
        updata_youtube_transcript = UpdateYoutubeTranscript(data_path, clone_name, repeat_count)
        parsed_data = updata_youtube_transcript.get_parsing_youtube_data(model_name=model_name, topic=topic, count=count)
        updata_youtube_transcript.save_youtube_transcript(parsed_data, output_path)
        print(f"YouTube transcript parsing completed. Results saved to {output_path}")
    else:
        print("Invalid task. Please choose from 'pdf', 'tweet', or 'youtube'.")

if __name__ == "__main__":
    main()
