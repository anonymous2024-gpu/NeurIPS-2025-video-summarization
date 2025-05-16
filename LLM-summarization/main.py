#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import logging
from dotenv import load_dotenv

# Import the component modules
from LLM.transcript_processor import process_transcript
from LLM.summary_generator import generate_summaries
from LLM.video_generator import generate_summary_videos
from LLM.config import Config

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Video Summarization Tool")
    parser.add_argument("--video_dir", type=str, help="Directory containing video files")
    parser.add_argument("--output_dir", type=str, help="Directory for output files")
    parser.add_argument("--mfa_dir", type=str, help="Directory containing MFA output files")
    parser.add_argument("--model", type=str, default="gpt-4.5-preview", help="OpenAI model to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("video_summarizer.log"),
            logging.StreamHandler()
        ]
    )
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Create configuration from arguments
    config = Config()
    if args.video_dir:
        config.video_folder = args.video_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.mfa_dir:
        config.mfa_output_dir = args.mfa_dir
    if args.model:
        config.model = args.model
    
    # Create required directories
    os.makedirs(config.audio_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.temp_dir_base, exist_ok=True)
    
    # Run the pipeline
    logging.info("Starting video summarization pipeline")
    
    # Step 1: Process transcripts
    logging.info("Step 1: Processing transcripts")
    process_transcript(config)
    
    # Step 2: Generate summaries
    logging.info("Step 2: Generating summaries")
    generate_summaries(config)
    
    # Step 3: Create summary videos
    logging.info("Step 3: Creating summary videos")
    generate_summary_videos(config)
    
    logging.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()
