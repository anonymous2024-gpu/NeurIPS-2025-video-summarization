"""Summary generation module using multiple LLM providers (OpenAI and LLaMA)."""

import os
import json
import time
import random
import logging
from tqdm import tqdm
import numpy as np
from openai import OpenAI
from openai import RateLimitError, APIError, Timeout
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def build_prompt(transcript_text):
    """Build prompt for the LLM models.
    
    Args:
        transcript_text: Transcript text with timestamps
        
    Returns:
        Prompt for LLM
    """
    return f"""Given a transcript of a video with sentence-level timestamps, generate an extractive summary that selects sentences conveying both high semantic importance and behavioural salience.
Guidelines:
1. Select only sentences that are both contextually informative and exhibit significant behavioural cues.
2. Preserve the exact wording of each selected sentence. No paraphrasing.
3. Retain the original timestamp boundaries for each selected sentence.
4. Format the output as a list of triplets in the form: [start_time, end_time, sentence].
{transcript_text}
"""

def load_transcript(file_path):
    """Load transcript from file.
    
    Args:
        file_path: Path to transcript file
        
    Returns:
        Transcript text
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def save_summary(base_name, output_dir, summary_text, model_name):
    """Save summary to file in text and JSON formats.
    
    Args:
        base_name: Base name for output files
        output_dir: Output directory
        summary_text: Summary text
        model_name: Name of the model used for summary
        
    Returns:
        Tuple of (summary_path, json_path)
    """
    summary_path = os.path.join(output_dir, f"{base_name}_{model_name}_sum.txt")
    
    if os.path.exists(summary_path):
        logging.info(f"Skipping existing file: {summary_path}")
        return None, None
        
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    # Parse summary text to JSON format
    blocks = summary_text.strip().split("\n\n")
    summary_json = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 2 and "," in lines[0]:
            try:
                time_range = lines[0].strip().replace("s", "").split(",")
                start = float(time_range[0])
                end = float(time_range[1])
                sentence = lines[1].strip()
                summary_json.append({
                    "start": start,
                    "end": end,
                    "sentence": sentence
                })
            except:
                continue

    json_path = os.path.join(output_dir, f"{base_name}_{model_name}_sum.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    return summary_path, json_path

def call_openai_api(prompt, model="gpt-4.5-preview", max_tokens=1024, temperature=1, retries=5, delay=5):
    """Call OpenAI API with retries.
    
    Args:
        prompt: Prompt for the API
        model: Model to use (gpt-4.5-preview or gpt-3.5-turbo)
        max_tokens: Maximum number of tokens in the response
        temperature: Temperature for the API
        retries: Number of retries
        delay: Delay between retries in seconds
        
    Returns:
        API response text
    """
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    
    client = OpenAI(api_key=api_key)
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
            )
            return response.choices[0].message.content
        except (RateLimitError, APIError, Timeout) as e:
            wait_time = delay + random.uniform(0, 2)
            logging.warning(f"Retry {attempt + 1}/{retries} in {wait_time:.1f}s due to: {type(e).__name__}")
            time.sleep(wait_time)
    
    raise Exception(f"Failed to call OpenAI API for model {model} after multiple retries")

def call_llama_model(prompt, max_tokens=1024, temperature=0.7):
    """Call LLaMA-3.2-3B model using HuggingFace transformers.
    
    Args:
        prompt: Prompt for the model
        max_tokens: Maximum number of tokens in the response
        temperature: Temperature parameter
        
    Returns:
        Model response text
    """
    try:
        # Check for GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Loading LLaMA model on {device}")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
        
        # Move model to device
        model = model.to(device)
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        response = response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
        
        return response.strip()
    except Exception as e:
        logging.error(f"Error calling LLaMA model: {e}")
        raise

def process_batch(config, model_name="gpt-4.5"):
    """Process a batch of transcripts with specified model.
    
    Args:
        config: Configuration object
        model_name: Model name to use (gpt-4.5, gpt-3.5, or llama-3)
        
    Returns:
        List of processed files
    """
    transcript_dir = config.audio_dir
    output_dir = config.audio_dir
    
    model_mapping = {
        "gpt-4.5": {"api": "openai", "model": "gpt-4.5-preview", "max_tokens": 1024, "temperature": 1.0},
        "gpt-3.5": {"api": "openai", "model": "gpt-3.5-turbo", "max_tokens": 1024, "temperature": 0.7},
        "llama-3": {"api": "huggingface", "model": "llama-3.2-3b", "max_tokens": 1024, "temperature": 0.7}
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Unknown model: {model_name}. Supported models: {', '.join(model_mapping.keys())}")
    
    model_config = model_mapping[model_name]
    processed_files = []
    
    files = [f for f in os.listdir(transcript_dir) if f.endswith("_timestamps.txt")]
    for file in tqdm(files, desc=f"Processing Transcripts with {model_name}", unit="file"):
        file_path = os.path.join(transcript_dir, file)
        base_name = os.path.splitext(file)[0].replace("_timestamps", "")

        txt_path = os.path.join(output_dir, f"{base_name}_{model_name}_sum.txt")
        json_path = os.path.join(output_dir, f"{base_name}_{model_name}_sum.json")

        if os.path.exists(txt_path) and os.path.exists(json_path):
            logging.info(f"Skipping {base_name} (summary files already exist)")
            processed_files.append(base_name)
            continue
        
        try:
            logging.info(f"Processing {base_name} with {model_name}...")
            transcript_text = load_transcript(file_path)
            prompt = build_prompt(transcript_text)
            
            if model_config["api"] == "openai":
                summary_text = call_openai_api(
                    prompt, 
                    model=model_config["model"], 
                    max_tokens=model_config["max_tokens"], 
                    temperature=model_config["temperature"]
                )
            elif model_config["api"] == "huggingface":
                summary_text = call_llama_model(
                    prompt,
                    max_tokens=model_config["max_tokens"],
                    temperature=model_config["temperature"]
                )
            else:
                raise ValueError(f"Unknown API type: {model_config['api']}")

            txt_out, json_out = save_summary(base_name, output_dir, summary_text, model_name)
            if txt_out and json_out:
                logging.info(f"Summary saved to {txt_out} and {json_out}")
                processed_files.append(base_name)
        except Exception as e:
            logging.error(f"Failed to process {base_name} with {model_name}: {e}")
            continue

        # Add a small delay between API calls
        time.sleep(5 + random.uniform(0, 2))
    
    return processed_files

def calculate_jaccard_similarity(summary1, summary2):
    """Calculate Jaccard similarity between two summaries.
    
    Args:
        summary1: First summary as a list of sentences
        summary2: Second summary as a list of sentences
        
    Returns:
        Jaccard similarity score
    """
    set1 = set(summary1)
    set2 = set(summary2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0
    
    return intersection / union

def extract_sentences_from_summary(summary_path):
    """Extract sentences from a summary file.
    
    Args:
        summary_path: Path to summary file
        
    Returns:
        List of sentences
    """
    sentences = []
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_text = f.read()
        blocks = summary_text.strip().split("\n\n")
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) >= 2:
                sentence = lines[1].strip()
                sentences.append(sentence)
    
    return sentences

def consistency_verification(config, num_videos=10, num_iterations=3):
    """Perform consistency verification for a subset of videos.
    
    Args:
        config: Configuration object
        num_videos: Number of videos to verify
        num_iterations: Number of iterations per video
        
    Returns:
        Average Jaccard similarity score
    """
    transcript_dir = config.audio_dir
    output_dir = config.audio_dir
    
    # Get list of all transcript files
    files = [f for f in os.listdir(transcript_dir) if f.endswith("_timestamps.txt")]
    
    # Randomly select a subset of videos
    if num_videos > len(files):
        num_videos = len(files)
    
    selected_files = random.sample(files, num_videos)
    similarities = []
    
    logging.info(f"Performing consistency verification on {num_videos} videos with {num_iterations} iterations each")
    
    for file in tqdm(selected_files, desc="Verifying Consistency", unit="file"):
        file_path = os.path.join(transcript_dir, file)
        base_name = os.path.splitext(file)[0].replace("_timestamps", "")
        
        # Generate multiple summaries with different temperatures
        temperatures = [0.5, 0.7, 0.9]
        summaries = []
        
        for i, temp in enumerate(temperatures):
            iteration_name = f"gpt-4.5_cv_{i+1}"
            output_path = os.path.join(output_dir, f"{base_name}_{iteration_name}_sum.txt")
            
            if os.path.exists(output_path):
                logging.info(f"Using existing summary for {base_name} (iteration {i+1})")
            else:
                logging.info(f"Generating summary for {base_name} (iteration {i+1})")
                transcript_text = load_transcript(file_path)
                prompt = build_prompt(transcript_text)
                
                summary_text = call_openai_api(
                    prompt, 
                    model="gpt-4.5-preview", 
                    max_tokens=1024, 
                    temperature=temp
                )
                
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(summary_text)
            
            sentences = extract_sentences_from_summary(output_path)
            summaries.append(sentences)
        
        # Calculate pairwise Jaccard similarities
        for i in range(len(summaries)):
            for j in range(i+1, len(summaries)):
                similarity = calculate_jaccard_similarity(summaries[i], summaries[j])
                similarities.append(similarity)
                logging.info(f"Jaccard similarity for {base_name} (iterations {i+1} and {j+1}): {similarity:.4f}")
        
        # Add a small delay between videos
        time.sleep(2)
    
    # Calculate average similarity
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    logging.info(f"Average Jaccard similarity across {num_videos} videos: {avg_similarity:.4f}")
    
    # Save results to file
    results_path = os.path.join(output_dir, "consistency_verification_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_videos": num_videos,
            "num_iterations": num_iterations,
            "similarities": similarities,
            "average_similarity": avg_similarity
        }, f, indent=2)
    
    return avg_similarity

def generate_summaries(config):
    """Generate summaries for all transcripts using multiple models.
    
    Args:
        config: Configuration object
    """
    try:
        # Process with GPT-4.5
        logging.info("Generating summaries with GPT-4.5...")
        process_batch(config, model_name="gpt-4.5")
        
        # Process with GPT-3.5
        logging.info("Generating summaries with GPT-3.5...")
        process_batch(config, model_name="gpt-3.5")
        
        # Process with LLaMA-3
        logging.info("Generating summaries with LLaMA-3.2-3B...")
        process_batch(config, model_name="llama-3")
        
        logging.info("Summary generation completed for all models.")
    except Exception as e:
        logging.error(f"Error during summary generation: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate summaries using multiple LLM models")
    parser.add_argument("--transcript_dir", required=True, help="Directory containing transcript files")
    parser.add_argument("--output_dir", help="Output directory (defaults to transcript_dir)")
    parser.add_argument("--models", default="all", help="Models to use (comma-separated, e.g., 'gpt-4.5,gpt-3.5' or 'all')")
    parser.add_argument("--verify", action="store_true", help="Perform consistency verification")
    parser.add_argument("--num_videos", type=int, default=10, help="Number of videos for consistency verification")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("summary_generator.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create mock config object
    class MockConfig:
        def __init__(self):
            self.audio_dir = args.transcript_dir
            self.output_dir = args.output_dir or args.transcript_dir
    
    config = MockConfig()
    
    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Generate summaries
    if args.models.lower() == "all":
        models = ["gpt-4.5", "gpt-3.5", "llama-3"]
    else:
        models = [m.strip() for m in args.models.split(",")]
    
    for model in models:
        try:
            logging.info(f"Generating summaries with {model}...")
            process_batch(config, model_name=model)
        except Exception as e:
            logging.error(f"Error during {model} summary generation: {e}")
    
    # Perform consistency verification if requested
    if args.verify:
        consistency_verification(config, num_videos=args.num_videos)

if __name__ == "__main__":
    main()