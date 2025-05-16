"""Transcript processing and word alignment module."""

import os
import re
import logging
import torch
import whisper
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from textgrid import TextGrid

# Initialize NLP components
def init_nlp():
    """Initialize NLP components."""
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # If model not found, download it
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    # Download NLTK resources
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    
    # Check for device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Running on: {device}")
    
    return nlp, device

def extract_word_timestamps(textgrid_path):
    """Extract word-level timestamps from TextGrid file.
    
    Args:
        textgrid_path: Path to the TextGrid file
        
    Returns:
        List of tuples (word, start_time, end_time)
    """
    try:
        tg = TextGrid.fromFile(textgrid_path)
        return [(interval.mark.strip(), interval.minTime, interval.maxTime) 
                for interval in tg[0] if interval.mark.strip()]
    except Exception as e:
        logging.error(f"Error extracting word timestamps from {textgrid_path}: {e}")
        return []

def process_transcript_from_file(audio_dir, video_name, textgrid_path):
    """Process transcript from file and extract word timestamps.
    
    Args:
        audio_dir: Directory containing audio files
        video_name: Name of the video file (without extension)
        textgrid_path: Path to the TextGrid file
        
    Returns:
        Tuple of (corrected_transcript, word_timestamps)
    """
    transcription_file = os.path.join(audio_dir, f"{video_name}.txt")
    timestamps_file = os.path.join(audio_dir, f"{video_name}_timestamps.txt")

    # Check if files already exist
    if os.path.exists(transcription_file) and os.path.exists(timestamps_file):
        logging.info(f"Skipping processing for {video_name}: Transcript and timestamps already exist.")
        with open(transcription_file, "r", encoding="utf-8") as f:
            corrected_transcript = f.read().strip()
        word_timestamps = extract_word_timestamps(textgrid_path)
        return corrected_transcript, word_timestamps

    if not os.path.exists(transcription_file):
        logging.error(f"Transcript not found: {transcription_file}")
        return None, None
    
    # Load the transcript
    with open(transcription_file, "r", encoding="utf-8") as f:
        raw_transcript = f.read().strip()

    # Initialize NLP
    nlp, _ = init_nlp()
    
    # Process and clean the transcript
    doc = nlp(raw_transcript)
    corrected_sentences = []
    for sent in doc.sents:
        sentence = sent.text.strip()
        sentence = sentence.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
        sentence = re.sub(r'\.{2,}', '.', sentence)
        corrected_sentences.append(sentence)
    corrected_transcript = " ".join(corrected_sentences)

    # Extract word-level timestamps
    word_timestamps = extract_word_timestamps(textgrid_path)

    # Save corrected transcript if it doesn't exist
    if not os.path.exists(transcription_file):
        with open(transcription_file, "w", encoding="utf-8") as f:
            f.write(corrected_transcript)
        logging.info(f"Cleaned transcript saved: {transcription_file}")

    # Map sentences to timestamps and save
    if not os.path.exists(timestamps_file):
        sentence_timestamps = map_sentences_to_timestamps(corrected_transcript, word_timestamps)
        with open(timestamps_file, "w", encoding="utf-8") as f:
            for sentence, start_time, end_time in sentence_timestamps:
                f.write(f"{start_time:.2f}s, {end_time:.2f}s\n{sentence}\n\n")
        logging.info(f"Timestamps saved: {timestamps_file}")

    return corrected_transcript, word_timestamps

def map_sentences_to_timestamps(text, word_timestamps, min_match_ratio=0.3, debug=False):
    """Map sentences to timestamps based on word-level alignment.
    
    Args:
        text: Text to map
        word_timestamps: List of (word, start_time, end_time) tuples
        min_match_ratio: Minimum ratio of matching words to consider a match
        debug: Whether to print debug information
        
    Returns:
        List of (sentence, start_time, end_time) tuples
    """
    sentences = sent_tokenize(text)
    sentence_timestamps = []

    for sentence in sentences:
        # Skip empty or trivial sentences
        if not sentence.strip() or sentence.strip() in ['"', '.', '."']:
            continue

        words = sentence.lower().split()
        start_time = None
        end_time = None
        best_match_start_idx = None
        best_match_end_idx = None
        max_matches = 0

        # Find best matching window of words
        window_size = min(len(words), len(word_timestamps))
        for i in range(len(word_timestamps) - window_size + 1):
            matches = 0
            for j, (w, start, end) in enumerate(word_timestamps[i:i + window_size]):
                if j < len(words) and w.lower() == words[j]:
                    matches += 1
            if matches > max_matches:
                max_matches = matches
                best_match_start_idx = i
                best_match_end_idx = i + window_size - 1

        # Accept match if at least min_match_ratio of words match
        if max_matches >= len(words) * min_match_ratio:
            start_time = word_timestamps[best_match_start_idx][1]
            end_time = word_timestamps[best_match_end_idx][2]
            if debug:
                logging.debug(f"Mapped: '{sentence}' -> {start_time}s - {end_time}s (matches: {max_matches}/{len(words)})")
        else:
            # Fallback: Use first occurrence of any word or pause detection
            if debug:
                logging.debug(f"Warning: No good match for '{sentence}' (matches: {max_matches}/{len(words)})")
            for i, (w, start, end) in enumerate(word_timestamps):
                if start_time is None and w.lower() in [word.lower() for word in words]:
                    start_time = start
                if start_time is not None and i > 0 and (start - word_timestamps[i - 1][2]) > 0.5:
                    end_time = word_timestamps[i - 1][2]
                    break
            if start_time is None:
                start_time = word_timestamps[0][1]  # Default to start of video
            if end_time is None:
                end_time = word_timestamps[-1][2]  # Default to end of video
            if debug:
                logging.debug(f"Fallback: '{sentence}' -> {start_time}s - {end_time}s")

        sentence_timestamps.append((sentence, start_time, end_time))

    return sorted(sentence_timestamps, key=lambda x: x[1])

def transcribe_audio(audio_path, model_name="large"):
    """Transcribe audio using Whisper model.
    
    Args:
        audio_path: Path to the audio file
        model_name: Whisper model name (tiny, base, small, medium, large)
        
    Returns:
        Transcript text
    """
    _, device = init_nlp()
    
    # Load Whisper model
    model = whisper.load_model(model_name)
    model = model.to(device)
    
    # Transcribe audio
    result = model.transcribe(audio_path)
    return result["text"]

def process_transcript(config):
    """Process transcripts for all videos in the input directory.
    
    Args:
        config: Configuration object
    """
    video_folder = config.video_folder
    audio_dir = config.audio_dir
    mfa_output_dir = config.mfa_output_dir
    
    # Create audio directory if it doesn't exist
    os.makedirs(audio_dir, exist_ok=True)
    
    # Process all video files
    for file in os.listdir(video_folder):
        if file.endswith(".mp4"):
            video_name = os.path.splitext(file)[0]
            textgrid_path = os.path.join(mfa_output_dir, f"{video_name}.TextGrid")
            
            if os.path.exists(textgrid_path):
                logging.info(f"Processing transcript for {video_name}")
                process_transcript_from_file(audio_dir, video_name, textgrid_path)
            else:
                logging.warning(f"Skipping (no TextGrid): {video_name}")
