"""Utility functions."""

import os
import subprocess
import logging
import json

def check_dependencies():
    """Check if required external dependencies are installed.
    
    Returns:
        bool: True if all dependencies are installed, False otherwise
    """
    dependencies = {
        'ffmpeg': 'ffmpeg -version',
        'ffprobe': 'ffprobe -version'
    }
    
    missing = []
    for dep, cmd in dependencies.items():
        try:
            subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            logging.info(f"{dep} is installed")
        except subprocess.CalledProcessError:
            logging.error(f"{dep} is not installed")
            missing.append(dep)
    
    if missing:
        logging.error(f"Missing dependencies: {', '.join(missing)}")
        logging.error("Please install the missing dependencies and try again.")
        return False
    
    return True

def extract_audio_from_video(video_path, audio_path, format='wav'):
    """Extract audio from video file.
    
    Args:
        video_path: Path to video file
        audio_path: Path to output audio file
        format: Audio format (wav, mp3, etc.)
        
    Returns:
        bool: True if successful, False otherwise
    """
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le' if format == 'wav' else 'libmp3lame',
        '-ar', '16000', '-ac', '1', '-y', audio_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f"Audio extracted to {audio_path}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to extract audio: {e}")
        return False

def get_video_duration(video_path):
    """Get duration of video in seconds.
    
    Args:
        video_path: Path to video file
        
    Returns:
        float: Duration in seconds or None if failed
    """
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        return float(info['format']['duration'])
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        logging.error(f"Failed to get video duration: {e}")
        return None

def get_video_metadata(video_path):
    """Get video metadata.
    
    Args:
        video_path: Path to video file
        
    Returns:
        dict: Video metadata or None if failed
    """
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_streams', '-show_format', video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        logging.error(f"Failed to get video metadata: {e}")
        return None