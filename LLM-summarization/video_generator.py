"""Creating summary videos."""

import os
import re
import json
import shutil
import logging
import subprocess
from nltk.tokenize import sent_tokenize

from LLM.transcript_processor import process_transcript_from_file, map_sentences_to_timestamps

def probe_video(video_path):
    """Get video information using ffprobe.
    
    Args:
        video_path: Path to video file
        
    Returns:
        JSON object with video information
    """
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', 
           '-show_streams', '-show_format', video_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to probe video {video_path}: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse ffprobe output: {e}")
        return None

def extract_video_segment_with_frames(video_path, start_frame, end_frame, output_path):
    """Extract a segment of video based on frame numbers.
    
    Args:
        video_path: Path to input video
        start_frame: Start frame number
        end_frame: End frame number
        output_path: Path to output video
        
    Returns:
        Path to output video
    """
    probe = probe_video(video_path)
    if not probe:
        logging.error(f"Failed to probe video {video_path}")
        return None
    
    # Get video fps
    video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
    if not video_stream:
        logging.error(f"No video stream found in {video_path}")
        return None
    
    # Parse frame rate as a fraction (e.g., "30000/1001") and evaluate it
    fps = eval(video_stream['r_frame_rate'])
    
    # Convert frame numbers to time in seconds
    start_time = start_frame / fps
    end_time = (end_frame + 1) / fps

    # Use ffmpeg to extract the segment
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f"select='between(n,{start_frame},{end_frame})',setpts=PTS-STARTPTS",
        '-af', f"atrim=start={start_time}:end={end_time},asetpts=PTS-STARTPTS",
        '-c:v', 'libx264', '-c:a', 'aac', '-y', output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check output video
        probe = probe_video(output_path)
        if probe:
            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            if video_stream:
                extracted_frames = int(video_stream['nb_frames'])
                actual_duration = float(probe['format']['duration'])
                
                logging.info(f"Segment: {output_path}")
                logging.info(f"Expected frames: {end_frame - start_frame + 1}")
                logging.info(f"Actual frames: {extracted_frames}")
                logging.info(f"Actual duration: {actual_duration:.3f}s")
                
                return output_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to extract segment: {e}")
    
    return None

def generate_srt_file(frame_timestamps, srt_path):
    """Generate SRT subtitle file from frame timestamps.
    
    Args:
        frame_timestamps: List of (sentence, start_time, end_time, start_frame, end_frame) tuples
        srt_path: Path to output SRT file
    """
    total_time_offset = 0.0
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, (sentence, start_time, end_time, start_frame, end_frame) in enumerate(frame_timestamps, 1):
            duration = end_time - start_time
            start_srt = total_time_offset
            end_srt = total_time_offset + duration
            
            # Format timestamps in SRT format: HH:MM:SS,mmm
            start_srt_str = f"{int(start_srt // 3600):02d}:{int((start_srt % 3600) // 60):02d}:{int(start_srt % 60):02d},{int((start_srt % 1) * 1000):03d}"
            end_srt_str = f"{int(end_srt // 3600):02d}:{int((end_srt % 3600) // 60):02d}:{int(end_srt % 60):02d},{int((end_srt % 1) * 1000):03d}"
            
            f.write(f"{i}\n")
            f.write(f"{start_srt_str} --> {end_srt_str}\n")
            f.write(f"{sentence}\n\n")
            
            total_time_offset += duration
    
    logging.info(f"SRT file generated: {srt_path}")

def create_summary_video(segment_paths, output_path, srt_path, temp_dir):
    """Create summary video by concatenating segments and adding subtitles.
    
    Args:
        segment_paths: List of segment video paths
        output_path: Path to output video
        srt_path: Path to SRT subtitle file
        temp_dir: Temporary directory
    """
    if not segment_paths:
        logging.error("No segments to concatenate")
        return
    
    # Create concat list file
    concat_list = os.path.join(temp_dir, 'concat_list.txt')
    with open(concat_list, 'w') as f:
        for segment in segment_paths:
            f.write(f"file '{segment}'\n")

    # Concatenate segments
    temp_output = os.path.join(temp_dir, 'temp_concat.mp4')
    cmd_concat = [
        'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_list,
        '-c', 'copy', '-y', temp_output
    ]
    
    try:
        subprocess.run(cmd_concat, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Add subtitles
        cmd_subtitle = [
            'ffmpeg', '-i', temp_output,
            '-vf', f"subtitles={srt_path}:force_style='Fontsize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H80000000'",
            '-c:v', 'libx264', '-c:a', 'aac', '-y', output_path
        ]
        
        subprocess.run(cmd_subtitle, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f"Summary video created: {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to create summary video: {e}")

def process_video(video_name, video_path, textgrid_path, summary_file, output_video_dir, temp_dir, config):
    """Process a single video to create a summary video.
    
    Args:
        video_name: Name of the video file (without extension)
        video_path: Path to the video file
        textgrid_path: Path to the TextGrid file
        summary_file: Path to the summary file
        output_video_dir: Directory for output video
        temp_dir: Temporary directory
        config: Configuration object
        
    Returns:
        Path to output video or None if failed
    """
    output_path = os.path.join(output_video_dir, f"{video_name}_LLMSum.mp4")
    
    logging.info(f"Processing video: {video_name}")
    
    # Get transcript and word timestamps
    corrected_transcript, word_timestamps = process_transcript_from_file(
        config.audio_dir, video_name, textgrid_path
    )
    
    if word_timestamps is None:
        logging.error(f"Failed to process transcript for {video_name}")
        return None
    
    # Read summary file
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary_lines = f.read().strip().split('\n')
    
    # Clean and process summary
    cleaned_summary = []
    for line in summary_lines:
        if not re.match(r'^\d+\.\d+s,\s*\d+\.\d+s$', line.strip()):
            # Clean quotes but preserve sentence-ending punctuation
            cleaned_line = line.strip().strip('"')
            if cleaned_line and not cleaned_line.isspace():
                # Ensure the sentence ends with a period if it lacks terminal punctuation
                if not cleaned_line.endswith(('.', '!', '?')):
                    cleaned_line += '.'
                cleaned_summary.append(cleaned_line)
    
    # Join sentences with a space, preserving individual sentence boundaries
    cleaned_summary_text = " ".join(cleaned_summary)
    logging.info(f"Cleaned summary for {video_name}: '{cleaned_summary_text}'")
    
    # Save cleaned summary
    cleaned_summary_file = os.path.join(config.audio_dir, f"{video_name}_LLM.txt")
    with open(cleaned_summary_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_summary_text)
    logging.info(f"Cleaned summary saved: {cleaned_summary_file}")
    
    # Map sentences to timestamps
    sentence_timestamps = map_sentences_to_timestamps(
        cleaned_summary_text, word_timestamps, min_match_ratio=config.min_match_ratio, debug=config.debug
    )
    
    if not sentence_timestamps:
        logging.error(f"No sentence timestamps found for {video_name}")
        return None
    
    logging.info(f"Sentence timestamps for {video_name}: {sentence_timestamps}")
    
    # Get video fps
    probe = probe_video(video_path)
    if not probe:
        logging.error(f"Failed to probe video {video_path}")
        return None
    
    video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
    if not video_stream:
        logging.error(f"No video stream found in {video_path}")
        return None
    
    fps = eval(video_stream['r_frame_rate'])
    
    # Convert timestamps to frame numbers
    frame_timestamps = []
    for sentence, start_time, end_time in sentence_timestamps:
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        frame_timestamps.append((sentence, start_time, end_time, start_frame, end_frame))
        logging.info(f"Sentence: '{sentence}' -> {start_time}s - {end_time}s (Frames: {start_frame} - {end_frame})")
    
    # Extract video segments
    segment_files = []
    for i, (sentence, start_time, end_time, start_frame, end_frame) in enumerate(frame_timestamps):
        segment_path = os.path.join(temp_dir, f"temp_segment_{i}.mp4")
        output = extract_video_segment_with_frames(video_path, start_frame, end_frame, segment_path)
        if output:
            segment_files.append(output)
    
    if not segment_files:
        logging.error(f"No segments extracted for {video_name}")
        return None
    
    # Generate SRT file
    srt_path = os.path.join(temp_dir, "subtitles.srt")
    generate_srt_file(frame_timestamps, srt_path)
    
    # Create summary video
    create_summary_video(segment_files, output_path, srt_path, temp_dir)
    
    return output_path

def generate_summary_videos(config):
    """Generate summary videos for all processed videos.
    
    Args:
        config: Configuration object
    """
    video_folder = config.video_folder
    mfa_output_dir = config.mfa_output_dir
    audio_dir = config.audio_dir
    output_dir = config.output_dir
    temp_dir_base = config.temp_dir_base
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all videos
    for video_file in os.listdir(video_folder):
        if not video_file.endswith(".mp4"):
            continue
        
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(video_folder, video_file)
        textgrid_path = os.path.join(mfa_output_dir, f"{video_name}.TextGrid")
        summary_file = os.path.join(audio_dir, f"{video_name}_LLMsum.txt")
        output_video_dir = os.path.join(output_dir, video_name)
        output_path = os.path.join(output_video_dir, f"{video_name}_LLMSum.mp4")
        
        # Skip if summary video already exists
        if os.path.exists(output_path):
            logging.info(f"Skipping {video_name}: Summary video already exists at {output_path}.")
            continue
        
        # Check required files
        if not os.path.exists(textgrid_path):
            logging.warning(f"Skipping {video_name}: No TextGrid file found at {textgrid_path}.")
            continue
        if not os.path.exists(summary_file):
            logging.warning(f"Skipping {video_name}: No LLM summary file found at {summary_file}.")
            continue
        
        # Create temporary directory
        temp_dir = os.path.join(temp_dir_base, video_name)
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(output_video_dir, exist_ok=True)
        
        try:
            # Process video
            process_video(
                video_name, video_path, textgrid_path, summary_file, 
                output_video_dir, temp_dir, config
            )
        except Exception as e:
            logging.error(f"Error processing {video_name}: {e}")
        finally:
            # Clean up
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logging.info(f"Temporary folder {temp_dir} removed.")
