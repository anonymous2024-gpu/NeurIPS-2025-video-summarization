import os
import subprocess
import logging
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np

def setup_logging(log_file=None, level=logging.INFO):
    """Setup logging configuration"""
    logging_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if log_file:
        logging.basicConfig(
            level=level,
            format=logging_format,
            filename=log_file,
            filemode='a'  # Append mode
        )
        # Add console handler to also print to console
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(logging_format))
        logging.getLogger('').addHandler(console)
    else:
        logging.basicConfig(
            level=level,
            format=logging_format
        )
    return logging.getLogger()

def check_ffmpeg():
    """Check if FFmpeg is installed and available in path"""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def generate_segment_file(timestamps, duration=5.0, output_file="segments.txt"):
    """
    Generate a segment file for FFmpeg with the given timestamps.
    Each segment starts at the given timestamp and has the specified duration.
    
    Args:
        timestamps: List of timestamps in seconds
        duration: Duration of each segment in seconds
        output_file: Path to write the segment file
    
    Returns:
        Path to the segment file
    """
    with open(output_file, 'w') as f:
        for i, ts in enumerate(timestamps):
            start_time = max(0, ts)  # Ensure we don't have negative start times
            f.write(f"file '{i:04d}.mp4'\n")
            
    return output_file

def extract_segments(video_path, timestamps, duration=5.0, temp_dir=None, with_subtitles=True):
    """
    Extract segments from a video based on timestamps.
    
    Args:
        video_path: Path to the input video file
        timestamps: List of timestamps in seconds
        duration: Duration of each segment in seconds
        temp_dir: Directory to store temporary files
        with_subtitles: Whether to include subtitles in the segments
        
    Returns:
        Directory containing the extracted segments
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    else:
        os.makedirs(temp_dir, exist_ok=True)
    
    # Check if video contains subtitles
    has_subtitles = False
    if with_subtitles:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 's', 
            '-show_entries', 'stream=index', '-of', 'json', video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            probe_result = json.loads(result.stdout)
            has_subtitles = 'streams' in probe_result and len(probe_result['streams']) > 0
            logging.info(f"Subtitle streams detected: {has_subtitles}")
        except json.JSONDecodeError:
            logging.warning("Failed to parse FFprobe output for subtitle detection")
    
    # Extract segments
    logging.info(f"Extracting {len(timestamps)} segments...")
    segment_paths = []
    
    for i, ts in enumerate(timestamps):
        start_time = max(0, ts)  # Ensure we don't have negative start times
        output_path = os.path.join(temp_dir, f"{i:04d}.mp4")
        
        # Build FFmpeg command
        # -ss before -i is more accurate for seeking
        cmd = ['ffmpeg', '-y', '-ss', str(start_time), '-i', video_path]
        
        # Add subtitle mapping if available
        if has_subtitles and with_subtitles:
            cmd.extend(['-map', '0:v:0', '-map', '0:a:0?', '-map', '0:s:0?'])
            cmd.extend(['-c:s', 'copy'])  # Copy subtitles stream
        
        # Add duration and output settings
        cmd.extend([
            '-t', str(duration),
            '-c:v', 'libx264', '-c:a', 'aac', 
            '-preset', 'fast', '-crf', '22',
            output_path
        ])
        
        logging.debug(f"Running command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            segment_paths.append(output_path)
            logging.debug(f"Extracted segment {i+1}/{len(timestamps)}")
        except subprocess.SubprocessError as e:
            logging.error(f"Failed to extract segment at timestamp {start_time}: {e}")
    
    logging.info(f"Extracted {len(segment_paths)} segments successfully")
    return temp_dir, segment_paths

def concatenate_segments(segments_dir, output_path, cleanup=True):
    """
    Concatenate extracted segments into a final summary video.
    
    Args:
        segments_dir: Directory containing the extracted segments
        output_path: Path to the output summary video
        cleanup: Whether to clean up temporary files after concatenation
        
    Returns:
        Path to the output summary video
    """
    # Create segment list file
    segments_file = os.path.join(segments_dir, "segments.txt")
    
    # List all segment files and sort them numerically
    segment_files = sorted([f for f in os.listdir(segments_dir) if f.endswith('.mp4')])
    
    if not segment_files:
        logging.error("No segments found to concatenate")
        return None
    
    # Create segments file
    with open(segments_file, 'w') as f:
        for segment in segment_files:
            f.write(f"file '{segment}'\n")
    
    # Build FFmpeg concatenation command
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', segments_file,
        '-c', 'copy',
        output_path
    ]
    
    logging.info(f"Concatenating segments to {output_path}")
    logging.debug(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f"Summary video created successfully: {output_path}")
    except subprocess.SubprocessError as e:
        logging.error(f"Failed to concatenate segments: {e}")
        return None
    
    # Clean up
    if cleanup:
        logging.debug("Cleaning up temporary files")
        try:
            # Keep the directory but remove the segment files
            for segment in segment_files:
                os.remove(os.path.join(segments_dir, segment))
            os.remove(segments_file)
        except OSError as e:
            logging.warning(f"Failed to clean up some temporary files: {e}")
    
    return output_path

def create_summary_video(video_path, timestamps, output_path, segment_duration=5.0, 
                        temp_dir=None, with_subtitles=True, cleanup=True):
    """
    Create a summary video from the given timestamps.
    
    Args:
        video_path: Path to the input video file
        timestamps: List of timestamps in seconds
        output_path: Path to the output summary video
        segment_duration: Duration of each segment in seconds
        temp_dir: Directory to store temporary files
        with_subtitles: Whether to include subtitles
        cleanup: Whether to clean up temporary files after concatenation
        
    Returns:
        Path to the output summary video
    """
    # First, check if FFmpeg is available
    if not check_ffmpeg():
        logging.error("FFmpeg is not installed or not in PATH. Please install FFmpeg first.")
        return None
    
    # Create temp directory if not provided
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
        logging.debug(f"Created temporary directory: {temp_dir}")
    else:
        os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Sort timestamps
        sorted_timestamps = sorted(timestamps)
        
        # Extract segments
        segments_dir, segment_paths = extract_segments(
            video_path=video_path,
            timestamps=sorted_timestamps,
            duration=segment_duration,
            temp_dir=temp_dir,
            with_subtitles=with_subtitles
        )
        
        if not segment_paths:
            logging.error("No segments were extracted successfully")
            return None
        
        # Concatenate segments
        summary_path = concatenate_segments(
            segments_dir=segments_dir,
            output_path=output_path,
            cleanup=cleanup
        )
        
        # Final cleanup
        if cleanup:
            try:
                shutil.rmtree(temp_dir)
                logging.debug(f"Removed temporary directory: {temp_dir}")
            except OSError as e:
                logging.warning(f"Failed to remove temporary directory: {e}")
        
        return summary_path
    
    except Exception as e:
        logging.error(f"Failed to create summary video: {e}")
        # Cleanup on error
        if cleanup and temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except OSError:
                pass
        return None

def read_timestamps_file(file_path):
    """
    Read timestamps from a file.
    
    Args:
        file_path: Path to the timestamps file
        
    Returns:
        List of timestamps in seconds
    """
    timestamps = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                ts = float(line.strip())
                timestamps.append(ts)
            except ValueError:
                logging.warning(f"Invalid timestamp in file: {line.strip()}")
    
    return timestamps

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate a summary video from timestamps')
    parser.add_argument('--video', required=True, help='Path to the input video file')
    parser.add_argument('--timestamps', required=True, help='Path to the timestamps file or comma-separated list of timestamps')
    parser.add_argument('--output', required=True, help='Path to the output summary video')
    parser.add_argument('--duration', type=float, default=5.0, help='Duration of each segment in seconds')
    parser.add_argument('--temp-dir', help='Directory to store temporary files')
    parser.add_argument('--no-subtitles', action='store_true', help='Do not include subtitles')
    parser.add_argument('--keep-temp', action='store_true', help='Keep temporary files after processing')
    parser.add_argument('--log', help='Path to log file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log)
    
    # Parse timestamps
    if os.path.exists(args.timestamps):
        timestamps = read_timestamps_file(args.timestamps)
    else:
        try:
            timestamps = [float(ts.strip()) for ts in args.timestamps.split(',')]
        except ValueError:
            logging.error("Invalid timestamps format. Use a file path or comma-separated list of numbers.")
            return 1
    
    if not timestamps:
        logging.error("No valid timestamps provided")
        return 1
    
    # Create summary video
    output_path = create_summary_video(
        video_path=args.video,
        timestamps=timestamps,
        output_path=args.output,
        segment_duration=args.duration,
        temp_dir=args.temp_dir,
        with_subtitles=not args.no_subtitles,
        cleanup=not args.keep_temp
    )
    
    if output_path:
        logging.info(f"Summary video created at: {output_path}")
        return 0
    else:
        logging.error("Failed to create summary video")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())