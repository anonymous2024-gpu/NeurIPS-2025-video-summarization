# Multimodal Video Summarization Framework

This repository contains an implementation of a multimodal video summarization framework that utilizes visual, audio, and text features to generate concise video summaries.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/anonymous2024-gpu/behaviour-aware-video-summarization.git
cd behaviour-aware-video-summarization
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt
```

3. Install FFmpeg (required for video processing):
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows: Download from https://ffmpeg.org/download.html and add to PATH
```

## Usage

### Generate Video Summary

To summarize a video and create a summary video:

```bash
python main.py summarize --video path/to/video.mp4 --video-output summary.mp4
```

Options:
- `--video`: Path to the video file (required)
- `--audio`: Path to the audio file (optional, extracted from video if not provided)
- `--text`: Path to the transcript file (optional, auto-generated if not provided)
- `--timestamps-output`: Path to save timestamps file (optional)
- `--video-output`: Path to save summary video (optional)
- `--coverage`: Content coverage threshold (0.0-1.0, default: 0.85)
- `--similarity`: Novelty similarity threshold (0.0-1.0, default: 0.6)
- `--max-frames`: Maximum number of frames to select (default: 30)
- `--segment-duration`: Duration of each segment in seconds (default: 5.0)
- `--no-subtitles`: Disable subtitles in summary video
- `--log`: Path to log file (optional)
- `--device`: Device to run on ('cuda' or 'cpu', default: 'cuda')
- `--no-video`: Disable video features
- `--no-text`: Disable text features
- `--no-audio`: Disable audio features

### Generate Video from Existing Timestamps

If you already have timestamps from a previous run or another source:

```bash
python main.py generate --video path/to/video.mp4 --timestamps path/to/timestamps.txt --output summary.mp4
```

Options:
- `--video`: Path to the video file (required)
- `--timestamps`: Path to the timestamps file (required)
- `--output`: Path to the output summary video (required)
- `--segment-duration`: Duration of each segment in seconds (default: 5.0)
- `--no-subtitles`: Disable subtitles in summary video
- `--log`: Path to log file (optional)

### Run Ablation Study

To perform ablation studies on multiple videos:

```bash
python main.py ablation --summary_dir path/to/summaries --audio_dir path/to/audio --output_dir path/to/results
```

Options:
- `--summary_dir`: Directory containing summary videos (required)
- `--audio_dir`: Directory containing audio files and transcriptions (required)
- `--output_dir`: Directory to save results (required)
- `--log_file`: Path to log file (optional)
- `--device`: Device to run on ('cuda' or 'cpu', default: 'cuda')
- `--max_videos`: Maximum number of videos to process (optional)

## Project Structure

- `models.py`: Neural network models for feature encoding and summary generation
- `feature_extraction.py`: Functions for extracting features from different modalities
- `evaluation.py`: Functions for evaluating summary quality
- `summarize.py`: Original summarization script (timestamps only)
- `summarize_enhanced.py`: Enhanced version with video generation capabilities
- `video_generator.py`: Utilities for creating summary videos using FFmpeg
- `ablation.py`: Script for conducting ablation studies
- `main.py`: Original command-line interface
- `main_enhanced.py`: Enhanced command-line interface with video generation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
