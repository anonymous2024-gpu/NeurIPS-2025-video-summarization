#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import logging
from summarize import summarize_and_create_video, setup_logging
from ablation import run_ablation_study, generate_ablation_tables
from feature_extraction import read_text_file
from collections import defaultdict
import json
import numpy as np
import torch

def main():
    parser = argparse.ArgumentParser(description='Multimodal Video Summarization Framework')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Parser for summarize command
    summarize_parser = subparsers.add_parser('summarize', help='Generate video summary')
    summarize_parser.add_argument('--video', required=True, help='Path to the video file')
    summarize_parser.add_argument('--audio', help='Path to the audio file (optional)')
    summarize_parser.add_argument('--text', help='Path to the text transcription file (optional)')
    summarize_parser.add_argument('--timestamps-output', help='Path to output summary timestamps file')
    summarize_parser.add_argument('--video-output', help='Path to output summary video file')
    summarize_parser.add_argument('--coverage', type=float, default=0.85, help='Content coverage threshold (0.0-1.0)')
    summarize_parser.add_argument('--similarity', type=float, default=0.6, help='Minimum novelty similarity threshold (0.0-1.0)')
    summarize_parser.add_argument('--max-frames', type=int, default=30, help='Maximum number of frames to select')
    summarize_parser.add_argument('--segment-duration', type=float, default=5.0, help='Duration of each segment in seconds')
    summarize_parser.add_argument('--no-subtitles', action='store_true', help='Disable subtitles in summary video')
    summarize_parser.add_argument('--log', help='Path to log file')
    summarize_parser.add_argument('--device', default='cuda', help='Device to run on (cuda/cpu)')
    summarize_parser.add_argument('--no-video', action='store_true', help='Disable video features')
    summarize_parser.add_argument('--no-text', action='store_true', help='Disable text features')
    summarize_parser.add_argument('--no-audio', action='store_true', help='Disable audio features')
    
    # Parser for video generation from timestamps
    generate_parser = subparsers.add_parser('generate', help='Generate summary video from timestamps')
    generate_parser.add_argument('--video', required=True, help='Path to the input video file')
    generate_parser.add_argument('--timestamps', required=True, help='Path to the timestamps file')
    generate_parser.add_argument('--output', required=True, help='Path to the output summary video')
    generate_parser.add_argument('--segment-duration', type=float, default=5.0, help='Duration of each segment in seconds')
    generate_parser.add_argument('--no-subtitles', action='store_true', help='Disable subtitles in summary video')
    generate_parser.add_argument('--log', help='Path to log file')
    
    # Parser for ablation command
    ablation_parser = subparsers.add_parser('ablation', help='Run ablation study')
    ablation_parser.add_argument('--summary_dir', required=True, help='Directory containing summary videos')
    ablation_parser.add_argument('--audio_dir', required=True, help='Directory containing audio files and text transcriptions')
    ablation_parser.add_argument('--output_dir', required=True, help='Directory to save results')
    ablation_parser.add_argument('--log_file', help='Path to log file')
    ablation_parser.add_argument('--device', default='cuda', help='Device to run on (cuda/cpu)')
    ablation_parser.add_argument('--max_videos', type=int, default=None, help='Maximum number of videos to process')
    
    args = parser.parse_args()
    
    if args.command == 'summarize':
        run_summarize(args)
    elif args.command == 'generate':
        run_generate(args)
    elif args.command == 'ablation':
        run_ablation(args)
    else:
        parser.print_help()

def run_summarize(args):
    # Setup logging
    logger = setup_logging(args.log)
    
    # Create configuration based on arguments
    config = {
        "use_video": not args.no_video,
        "use_text": not args.no_text,
        "use_audio": not args.no_audio,
        "use_clip": True,
        "use_face_emotion": True,
        "use_hubert": True,
        "use_prosodic": True,
        "use_video_encoder": True,
        "use_text_encoder": True,
        "use_audio_encoder": True,
        "use_cross_attention": True
    }
    
    # Read transcription if provided
    transcription_text = None
    if args.text and os.path.exists(args.text):
        transcription_text = read_text_file(args.text)
    
    # Run summarization and create video
    logger.info(f"Running summarization for video: {args.video}")
    summary_timestamps, summary_video_path = summarize_and_create_video(
        video_path=args.video,
        audio_path=args.audio,
        transcription_text=transcription_text,
        timestamps_output=args.timestamps_output,
        video_output=args.video_output,
        config=config,
        coverage_threshold=args.coverage,
        similarity_threshold=args.similarity,
        max_frames=args.max_frames,
        segment_duration=args.segment_duration,
        with_subtitles=not args.no_subtitles,
        device=args.device
    )
    
    if not summary_timestamps:
        logger.error("Failed to generate summary")
        return 1
    
    # Print timestamps
    print("Summary timestamps (seconds):")
    for ts in summary_timestamps:
        print(f"{ts:.2f}")
    
    if summary_video_path:
        print(f"Summary video created at: {summary_video_path}")
    
    return 0

def run_generate(args):
    # Setup logging
    logger = setup_logging(args.log)
    
    # Import video_generator module
    from video_generator import read_timestamps_file, create_summary_video
    
    # Read timestamps from file
    timestamps = read_timestamps_file(args.timestamps)
    if not timestamps:
        logger.error(f"No valid timestamps found in {args.timestamps}")
        return 1
    
    # Create summary video
    logger.info(f"Generating summary video from timestamps in {args.timestamps}")
    summary_video_path = create_summary_video(
        video_path=args.video,
        timestamps=timestamps,
        output_path=args.output,
        segment_duration=args.segment_duration,
        with_subtitles=not args.no_subtitles
    )
    
    if summary_video_path:
        logger.info(f"Summary video created at: {summary_video_path}")
        return 0
    else:
        logger.error("Failed to create summary video")
        return 1

def run_ablation(args):
    # Setup logging
    if args.log_file is None:
        args.log_file = os.path.join(args.output_dir, "ablation_study.log")
    
    logger = setup_logging(args.log_file)
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of videos (subdirectories in summary_dir)
    video_names = [name for name in os.listdir(args.summary_dir)
                   if os.path.isdir(os.path.join(args.summary_dir, name))]
    
    if args.max_videos is not None:
        video_names = video_names[:args.max_videos]
    
    logger.info(f"Found {len(video_names)} videos to process")
    
    # Run ablation for selected videos
    all_results = {}
    for video_name in video_names:
        logger.info(f"Processing video: {video_name}")
        device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
        
        try:
            result = run_ablation_study(video_name, args.summary_dir, args.audio_dir, device=device)
            all_results[video_name] = result
            
            # Save individual result
            with open(os.path.join(args.output_dir, f"{video_name}_ablation.json"), 'w') as f:
                json.dump(result, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error processing video {video_name}: {str(e)}")
    
    # Compute average results
    avg_results = {}
    metric_sums = defaultdict(lambda: defaultdict(float))
    metric_counts = defaultdict(lambda: defaultdict(int))
    
    for video_name, video_results in all_results.items():
        if isinstance(video_results, dict) and "error" in video_results:
            continue
        
        for config_name, metrics in video_results.items():
            if isinstance(metrics, dict) and "error" not in metrics:
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        metric_sums[config_name][metric_name] += value
                        metric_counts[config_name][metric_name] += 1
    
    # Calculate averages
    for config_name, metrics in metric_sums.items():
        avg_results[config_name] = {}
        for metric_name, total in metrics.items():
            count = metric_counts[config_name][metric_name]
            if count > 0:
                avg_results[config_name][metric_name] = total / count
    
    # Save average results
    with open(os.path.join(args.output_dir, "average_results.json"), 'w') as f:
        json.dump(avg_results, f, indent=2)
    
    # Generate tables
    generate_ablation_tables(avg_results, os.path.join(args.output_dir, "ablation_scores.md"))
    
    logger.info("Ablation study complete!")

if __name__ == "__main__":
    import sys
    sys.exit(main())