import os
import torch
import numpy as np
import argparse
import logging
from models import VideoEncoder, TextEncoder, AudioEncoder, CrossModalAttention, SummaryDecoder, ProjectionLayer
from feature_extraction import (
    extract_clip_features, extract_face_emotion_features, extract_significant_visual_cues,
    extract_hubert_features, extract_prosodic_features, extract_audio_features,
    extract_text_features, transcribe_video, read_text_file
)

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

def autoregressive_summary_selection(multimodal_features, timestamps, clip_features=None, coverage_threshold=0.85, similarity_threshold=0.6, max_frames=30, device='cuda'):
    """
    Performs CLIP-derived autoregressive selection of summary segments.
    Selection continues until content coverage reaches a threshold or maximum frames are selected.
    Each selection conditions on all previous selections for better temporal coherence.
    
    Args:
        multimodal_features: Tensor of shape [1, seq_len, embed_dim]
        timestamps: List of timestamps corresponding to the sequence
        clip_features: CLIP visual features for frames (if None, will use multimodal_features)
        coverage_threshold: Stop when this much of the content is covered (0.0-1.0)
        similarity_threshold: Minimum similarity for a frame to be considered novel (0.0-1.0)
        max_frames: Maximum number of frames to select (safety limit)
        device: Device to run computations on
        
    Returns:
        List of selected timestamp indices in temporal order
    """
    device = torch.device(device)
    
    # If no specific CLIP features provided, use the multimodal features
    if clip_features is None:
        clip_features = multimodal_features
    
    # Initialize with SOS token
    s_decoder = SummaryDecoder(d_model=multimodal_features.size(-1), nhead=8, num_layers=6).to(device)
    sos_token = torch.zeros(1, 1, multimodal_features.size(-1)).to(device)
    summary_sequence = sos_token
    
    # Selected segments tracker
    selected_indices = []
    selected_features = []
    
    # Keep track of content coverage
    num_frames = clip_features.size(1)
    coverage_map = torch.zeros(num_frames).to(device)
    
    # Generate summary segments autoregressively until reaching coverage threshold
    step = 0
    current_coverage = 0.0
    
    while current_coverage < coverage_threshold and step < max_frames:
        logging.info(f"Step {step}: Current coverage: {current_coverage:.4f}")
        
        # Create target mask for autoregressive generation
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(summary_sequence.shape[1]).to(device)
        
        # Generate next embedding conditioned on previous selections
        next_output = s_decoder(multimodal_features, summary_sequence, tgt_mask=tgt_mask)
        next_embedding = next_output[:, -1:, :]  # Take the most recent prediction
        
        # Compute similarity with all frames using CLIP embeddings
        similarity_scores = torch.nn.functional.cosine_similarity(
            next_embedding, 
            clip_features,
            dim=2
        ).squeeze(0)
        
        # Apply penalties to already selected indices to avoid repetition
        for idx in selected_indices:
            similarity_scores[idx] -= 0.5
        
        # Get the best match
        best_idx = torch.argmax(similarity_scores).item()
        best_score = similarity_scores[best_idx].item()
        
        # Check if the best match is good enough
        if step > 0 and best_score < similarity_threshold:
            logging.info(f"Stopping: best similarity {best_score:.4f} below threshold {similarity_threshold:.4f}")
            break
            
        selected_indices.append(best_idx)
        
        # Add the selected feature to the conditioning context
        selected_feature = multimodal_features[:, best_idx:best_idx+1, :]
        selected_features.append(selected_feature)
        
        # Update summary sequence for next iteration
        summary_sequence = torch.cat([summary_sequence, selected_feature], dim=1)
        
        # Update coverage map - mark frames similar to the selected one as "covered"
        # Calculate similarity of the selected frame to all other frames
        selected_clip_feature = clip_features[:, best_idx:best_idx+1, :]
        frame_similarities = torch.nn.functional.cosine_similarity(
            selected_clip_feature, 
            clip_features,
            dim=2
        ).squeeze(0)
        
        # Frames with similarity > threshold are considered covered
        newly_covered = (frame_similarities > 0.8) & (coverage_map < 0.5)
        coverage_map[newly_covered] = 1.0
        
        # Update coverage percentage
        current_coverage = coverage_map.sum().item() / num_frames
        
        step += 1
    
    logging.info(f"Selection complete: {step} frames selected, coverage: {current_coverage:.4f}")
    
    # If no frames were selected, select at least one (the most central/representative frame)
    if not selected_indices:
        logging.warning("No frames selected, choosing the most representative frame")
        # Find the frame with highest average similarity to all others
        all_similarities = torch.zeros(num_frames, num_frames).to(device)
        for i in range(num_frames):
            all_similarities[i] = torch.nn.functional.cosine_similarity(
                clip_features[:, i:i+1, :], clip_features, dim=2
            ).squeeze(0)
        
        avg_similarities = all_similarities.mean(dim=1)
        most_representative = torch.argmax(avg_similarities).item()
        selected_indices = [most_representative]
    
    # Map indices to timestamps and sort chronologically
    selected_timestamps = sorted([timestamps[idx] for idx in selected_indices])
    
    # Get the indices in temporal order
    temporal_indices = [timestamps.index(ts) for ts in selected_timestamps]
    
    return temporal_indices

def video_summarizer(video_path, audio_path=None, transcription_text=None, config=None, 
                coverage_threshold=0.85, similarity_threshold=0.6, max_frames=30, device=None):
    """
    Generate a video summary based on multimodal features.
    Uses CLIP-derived autoregressive segment selection to create a temporally coherent summary.
    
    Args:
        video_path: Path to the video file
        audio_path: Path to the audio file (optional)
        transcription_text: Transcription of the video (optional, will be generated if not provided)
        config: Configuration dictionary for ablation study
        coverage_threshold: Coverage threshold for stopping criteria (0.0-1.0) 
        similarity_threshold: Minimum similarity for frames to be considered novel (0.0-1.0)
        max_frames: Maximum number of frames to select (safety limit)
        device: Device to run computations on
        
    Returns:
        List of selected timestamps for the summary
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Default configuration with all features enabled
    if config is None:
        config = {
            "use_video": True, "use_text": True, "use_audio": True,
            "use_clip": True, "use_face_emotion": True,
            "use_hubert": True, "use_prosodic": True,
            "use_video_encoder": True, "use_text_encoder": True, "use_audio_encoder": True,
            "use_cross_attention": True
        }
    
    logging.info(f"Running video summarizer with config: {config}")
    
    # Step 1: Extract raw features based on config
    visual_features, timestamps = None, None
    text_features = None
    audio_features = None
    clip_features = None
    
    # Process video features if enabled
    if config['use_video']:
        # Always extract CLIP features for frame selection (even if we use other visual features)
        if config['use_clip']:
            clip_features, timestamps = extract_clip_features(video_path, device=device)
        
        if config['use_clip'] and config['use_face_emotion']:
            visual_features, timestamps = extract_significant_visual_cues(video_path, device=device)
        elif config['use_clip']:
            visual_features, timestamps = clip_features, timestamps
        elif config['use_face_emotion']:
            visual_features, timestamps = extract_face_emotion_features(video_path, device=device)
    
    # Process text features if enabled
    if config['use_text']:
        # If transcription is not provided, generate it
        if transcription_text is None and os.path.exists(video_path):
            transcription_result = transcribe_video(video_path)
            transcription_text = " ".join(seg['text'].strip() for seg in transcription_result['segments'] if seg['text'].strip())
        
        if transcription_text:
            text_features = extract_text_features(transcription_text, device=device)
    
    # Process audio features if enabled and audio path is provided
    if config['use_audio'] and audio_path and os.path.exists(audio_path):
        audio_features = extract_audio_features(
            audio_path, 
            use_hubert=config['use_hubert'], 
            use_prosodic=config['use_prosodic'],
            device=device
        )
    
    # Check if we have any features to work with
    if visual_features is None and text_features is None and audio_features is None:
        logging.error("Error: No features could be extracted based on configuration and inputs")
        return []
    
    # Step 2: Apply encoders from models.py
    encoded_visual = None
    encoded_text = None
    encoded_audio = None
    
    # Apply VideoEncoder if enabled
    if visual_features is not None:
        # Ensure visual features have batch dimension
        if visual_features.dim() == 2:
            visual_features = visual_features.unsqueeze(0)  # [batch, seq_len, dim]
        
        if config['use_video_encoder']:
            # Project to common dimension if needed
            input_dim = visual_features.size(-1)
            if input_dim != 1024:
                projection = ProjectionLayer(input_dim, 1024).to(device)
                visual_features = projection(visual_features)
            
            # Apply VideoEncoder
            video_encoder = VideoEncoder(
                embed_dim=1024, 
                num_layers=6, 
                num_heads=8, 
                ff_dim=2048
            ).to(device)
            encoded_visual = video_encoder(visual_features)
        else:
            # Project to common dimension without encoding
            input_dim = visual_features.size(-1)
            if input_dim != 1024:
                projection = ProjectionLayer(input_dim, 1024).to(device)
                encoded_visual = projection(visual_features)
            else:
                encoded_visual = visual_features
    
    # Apply TextEncoder if enabled
    if text_features is not None:
        # Ensure text features have batch dimension
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(0)  # [batch, seq_len, dim]
        
        if config['use_text_encoder']:
            # Project to common dimension if needed
            input_dim = text_features.size(-1)
            if input_dim != 1024:
                projection = ProjectionLayer(input_dim, 1024).to(device)
                text_features = projection(text_features)
            
            # Apply TextEncoder
            text_encoder = TextEncoder(
                embed_dim=1024,
                num_layers=6,
                num_heads=8,
                ff_dim=2048
            ).to(device)
            encoded_text = text_encoder(text_features)
        else:
            # Project to common dimension without encoding
            input_dim = text_features.size(-1)
            if input_dim != 1024:
                projection = ProjectionLayer(input_dim, 1024).to(device)
                encoded_text = projection(text_features)
            else:
                encoded_text = text_features
    
    # Apply AudioEncoder if enabled
    if audio_features is not None:
        # Ensure audio features have batch dimension
        if audio_features.dim() == 1:
            audio_features = audio_features.unsqueeze(0).unsqueeze(0)  # [batch, seq_len, dim]
        elif audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)  # [batch, seq_len, dim]
        
        if config['use_audio_encoder']:
            # Project to common dimension if needed
            input_dim = audio_features.size(-1)
            if input_dim != 1024:
                projection = ProjectionLayer(input_dim, 1024).to(device)
                audio_features = projection(audio_features)
            
            # Apply AudioEncoder
            audio_encoder = AudioEncoder(
                embed_dim=1024,
                num_layers=6,
                num_heads=8,
                ff_dim=2048
            ).to(device)
            encoded_audio = audio_encoder(audio_features)
        else:
            # Project to common dimension without encoding
            input_dim = audio_features.size(-1)
            if input_dim != 1024:
                projection = ProjectionLayer(input_dim, 1024).to(device)
                encoded_audio = projection(audio_features)
            else:
                encoded_audio = audio_features
    
    # Step 3: Align sequence lengths for all modalities
    ref_seq_len = 4  # Default sequence length
    if encoded_visual is not None:
        ref_seq_len = encoded_visual.size(1)
    elif encoded_text is not None:
        ref_seq_len = encoded_text.size(1)
    elif encoded_audio is not None:
        ref_seq_len = encoded_audio.size(1)
    
    # Align sequence lengths
    def align_sequence_length(features, target_length):
        if features is None:
            return None
            
        current_length = features.size(1)
        if current_length == target_length:
            return features
        elif current_length > target_length:
            return features[:, :target_length, :]
        else:
            # Repeat sequence to match target length
            repeats = (target_length + current_length - 1) // current_length
            return features.repeat(1, repeats, 1)[:, :target_length, :]
    
    # Align all feature sequences
    encoded_visual = align_sequence_length(encoded_visual, ref_seq_len)
    encoded_text = align_sequence_length(encoded_text, ref_seq_len)
    encoded_audio = align_sequence_length(encoded_audio, ref_seq_len)
    if clip_features is not None:
        # Ensure clip features match the multimodal features sequence length
        if clip_features.dim() == 2:
            clip_features = clip_features.unsqueeze(0)
        clip_features = align_sequence_length(clip_features, ref_seq_len)
    
    # Step 4: Apply Cross-Modal Attention if enabled
    if config['use_cross_attention'] and ((encoded_visual is not None and encoded_text is not None) or 
                                        (encoded_visual is not None and encoded_audio is not None) or
                                        (encoded_text is not None and encoded_audio is not None)):
        cm_attention = CrossModalAttention(d_model=1024, nhead=8).to(device)
        
        # Select query based on available modalities (prioritize visual)
        if encoded_visual is not None:
            multimodal_features = cm_attention(encoded_visual, encoded_text, encoded_audio)
        elif encoded_text is not None:
            multimodal_features = cm_attention(encoded_text, None, encoded_audio)
        elif encoded_audio is not None:
            multimodal_features = cm_attention(encoded_audio, encoded_text, None)
        else:
            raise ValueError("At least one modality must be provided for cross-modal attention")
    else:
        # Combine available features by concatenation and projection
        available_features = []
        if encoded_visual is not None:
            available_features.append(encoded_visual)
        if encoded_text is not None:
            available_features.append(encoded_text)
        if encoded_audio is not None:
            available_features.append(encoded_audio)
        
        if len(available_features) > 1:
            # Concatenate along feature dimension
            multimodal_features = torch.cat(available_features, dim=-1)
            
            # Project to common dimension
            multimodal_projection = ProjectionLayer(multimodal_features.size(-1), 1024).to(device)
            multimodal_features = multimodal_projection(multimodal_features)
        elif len(available_features) == 1:
            multimodal_features = available_features[0]
        else:
            logging.error("Error: No features available for summary generation")
            return []
    
    # Step 5: Select summary segments using CLIP-derived autoregressive approach
    if timestamps is None or len(timestamps) == 0:
        logging.warning("No timestamps available, returning empty summary")
        return []
    
    # Ensure clip features are properly formatted for similarity computation
    if clip_features is None:
        clip_features = multimodal_features
    
    # Extract summary segments using CLIP-derived autoregressive selection
    selected_indices = autoregressive_summary_selection(
        multimodal_features=multimodal_features,
        timestamps=timestamps,
        clip_features=clip_features,
        coverage_threshold=coverage_threshold,
        similarity_threshold=similarity_threshold,
        max_frames=max_frames,
        device=device
    )
    
    # Map to timestamps
    summary_timestamps = [timestamps[i] for i in selected_indices]
    
    return sorted(list(set(summary_timestamps)))

def main():
    parser = argparse.ArgumentParser(description='Video Summarization Tool')
    parser.add_argument('--video', required=True, help='Path to the video file')
    parser.add_argument('--audio', help='Path to the audio file (optional)')
    parser.add_argument('--text', help='Path to the text transcription file (optional)')
    parser.add_argument('--output', help='Path to output summary timestamps file')
    parser.add_argument('--coverage', type=float, default=0.85, help='Content coverage threshold (0.0-1.0)')
    parser.add_argument('--similarity', type=float, default=0.6, help='Minimum novelty similarity threshold (0.0-1.0)')
    parser.add_argument('--max-frames', type=int, default=30, help='Maximum number of frames to select')
    parser.add_argument('--log', help='Path to log file')
    parser.add_argument('--device', default='cuda', help='Device to run on (cuda/cpu)')
    parser.add_argument('--no-video', action='store_true', help='Disable video features')
    parser.add_argument('--no-text', action='store_true', help='Disable text features')
    parser.add_argument('--no-audio', action='store_true', help='Disable audio features')
    
    args = parser.parse_args()
    
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
    
    # Run summarization
    logger.info(f"Running summarization for video: {args.video}")
    summary_timestamps = video_summarizer(
        video_path=args.video,
        audio_path=args.audio,
        transcription_text=transcription_text,
        config=config,
        coverage_threshold=args.coverage,
        similarity_threshold=args.similarity,
        max_frames=args.max_frames,
        device=args.device
    )
    
    logger.info(f"Generated {len(summary_timestamps)} summary segments")
    
    # Save output if requested
    if args.output:
        with open(args.output, 'w') as f:
            for ts in summary_timestamps:
                f.write(f"{ts:.2f}\n")
        logger.info(f"Summary timestamps saved to {args.output}")
    
    # Print timestamps
    print("Summary timestamps (seconds):")
    for ts in summary_timestamps:
        print(f"{ts:.2f}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())