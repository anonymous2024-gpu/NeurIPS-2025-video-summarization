import os
import torch
import cv2
import numpy as np
import logging
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import sacrebleu
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from transformers import CLIPProcessor, CLIPModel
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import f1_score

nltk.download('punkt', quiet=True)

def read_text_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if text:
                return text
            logging.warning(f"File {file_path} is empty")
            return ""
    logging.error(f"File {file_path} not found")
    return ""

def compute_length_ratio(reference_text_path, actual_text_path):
    reference_text = read_text_file(reference_text_path)
    actual_text = read_text_file(actual_text_path)
    if not reference_text or not actual_text:
        return None  # If one of the texts is missing, return None
    
    # Tokenize the text to get the word count
    reference_word_count = len(nltk.word_tokenize(reference_text))
    actual_word_count = len(nltk.word_tokenize(actual_text))
    
    # Calculate the length ratio
    if actual_word_count == 0:
        return None  # Avoid division by zero
    length_ratio = reference_word_count / actual_word_count
    return length_ratio

def evaluate_text_metrics(reference_text, hypothesis_text):
    if not reference_text or not hypothesis_text:
        return None
    
    # ROUGE metrics
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge_scorer_obj.score(reference_text, hypothesis_text)
    
    # ROUGE-S (skip-bigram) with max skip=4
    rouge_s = compute_rouge_s(hypothesis_text, reference_text, max_skip=4)
    
    # BLEU score
    bleu_score_value = sacrebleu.corpus_bleu([hypothesis_text], [[reference_text]]).score / 100
    
    # BERTScore
    P, R, F1 = bert_score([hypothesis_text], [reference_text], lang="en", verbose=False)
    
    results = {
        'ROUGE-1': rouge_scores['rouge1'].fmeasure,
        'ROUGE-2': rouge_scores['rouge2'].fmeasure,
        'ROUGE-L': rouge_scores['rougeL'].fmeasure,
        'ROUGE-S': rouge_s,
        'BLEU': bleu_score_value,
        'BERTScore_P': P.item(),
        'BERTScore_R': R.item(),
        'BERTScore_F1': F1.item()
    }
    return results

def compute_rouge_s(hyp, ref, max_skip=4):
    hyp_words = word_tokenize(hyp.lower())
    ref_words = word_tokenize(ref.lower())
    
    if not hyp_words or not ref_words:
        return 0.0
    
    skip_pairs = set()
    for i in range(len(hyp_words)):
        for j in range(i + 1, min(i + max_skip + 1, len(hyp_words))):
            pair = (hyp_words[i], hyp_words[j])
            skip_pairs.add(pair)
    
    ref_pairs = set()
    for i in range(len(ref_words)):
        for j in range(i + 1, min(i + max_skip + 1, len(ref_words))):
            pair = (ref_words[i], ref_words[j])
            ref_pairs.add(pair)
    
    if not skip_pairs or not ref_pairs:
        return 0.0
    
    matches = len(skip_pairs & ref_pairs)
    precision = matches / len(skip_pairs) if skip_pairs else 0.0
    recall = matches / len(ref_pairs) if ref_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def sample_frames(video_path, max_frames=60):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    frames = []
    if duration <= 0:
        cap.release()
        return frames
    
    # If video duration is shorter than max_frames, adjust max_frames
    num_frames_to_sample = min(int(duration), max_frames)
    
    # Sample frames at regular intervals
    step = max(1, frame_count // num_frames_to_sample)
    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            timestamp = i / fps
            frames.append((frame, timestamp))
    
    cap.release()
    return frames

def compute_frame_matching_score(video_path, reference_timestamps, device=None):
    """
    This function computes the frame matching score between predicted and reference frames using CLIP model.
    It uses CUDA if available.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert reference_timestamps to floats if they are strings
    try:
        reference_timestamps = [float(ts) for ts in reference_timestamps]
    except (ValueError, TypeError):
        logging.error("Reference timestamps must be convertible to float")
        return 0.0
        
    # Load CLIP model for frame feature extraction
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    try:
        frames = sample_frames(video_path)  # Get frames from the video
        if not frames:
            logging.warning(f"No frames found in video: {video_path}")
            return 0.0
            
        predicted_embeddings = []
        reference_embeddings = []
        
        # For each reference timestamp, find the closest frame
        for reference_ts in reference_timestamps:
            try:
                # Find the closest timestamp in frames (ensuring both are floats)
                closest_frame_idx = min(range(len(frames)), 
                                      key=lambda i: abs(float(frames[i][1]) - float(reference_ts)))
                reference_frame = frames[closest_frame_idx][0]
                
                # Convert BGR to RGB (OpenCV loads as BGR)
                reference_frame_rgb = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2RGB)
                
                # Process with CLIP processor
                image_input = clip_processor(images=reference_frame_rgb, return_tensors="pt").to(device)
                with torch.no_grad():
                    frame_embedding = clip_model.get_image_features(**image_input).squeeze().cpu().numpy()
                reference_embeddings.append(frame_embedding)
            except Exception as e:
                logging.warning(f"Error processing reference frame at ts {reference_ts}: {e}")
                continue
                
        # For each predicted frame, compute the embedding
        for frame, _ in frames:
            try:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with CLIP processor
                image_input = clip_processor(images=frame_rgb, return_tensors="pt").to(device)
                with torch.no_grad():
                    frame_embedding = clip_model.get_image_features(**image_input).squeeze().cpu().numpy()
                predicted_embeddings.append(frame_embedding)
            except Exception as e:
                logging.warning(f"Error processing predicted frame: {e}")
                continue
                
        # Check if we have enough embeddings to compute score
        if not reference_embeddings or not predicted_embeddings:
            logging.warning("Not enough embeddings to compute frame matching score")
            return 0.0
            
        # Compute similarity scores between predicted and reference frames
        similarity_scores = []
        for pred_embed in predicted_embeddings:
            scores = []
            for ref_embed in reference_embeddings:
                # Compute cosine similarity
                similarity = np.dot(pred_embed, ref_embed) / (np.linalg.norm(pred_embed) * np.linalg.norm(ref_embed))
                scores.append(similarity)
            if scores:
                similarity_scores.append(np.max(scores))
        
        frame_matching_score = np.mean(similarity_scores) if similarity_scores else 0.0
        logging.info(f"Frame Matching Score: {frame_matching_score:.4f}")
        return float(frame_matching_score)  # Convert to Python float for JSON serialization
        
    except Exception as e:
        logging.error(f"Error in compute_frame_matching_score: {e}")
        return 0.0

def compute_clipscore(video_path, reference_sentences, device=None):
    """
    Computes CLIPScore between video frames and reference sentences.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not reference_sentences:
        logging.warning(f"No reference sentences for CLIPScore")
        return 0.0
        
    # Load models
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    try:
        frames = sample_frames(video_path)
        if not frames:
            logging.warning(f"No frames sampled for {video_path}")
            return 0.0
        
        scores = []
        for frame, _ in frames:
            try:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with CLIP processor
                inputs = clip_processor(
                    text=reference_sentences,
                    images=[frame_rgb],
                    return_tensors="pt",
                    padding=True
                ).to(device)
                
                with torch.no_grad():
                    outputs = clip_model(**inputs)
                    
                similarities = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()[0]
                mean_similarity = np.mean(similarities)
                scores.append(mean_similarity)
            except Exception as e:
                logging.warning(f"Error processing frame for CLIPScore: {e}")
                continue
        
        score = np.mean(scores) if scores else 0.0
        logging.info(f"CLIPScore: {score:.4f}")
        return float(score)  # Convert to Python float for JSON serialization
        
    except Exception as e:
        logging.error(f"Error in compute_clipscore: {e}")
        return 0.0

def compute_temporal_consistency(predicted_timestamps, reference_timestamps):
    """Compute Kendall's Tau and Spearman's Rho based on rank correlation."""
    
    # Ensure we have sufficient timestamps
    if len(predicted_timestamps) < 2 or len(reference_timestamps) < 2:
        logging.warning("Insufficient timestamps for temporal consistency")
        return 0.0, 0.0

    penalty = 0.0
    total_timestamps = len(predicted_timestamps) + len(reference_timestamps)
    
    # Find common timestamps or closest matches
    common_predicted = []
    common_reference = []
    
    for p_ts in predicted_timestamps:
        # Find closest reference timestamp
        closest_idx = np.argmin([abs(p_ts - r_ts) for r_ts in reference_timestamps])
        common_predicted.append(p_ts)
        common_reference.append(reference_timestamps[closest_idx])

    unmatched_predicted = len(predicted_timestamps) - len(common_predicted)
    unmatched_reference = len(reference_timestamps) - len(common_reference)

    penalty = (unmatched_predicted + unmatched_reference) / total_timestamps
    
    # Compute Kendall's Tau and Spearman's Rho
    try:
        tau, _ = kendalltau(common_predicted, common_reference)
        rho, _ = spearmanr(common_predicted, common_reference)
        
        tau = tau if not np.isnan(tau) else 0.0
        rho = rho if not np.isnan(rho) else 0.0
        
        adjusted_tau = tau - penalty
        adjusted_rho = rho - penalty
        
        return adjusted_tau, adjusted_rho
    except:
        return 0.0, 0.0

def evaluate_video_metrics(video_path, reference_path, reference_text, device=None):
    """Evaluate video summary using video-based metrics"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract timestamps from videos
    predicted_frames = sample_frames(video_path)
    reference_frames = sample_frames(reference_path) if os.path.exists(reference_path) else []
    
    predicted_timestamps = [float(ts) for _, ts in predicted_frames]
    reference_timestamps = [float(ts) for _, ts in reference_frames]
    
    if not predicted_timestamps or not reference_timestamps:
        logging.warning(f"Insufficient frames in videos for comparison")
        return None
    
    # Compute temporal consistency metrics
    tau, rho = compute_temporal_consistency(predicted_timestamps, reference_timestamps)
    
    # Compute frame-level F1 score
    def create_binary_array(timestamps, total_frames, fps):
        binary = np.zeros(total_frames)
        for ts in timestamps:
            frame_idx = min(int(ts * fps), total_frames - 1)
            binary[frame_idx] = 1
        return binary
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Create binary arrays and compute F1
    pred_binary = create_binary_array(predicted_timestamps, total_frames, fps)
    ref_binary = create_binary_array(reference_timestamps, total_frames, fps)
    
    try:
        f1 = f1_score(ref_binary, pred_binary)
    except:
        f1 = 0.0
    
    # Parse reference text into sentences
    reference_sentences = sent_tokenize(reference_text) if reference_text else []
    
    # Compute CLIP-based metrics - use reference timestamps instead of reference_sentences
    # frame_matching_score = compute_frame_matching_score(video_path, reference_timestamps, device)
    clipscore = compute_clipscore(video_path, reference_sentences, device)
    
    return {
        'F1_Score': float(f1),
        'Kendall_Tau': float(tau),
        'Spearman_Rho': float(rho),
        'CLIPScore': float(clipscore),
    }