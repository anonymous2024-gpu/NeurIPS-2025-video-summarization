import os
import torch
import cv2
import numpy as np
import logging
from transformers import CLIPProcessor, CLIPModel, AutoFeatureExtractor, AutoModel
from sentence_transformers import SentenceTransformer
import whisper
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
import opensmile
from scipy.io import wavfile
from torch import nn
import mediapipe as mp
from deepface import DeepFace

def read_text_file(file_path):
    """Read text from a file"""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if text:
                return text
            logging.warning(f"File {file_path} is empty")
            return ""
    logging.error(f"File {file_path} not found")
    return ""

def extract_clip_features(video_path, fps=1, device='cuda'):
    """Extract raw CLIP visual features from video without any further encoding"""
    device = torch.device(device)
    # Load CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None, None

    frame_embeddings, timestamps = [], []
    frame_idx, frame_interval = 0, int(cap.get(cv2.CAP_PROP_FPS) / fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # CLIP embedding
            try:
                image_input = clip_processor(images=frame_rgb, return_tensors="pt").to(device)
                with torch.no_grad():
                    frame_embedding = clip_model.get_image_features(**image_input).squeeze().to(device)
                frame_embeddings.append(frame_embedding)
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            except Exception as e:
                print(f"Error in CLIP embedding: {e}")

        frame_idx += 1

    cap.release()

    if not frame_embeddings:
        print("No frames were processed successfully")
        return None, None

    # Stack all the embeddings
    frame_embeddings = torch.stack(frame_embeddings).to(device)
    print(f"CLIP features shape: {frame_embeddings.shape}")
    
    return frame_embeddings, timestamps

def extract_face_emotion_features(video_path, fps=1, device='cuda'):
    """Extract face and emotion features without any transformer encoding"""
    device = torch.device(device)
    # Initialize MediaPipe Face Mesh for head pose estimation
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        face_mesh.close()
        print(f"Error: Cannot open video file {video_path}")
        return None, None

    head_poses, emotion_embeddings, timestamps = [], [], []
    frame_idx, frame_interval = 0, int(cap.get(cv2.CAP_PROP_FPS) / fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Head pose estimation (MediaPipe)
            head_pose = np.zeros(3)
            try:
                results = face_mesh.process(frame_rgb)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    nose_tip = landmarks[1]
                    left_eye = landmarks[33]
                    right_eye = landmarks[263]
                    yaw = np.arctan2(right_eye.x - left_eye.x, right_eye.z - left_eye.z)
                    pitch = np.arctan2(nose_tip.y - (left_eye.y + right_eye.y) / 2, nose_tip.z - (left_eye.z + right_eye.z) / 2)
                    roll = np.arctan2(left_eye.y - right_eye.y, left_eye.x - right_eye.x)
                    head_pose = np.array([yaw, pitch, roll], dtype=np.float32)
                head_poses.append(torch.tensor(head_pose, dtype=torch.float32).to(device))
            except Exception as e:
                print(f"Error in head pose estimation: {e}")
                head_poses.append(torch.zeros(3).to(device))

            # Emotion embedding (DeepFace)
            try:
                # Try different parameter combinations for different DeepFace versions
                try:
                    analysis = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False, prog_bar=False)
                except TypeError:
                    # Remove prog_bar parameter if it's not supported
                    analysis = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)
                
                # Handle different return formats
                if isinstance(analysis, list):
                    emotion_probs = analysis[0]['emotion']
                else:
                    emotion_probs = analysis['emotion']
                
                emotion_vector = np.array([emotion_probs.get(emo, 0.0) for emo in 
                                        ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']], 
                                        dtype=np.float32)
                emotion_embeddings.append(torch.tensor(emotion_vector).to(device))
            except Exception as e:
                print(f"Error in emotion embedding: {e}")
                emotion_embeddings.append(torch.zeros(7).to(device))

            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

        frame_idx += 1

    cap.release()
    face_mesh.close()

    if not head_poses or not emotion_embeddings:
        print("No frames were processed successfully")
        return None, None

    # Stack all the embeddings
    head_poses = torch.stack(head_poses).to(device)
    emotion_embeddings = torch.stack(emotion_embeddings).to(device)
    
    # Combine features
    combined_features = torch.cat([head_poses, emotion_embeddings], dim=-1)
    print(f"Face and emotion features shape: {combined_features.shape}")
    
    return combined_features, timestamps

def extract_significant_visual_cues(video_path, fps=1, device='cuda'):
    """Extract combined visual features (CLIP + face + emotion) without transformer encoding"""
    device = torch.device(device)
    
    # Extract CLIP features
    clip_features, timestamps = extract_clip_features(video_path, fps, device)
    if clip_features is None:
        return None, None
    
    # Extract face and emotion features
    face_emotion_features, face_timestamps = extract_face_emotion_features(video_path, fps, device)
    if face_emotion_features is None:
        # If face features failed, just return CLIP features
        return clip_features, timestamps
        
    # Ensure same number of timestamps (they should be the same, but just in case)
    if len(timestamps) != len(face_timestamps):
        # Use the shorter list
        min_len = min(len(timestamps), len(face_timestamps))
        clip_features = clip_features[:min_len]
        face_emotion_features = face_emotion_features[:min_len]
        timestamps = timestamps[:min_len]
    
    # Combine features
    combined_features = torch.cat([clip_features, face_emotion_features], dim=-1)
    print(f"Combined visual features shape: {combined_features.shape}")
    
    return combined_features, timestamps

def extract_hubert_features(audio_path, device='cuda'):
    """Extract raw HuBERT audio features without any transformer encoding"""
    device = torch.device(device)
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        model = AutoModel.from_pretrained("facebook/hubert-base-ls960").to(device)
        
        sr, waveform = wavfile.read(audio_path)
        waveform = waveform.astype(np.float32) / np.max(np.abs(waveform))
        inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt").to(device)
        
        with torch.no_grad():
            audio_embeddings = model(**inputs).last_hidden_state  # [1, seq_len, 768]
            
        print(f"HuBERT audio features shape: {audio_embeddings.shape}")
        return audio_embeddings
        
    except Exception as e:
        print(f"Error in HuBERT audio processing: {e}")
        return torch.zeros((1, 4, 768)).to(device)  # Use 4 as a safe sequence length

def extract_prosodic_features(audio_path, device='cuda'):
    """Extract prosodic audio features (pitch, energy, etc.)"""
    device = torch.device(device)
    try:
        signal = basic.SignalObj(audio_path)
        
        # Extract pitch using YAAPT
        pitch = pYAAPT.yaapt(signal)
        f0_samp_interp = pitch.samp_interp
        f0_samp_interp = np.nan_to_num(f0_samp_interp, nan=np.nanmedian(f0_samp_interp))
        
        # Extract eGeMAPS features using openSMILE
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        features = smile.process_file(audio_path)
        
        # Extract key prosodic features
        hammarberg = features['hammarbergIndex_sma3'].values.mean()
        loudness = features['Loudness_sma3'].values.mean()
        
        # Combine into a feature vector
        prosodic_features = torch.tensor([
            f0_samp_interp.mean(), f0_samp_interp.std(), 
            hammarberg, loudness
        ], dtype=torch.float32).to(device)  # Shape: [4]
        
        print(f"Prosodic audio features shape: {prosodic_features.shape}")
        return prosodic_features
        
    except Exception as e:
        print(f"Error in prosodic audio processing: {e}")
        return torch.zeros(4).to(device)

def extract_audio_features(audio_path, use_hubert=True, use_prosodic=True, device='cuda'):
    """Extract combined audio features without transformer encoding"""
    features = []
    device = torch.device(device)
    
    if use_hubert:
        hubert_features = extract_hubert_features(audio_path, device)
        if hubert_features is not None:
            # Average across sequence dimension for simplicity
            if hubert_features.dim() == 3:
                hubert_features = hubert_features.mean(dim=1)  # [1, 768]
            features.append(hubert_features)
    
    if use_prosodic:
        prosodic_features = extract_prosodic_features(audio_path, device)
        if prosodic_features is not None:
            # Add batch dimension if missing
            if prosodic_features.dim() == 1:
                prosodic_features = prosodic_features.unsqueeze(0)  # [1, 4]
            features.append(prosodic_features)
    
    if not features:
        return None
    
    # Concatenate all features along the feature dimension
    combined_features = torch.cat(features, dim=-1)
    print(f"Combined audio features shape: {combined_features.shape}")
    
    return combined_features

def extract_text_features(transcription_text, device='cuda'):
    """Extract raw text features without transformer encoding"""
    device = torch.device(device)
    
    try:
        sroberta = SentenceTransformer("sentence-transformers/all-roberta-large-v1").to(device)
        sentences = transcription_text.split(". ")
        sentence_embeddings = sroberta.encode(sentences, convert_to_tensor=True).to(device)  # shape: (seq_len, 1024)
        
        print(f"Text features shape: {sentence_embeddings.shape}")
        return sentence_embeddings
        
    except Exception as e:
        print(f"Error in text processing: {e}")
        # Return a dummy tensor with the expected shape
        return torch.zeros((1, 1024)).to(device)

def transcribe_video(video_path):
    """Transcribe video to text using Whisper"""
    try:
        model = whisper.load_model("medium")
        result = model.transcribe(video_path, word_timestamps=True)
        captions = [(seg['text'].strip(), seg['start'], seg['end']) for seg in result['segments'] if seg['text'].strip()]
        if not captions:
            print("Warning: No valid transcriptions found.")
        print(f"Transcribed {len(captions)} captions: {captions[:7]}")
        return result
    except Exception as e:
        print(f"Error transcribing video: {e}. Proceeding without transcription.")
        return {'segments': []}