"""Configuration settings"""

import os

class Config:
    def __init__(self):
        # Default directories
        self.video_folder = os.environ.get("VIDEO_FOLDER", "./data/videos")
        self.mfa_output_dir = os.environ.get("MFA_OUTPUT_DIR", "./data/mfa_output")
        self.audio_dir = os.path.join(self.video_folder, "audio")
        self.output_dir = os.environ.get("OUTPUT_DIR", "./data/summary_videos")
        self.temp_dir_base = os.environ.get("TEMP_DIR", "./data/temp")
        
        # OpenAI settings
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4.5-preview")
        self.max_tokens = int(os.environ.get("OPENAI_MAX_TOKENS", "1024"))
        self.temperature = float(os.environ.get("OPENAI_TEMPERATURE", "1.0"))
        
        # Processing settings
        self.min_match_ratio = float(os.environ.get("MIN_MATCH_RATIO", "0.3"))
        self.debug = os.environ.get("DEBUG", "False").lower() in ('true', '1', 't')
