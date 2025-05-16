# LLM Video Summarization

## Features

- Automatic speech-to-text transcription (using Whisper)
- Word-level alignment with timestamps (using Montreal Forced Aligner)
- Multi-model extractive summarization:
  - OpenAI GPT-4.5
  - OpenAI GPT-3.5
  - Meta LLaMA-3.2-3B
- Consistency verification procedure to assess summarization reliability
- Automatic generation of summary videos with subtitles

## Installation

1. Clone the repository.
```bash
git clone https://github.com/anonymous2024-gpu/behaviour-aware-video-summarization.git
cd behaviour-aware-video-summarization\LLM-summarization
```

2. Create a virtual environment and install dependencies:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Install external dependencies:
   - [Montreal Forced Aligner (MFA)](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html)
   - FFmpeg: `sudo apt-get install ffmpeg` (Ubuntu/Debian) or download from [ffmpeg.org](https://ffmpeg.org/download.html)

4. Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

5. Login to Hugging Face to access LLaMA model:
```
huggingface-cli login
```
When prompted, enter your Hugging Face token. You need access rights to "meta-llama/Llama-3.2-3B" model.

## Usage

### Basic Usage

Run the entire pipeline on a video:

```bash
python -m video_summarizer --video_dir /path/to/videos --output_dir /path/to/output --mfa_dir /path/to/mfa_output
```

### Step-by-Step Processing

1. Process transcripts and extract timestamps:
```bash
python -m video_summarizer.transcript_processor --video_dir /path/to/videos
```

2. Generate summaries using multiple LLM models:
```bash
python -m video_summarizer.summary_generator --transcript_dir /path/to/transcripts --models all
```

3. Perform consistency verification (for a subset of videos):
```bash
python -m video_summarizer.summary_generator --transcript_dir /path/to/transcripts --verify --num_videos 50
```

4. Create summary videos:
```bash
python -m video_summarizer.video_generator --video_dir /path/to/videos --summary_dir /path/to/summaries
```

### Model Options

You can specify which models to use for summarization:

```bash
# Use all supported models
python -m video_summarizer.summary_generator --transcript_dir /path/to/transcripts --models all

# Use specific models
python -m video_summarizer.summary_generator --transcript_dir /path/to/transcripts --models gpt-4.5,gpt-3.5

# Use only one model
python -m video_summarizer.summary_generator --transcript_dir /path/to/transcripts --models llama-3
```

### Consistency Verification

The consistency verification procedure helps evaluate the reliability of the summaries:

```bash
python -m video_summarizer.summary_generator --transcript_dir /path/to/transcripts --verify --num_videos 10
```

You can adjust the number of videos used for verification with the `--num_videos` parameter.

## Configuration

You can customize the behavior by modifying the `config.py` file or by setting environment variables:

- `VIDEO_FOLDER`: Directory containing input videos
- `MFA_OUTPUT_DIR`: Directory containing Montreal Forced Aligner output
- `OUTPUT_DIR`: Directory for output files
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: OpenAI model to use (default: "gpt-4.5-preview")
- `OPENAI_TEMPERATURE`: Temperature for the API (default: 1.0)

## Project Structure

```
video_summarizer/
├── __init__.py
├── __main__.py               # Entry point for the module
├── config.py                 # Configuration settings
├── transcript_processor.py   # Transcript processing module
├── summary_generator.py      # LLM summary generation module
├── video_generator.py        # Summary video creation module
└── utils.py                  # Utility functions

data/
├── videos/                   # Input videos
├── audio/                    # Extracted audio and transcripts
├── mfa_output/               # Montreal Forced Aligner output
├── summaries/                # Generated summaries
└── temp/                     # Temporary files
```

## Using LLaMA-3.2-3B Model

The tool uses Hugging Face's transformers library to load the LLaMA model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
```

This requires authentication with Hugging Face and access rights to the model. Make sure to run `huggingface-cli login` before using the LLaMA model.

## License

This project is licensed under the MIT License - see the LICENSE file for details.