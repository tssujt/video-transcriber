# A Video Transcriber Demo Based on Streamlit

## Supported transcription engines

- Azure OpenAI Whisper
- Volcengine
- Groq Whisper

## How to run locally

1. Install ffmpeg

```bash
brew install ffmpeg
# or
sudo apt install ffmpeg
```

2. Install Python dependencies

```bash
python3 -m pip install -r requirements.txt
```

3. Set environment variables

```bash
export AZURE_OPENAI_API_KEY=
export AZURE_OPENAI_ENDPOINT=
export VOLCENGINE_ACCESS_TOKEN=
export VOLCENGINE_APPID=
export GROQ_API_KEY=
```

4. Run Streamlit Server

```bash
streamlit run streamlit_app.py --server.maxUploadSize 20000
```
