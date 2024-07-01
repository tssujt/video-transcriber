import os
import time

import pysrt
import requests
import streamlit as st
from ffmpeg import FFmpeg
from groq import Groq
from openai import AzureOpenAI
from pydub import AudioSegment
from pydub.silence import split_on_silence


INPUT_FILE = "/tmp/input"
OUTPUT_FILE = "/tmp/output.flac"
OUTPUT_SRT_FILE = "/tmp/merged.srt"

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
deployment_id = "whisper"

volcengine_base_url = "https://openspeech.bytedance.com/api/v1/vc"
volcengine_access_token = os.getenv("VOLCENGINE_ACCESS_TOKEN")
volcengine_appid = os.getenv("VOLCENGINE_APPID")


def merge_srt_files(filenames: list[str], output_filename: str) -> None:
    merged_subs = pysrt.SubRipFile()
    for filename in filenames:
        subs = pysrt.open(filename, encoding="utf-8")
        # Update index
        for i, sub in enumerate(subs):
            sub.index = i + len(merged_subs)
        merged_subs.extend(subs)
    merged_subs.save(output_filename, encoding="utf-8")


def volcengine_transcribe(audio_data: bytes, language: str = "zh-CN") -> dict:
    words_per_line_mapping = {
        "zh-CN": 15,
        "en-US": 55,
        "ja-JP": 32,
        "ko-KR": 32,
        "es-MX": 55,
        "ru-RU": 55,
        "fr-FR": 55,
    }
    response = requests.post(
        "{base_url}/submit".format(base_url=volcengine_base_url),
        params=dict(
            appid=volcengine_appid,
            language=language,
            use_itn="True",
            use_capitalize="True",
            max_lines=1,
            words_per_line=words_per_line_mapping[language],
        ),
        data=audio_data,
        headers={
            "content-type": "audio/flac",
            "Authorization": "Bearer; {}".format(volcengine_access_token),
        },
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Success"

    job_id = response.json()["id"]
    response = requests.get(
        "{base_url}/query".format(base_url=volcengine_base_url),
        params=dict(
            appid=volcengine_appid,
            id=job_id,
            blocking=1,
        ),
        headers={"Authorization": "Bearer; {}".format(volcengine_access_token)},
    )
    assert response.json()["message"] == "Success"
    return response.json()


choosed_model = st.selectbox(
    "Use which transcription engine?",
    ("Whisper", "Volcengine", "Groq Whisper"),
)
choosed_language = st.selectbox(
    "What language is the video in?",
    ("zh-CN", "en-US", "ja-JP", "ko-KR", "es-MX", "ru-RU", "fr-FR"),
)


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    video_data = uploaded_file.getvalue()
    with open(INPUT_FILE, "wb") as fi:
        fi.write(video_data)

    stime = time.time()
    with st.status("Converting to audio"):
        ffmpeg = (
            FFmpeg()
            .option("y")
            .input(INPUT_FILE)
            .output(
                OUTPUT_FILE,
                ar=16000,
                ac=1,
                sample_fmt="s16",
                map="0:a:",
            )
        )

        ffmpeg.execute()
    st.markdown("Cost: " + str(time.time() - stime) + "s")

    with st.status("Splitting audio"):
        audio = AudioSegment.from_file(OUTPUT_FILE, format="flac")
        segments = []
        chunk = None
        max_seconds = 60 * 5  # 5 minutes
        for seg in split_on_silence(
            audio,
            min_silence_len=500,
            silence_thresh=audio.dBFS - 16,
            keep_silence=True,
        ):
            if chunk and (chunk.frame_count() / chunk.frame_rate) > max_seconds:
                segments.append(chunk)
                chunk = None
            if not chunk:
                chunk = seg
            else:
                chunk += seg
        if chunk:
            segments.append(chunk)

    for seg in segments:
        st.markdown(f"Segment duration: {seg.duration_seconds}s")

    srt_files = []
    audio_files = []
    shift = 0
    for i, seg in enumerate(segments, start=1):
        fname = f"/tmp/output_{i}.flac"
        srt_fname = f"/tmp/output_{i}.srt"
        seg.export(fname, format="flac")
        audio_files.append(fname)

        with open(fname, "rb") as audio_file:
            stime = time.time()
            if choosed_model == "Groq Whisper":
                groq_client = Groq()
                with st.status("Transcribing using Groq whisper-large-v3: " + fname):
                    transcription = groq_client.audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-large-v3",
                        response_format="verbose_json",
                    )

                subs = pysrt.SubRipFile()
                for i, utter in enumerate(transcription.segments, start=1):
                    sub = pysrt.SubRipItem(
                        i,
                        start=int(utter["start"] * 1000),
                        end=int(utter["end"] * 1000),
                        text=utter["text"],
                    )
                    subs.append(sub)
            elif choosed_model == "Whisper":
                stime = time.time()
                with st.status("Transcribing using Azure OpenAI Whisper: " + fname):
                    transcript = client.audio.transcriptions.create(
                        file=audio_file,
                        model=deployment_id,
                        response_format="srt",
                    )

                subs = pysrt.from_string(transcript)
            elif choosed_model == "Volcengine":
                stime = time.time()
                with st.status("Transcribing using Azure OpenAI Whisper: " + fname):
                    volcengine_data = volcengine_transcribe(
                        audio_file.read(),
                        choosed_language,
                    )
                subs = pysrt.SubRipFile()
                for i, utter in enumerate(volcengine_data["utterances"], start=1):
                    sub = pysrt.SubRipItem(
                        i,
                        start=utter["start_time"],
                        end=utter["end_time"],
                        text=utter["text"],
                    )
                    subs.append(sub)

        subs.shift(seconds=shift)
        subs.save(srt_fname)
        shift += seg.duration_seconds

        srt_files.append(srt_fname)

        st.markdown("Cost: " + str(time.time() - stime) + "s")

    merge_srt_files(srt_files, OUTPUT_SRT_FILE)
    st.markdown("Merged: " + open(OUTPUT_SRT_FILE, "r").read())

    st.video(video_data, subtitles=OUTPUT_SRT_FILE)

    # Clean files
    for tmp_file in [INPUT_FILE, OUTPUT_FILE] + audio_files + srt_files:
        if os.path.isfile(tmp_file):
            os.remove(tmp_file)
