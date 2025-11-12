import os
import streamlit as st
from transformers import pipeline
from gradio_client import Client
from groq import Groq

st.title("Thumbnail Generator")
st.write("Upload your video (.mov, .mp4 are only supported) and create stunning a Thumbnail")
videofile = st.file_uploader("Upload Video", type=["mp4", "mov"])

# --- Convert video to audio locally (small audio file only) ---
if videofile is not None:
    audio_path = "audio.mp3"
    with open("uploaded_video.mp4", "wb") as f:
        f.write(videofile.read())
    
    # Use ffmpeg to extract audio (same as before)
    import subprocess
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", "uploaded_video.mp4",
        "-vn",
        "-acodec", "libmp3lame",
        "-ab", "192k",
        "-ar", "44100",
        "-y",
        audio_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    st.success("Audio extracted!")

    # --- Use hosted Whisper for transcription ---
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from datasets import load_dataset

    # Hosted API alternative using HuggingFace Inference API
    import requests

    hf_token = st.secrets["HF_TOKEN"]  # Put your HF token in Streamlit secrets
    headers = {"Authorization": f"Bearer {hf_token}"}
    files = {"file": open(audio_path, "rb")}
    response = requests.post(
        "https://api-inference.huggingface.co/models/openai/whisper-small",
        headers=headers,
        files=files,
    )
    transcript = response.json()["text"]

    with open("audio.txt", "w") as f:
        f.write(transcript)
    
    st.text_area("Transcript", transcript, height=200)

    # --- Summarization ---
    summarise = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarise(transcript, max_length=250, min_length=100, do_sample=False)[0]["summary_text"]

    # --- Groq prompt generation ---
    groq_client = Groq(api_key=st.secrets["GROQ_KEY"])
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""
You are an AI that generates highly relevant YouTube thumbnail prompts.
DO NOT make up unrelated topics. Focus ONLY on what the transcript discusses.
Analyze the transcript and generate a concise and engaging thumbnail description.
The image should not contain any text.
Now, generate a YouTube thumbnail prompt based on the following summary:{summary}
"""
            }
        ],
        model="llama3-70b-8192",
    )
    generated_prompt = chat_completion.choices[0].message.content

    # --- Prompt to Image ---
    hgf_token = st.secrets["HF_TOKEN"]
    client = Client("black-forest-labs/FLUX.1-schnell", hf_token=hgf_token)
    result = client.predict(
        prompt=generated_prompt,
        seed=0,
        width=800,
        height=800,
        num_inference_steps=4,
        api_name="/infer"
    )

    path = result[0]
    st.image(path, width=500)
