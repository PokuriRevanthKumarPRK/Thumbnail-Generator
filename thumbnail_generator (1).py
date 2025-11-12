import os
import subprocess
import streamlit as st
import whisper 
import cv2

# Set FFmpeg path for Streamlit Cloud
os.environ["FFMPEG_BINARY"] = "/usr/bin/ffmpeg"  # Default in Streamlit Cloud

st.title("Thumbnail Generator")
st.write("Upload your video (.mov, .mp4 are only supported) and create stunning a Thumbnail ")
videofile = st.file_uploader("Upload Video", type=["mp4","mov"])

def convert_vid_mp3(input_path, output_path):
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-ab", "192k",
        "-ar", "44100",
        "-y",
        output_path
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print("Successfully converted")
    except subprocess.CalledProcessError as e:
        print("Conversion Failed", e)

# Streamlit uploader fix
if videofile is not None:
    with open("uploaded_video.mp4", "wb") as f:
        f.write(videofile.read())
    convert_vid_mp3("uploaded_video.mp4", "audio.mp3")

"""Audio to Transcript"""
model = whisper.load_model("small.en")
result = model.transcribe("audio.mp3")

# Fix transcript reading
transcript = result["text"]
with open("audio.txt", "w") as f:
    f.write(transcript)

"""Transcript to Prompt"""
from transformers import pipeline

summarise = pipeline("summarization", model="facebook/bart-large-cnn")

# Read the content of the file into a string variable
with open("audio.txt", "r") as f:
    transcript_content = f.read()

# Pass the content string to the summarise function
summary = summarise(transcript_content, max_length=250, min_length=100, do_sample=False)

from groq import Groq
client = Groq(api_key=GROQ_KEY)
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role":"user",
            "content":f"""You are an AI that generates highly relevant YouTube thumbnail prompts.DO NOT make up unrelated topics. Focus ONLY on what the transcript discusses.Analyze the transcript and generate a concise and engaging thumbnail description The image should not contain any text. Now, generate a YouTube thumbnail prompt based on the following summary:{summary} """
        }
    ],
    model = "llama3-70b-8192",
)
generated_prompt = chat_completion.choices[0].message.content

"""Prompt to Image"""
from gradio_client import Client

# Fix environment variable usage
hgf_token = HGF_KEY

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
