from datetime import datetime
import os
import librosa
from librosa.core import audio
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from glob import glob
import matplotlib.pyplot as plt

st.set_page_config(page_title="Alvred", page_icon="🕴️")

st_autorefresh(interval=5000)

st.title("🕴️ Alv Dashboard")

data_path = st.sidebar.text_input("Path", "/tmp/alv")
view = st.sidebar.selectbox("View", ["INT", "ASR", "VAD"])
num_files = st.sidebar.slider("Number of Results", 1, 100, 10)
sort = st.sidebar.radio("Sort", ["newest first", "oldest first"])
reverse = sort == "newest first"


@st.cache
def load_audio(audio_path):
    return librosa.load(audio_path)


@st.cache
def load_text(text_path):
    return open(text_path).read()


def show_specgram(audio_path, component=st):
    audio, sr = load_audio(audio_path)
    fig = plt.figure(figsize=(10, 2))
    # plt.specgram(audio, Fs=sr)
    plt.plot(audio)
    component.pyplot(fig, clear_figure=True)


if view == "ASR":
    files = glob(os.path.join(data_path, "asr", "*.txt"))
    files.sort(key=os.path.getmtime, reverse=reverse)

    col1, col2, col3 = st.columns(3)
    col1.subheader("Time")
    col2.subheader("Recognized Text")
    col3.subheader("Audio")

    for i, f in enumerate(files):
        if i == num_files:
            break
        time = datetime.fromtimestamp(os.path.getmtime(f))
        col1, col2, col3 = st.columns(3)
        col1.write(time)
        col2.write(load_text(f))
        audio_path = f.replace("asr", "vad").replace(".txt", ".wav")
        show_specgram(audio_path, col3)
        col3.audio(audio_path)

if view == "VAD":
    files = glob(os.path.join(data_path, "vad", "*.wav"))
    files.sort(key=os.path.getmtime, reverse=reverse)
    col1, col2 = st.columns(2)
    col1.subheader("Time")
    col2.subheader("Audio")
    for i, f in enumerate(files):
        if i == num_files:
            break
        time = datetime.fromtimestamp(os.path.getmtime(f))
        col1, col2 = st.columns(2)
        col1.write(time)
        show_specgram(f, col2)
        col2.audio(f)

if view == "INT":
    files = glob(os.path.join(data_path, "int", "*.txt"))
    files.sort(key=os.path.getmtime, reverse=reverse)

    col1, col2, col3, col4 = st.columns(4)
    col1.subheader("Time")
    col2.subheader("Intent")
    col3.subheader("Recognized Text")
    col4.subheader("Audio")

    for i, f in enumerate(files):
        if i == num_files:
            break
        time = datetime.fromtimestamp(os.path.getmtime(f))
        col1, col2, col3, col4 = st.columns(4)
        col1.write(time)
        col2.info(load_text(f))
        col3.write(load_text(f.replace("int", "asr")))
        audio_path = f.replace("int", "vad").replace(".txt", ".wav")
        show_specgram(audio_path, col4)
        col4.audio(audio_path)