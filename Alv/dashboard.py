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
view = st.sidebar.selectbox("View", ["ASR", "VAD", "REC"])
num_files = st.sidebar.slider("Number of Results", 1, 100, 10)

def show_specgram(audio_path, component=st):
    audio, sr = librosa.load(audio_path)
    fig = plt.figure(figsize=(10, 2))
    #plt.specgram(audio, Fs=sr)
    plt.plot(audio)
    component.pyplot(fig, clear_figure=True)

files = glob(os.path.join(data_path, "asr", "*.txt"))
files.sort(key=os.path.getmtime, reverse=True)

col1, col2, col3 = st.columns(3)
col1.subheader('Time')
col2.subheader('Recognized Text')
col3.subheader('Audio')

for i,f in enumerate(files):
    if i == num_files:
        break
    time = datetime.fromtimestamp(os.path.getmtime(f))
    col1, col2, col3 = st.columns(3)
    col1.write(time)
    col2.write(open(f).read())
    audio_path = f.replace("asr", "vad").replace(".txt", ".wav")
    show_specgram(audio_path, col3)
    col3.audio(audio_path)
