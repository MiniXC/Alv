import os
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from glob import glob

st.set_page_config(page_title="Alvred", page_icon="🕴️")

st_autorefresh(interval=2000)

st.title("🕴️ Alv Dashboard")

data_path = st.text_input("Path", "/tmp/alv")

files = glob(os.path.join(data_path, "asr", "*.txt"))
files.sort(key=os.path.getmtime)

for f in files:
    st.write(open(f).read())
    st.audio(f.replace("asr", "vad").replace(".txt", ".wav"))
