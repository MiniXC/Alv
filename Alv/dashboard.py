import os
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from glob import glob

st_autorefresh(interval=2000)

st.title("🕴️ Alv Dashboard")

data_path = st.text_input("Path", "/tmp/alv")

files = glob(os.path.join(data_path, "segmented", "*.wav"))
files.sort(key=os.path.getmtime)

for f in files:
    st.write(f)
    st.audio(f)
