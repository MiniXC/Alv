from pathlib import Path
from ASR.HuggingfaceASR import HuggingfaceASR
from VAD.pyannote import PyannoteVAD
from Recorder.sounddevice import SounddeviceRecorder
import sounddevice as sd
import click
import os
from glob import glob

devices = [d for d in list(sd.query_devices())]
print(devices)

@click.command()
@click.option("--list-devices", is_flag=True)
@click.option("--input-device")
@click.option("--data-path", default="/tmp/alv")
@click.option("--hotword", default="alfred")
def start_alv(list_devices, input_device, data_path, hotword):
    # make paths
    Path(data_path).mkdir(parents=True, exist_ok=True)
    for subpath in ["segmented", "raw", "recognized"]:
        Path(os.path.join(data_path, subpath)).mkdir(parents=True, exist_ok=True)
    # remove old wavs
    for wav in glob(os.path.join(data_path, "segmented", "*.wav")):
        os.remove(wav)
    # remove old txts
    for txt in glob(os.path.join(data_path, "recognized", "*.txt")):
        os.remove(txt)

    if list_devices:
        for i, device in enumerate(devices):
            print(i, device["name"])
        return

    rec = SounddeviceRecorder(device=input_device, temp_dir=data_path)
    vad = PyannoteVAD()
    asr = HuggingfaceASR("flozi00/wav2vec2-large-xlsr-53-german-with-lm")

    prev_text = None
    for audio in vad.segment(rec):
        text = asr.recognize(audio)
        # TODO: add this to the ASR superclass 
        if prev_text is not None:
            text = prev_text + " " + text
            prev_text = None
        if text == hotword:
            prev_text = text
        if len(text) > 0:
            transcription_path = audio.replace(".wav", ".txt").replace("segmented", "recognized")
            with open(transcription_path, "w") as t:
                t.write(text)
        


if __name__ == "__main__":
    start_alv()
