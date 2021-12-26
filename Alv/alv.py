from pathlib import Path
from ASR.HuggingfaceASR import HuggingfaceASR
from VAD.pyannote import PyannoteVAD
from Recorder.sounddevice import SounddeviceRecorder
import sounddevice as sd
import click
import os
import uuid
from glob import glob

devices = [d for d in list(sd.query_devices())]

@click.command()
@click.option("--list-devices", is_flag=True)
@click.option("--input_device", "-i", type=click.Choice(range(len(devices))))
@click.option("--data_path", default="/tmp/alv")
def start_alv(list_devices, input_device, data_path):
    # make paths
    Path(data_path).mkdir(parents=True, exist_ok=True)
    for subpath in ["segmented", "raw", "recognized"]:
        Path(os.path.join(data_path, subpath)).mkdir(parents=True, exist_ok=True)
    # remove old wavs
    for wav in glob(os.path.join(data_path, "segmented", "*.wav")):
        os.remove(wav)

    if list_devices:
        for i, device in enumerate(devices):
            print(i, device["name"])
        return

    rec = SounddeviceRecorder(device=input_device, temp_dir=data_path)
    vad = PyannoteVAD()
    asr = HuggingfaceASR("flozi00/wav2vec2-large-xlsr-53-german-with-lm")

    for audio in vad.segment(rec):
        print(asr.recognize(audio))


if __name__ == "__main__":
    start_alv()
