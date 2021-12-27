from ASR.HuggingfaceASR import HuggingfaceASR
from VAD.pyannote import PyannoteVAD
from Recorder.sounddevice import SounddeviceRecorder
import sounddevice as sd
import click


@click.command()
@click.option("--list-devices", is_flag=True)
@click.option("--input-device", default=sd.default.device)
@click.option("--data-path", default="/tmp/alv")
@click.option("--callword", default="alfred")
def start_alv(list_devices, input_device, data_path, callword):
    if list_devices:
        devices = [d for d in list(sd.query_devices())]
        for i, device in enumerate(devices):
            print(i, device["name"])
        return

    rec = SounddeviceRecorder(device=input_device, data_path=data_path)
    vad = PyannoteVAD(data_path=data_path)
    asr = HuggingfaceASR(
        "flozi00/wav2vec2-large-xlsr-53-german-with-lm",
        callword=callword,
        data_path=data_path,
    )

    for text in asr.recognize_chunk(vad, rec):
        print(text)


if __name__ == "__main__":
    start_alv()
