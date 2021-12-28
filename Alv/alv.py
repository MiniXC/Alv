from ASR.HuggingfaceASR import HuggingfaceASR
from VAD.pyannote import PyannoteVAD
from Recorder.localfile import LocalfileRecorder
from Recorder.sounddevice import SounddeviceRecorder
import sounddevice as sd
import click


@click.command()
@click.option("--list-devices", is_flag=True)
@click.option("--input-device", default=sd.default.device)
@click.option("--local-file")
@click.option("--data-path", default="/tmp/alv")
@click.option("--callword", default="alfred")
def start_alv(list_devices, input_device, local_file, data_path, callword):
    if list_devices:
        devices = [d for d in list(sd.query_devices())]
        for i, device in enumerate(devices):
            print(i, device["name"])
        return

    if local_file is not None:
        rec = LocalfileRecorder(local_file, data_path=data_path+"/rec", chunk_duration=2)
    else:
        rec = SounddeviceRecorder(device=input_device, data_path=data_path, chunk_duration=2)
    vad = PyannoteVAD(data_path=data_path+"/vad")

    asr = HuggingfaceASR(
        "flozi00/wav2vec2-large-xlsr-53-german-with-lm",
        callword=callword,
        data_path=data_path+"/asr",
    )

    for text in asr.recognize(vad, rec):
        print(text)


if __name__ == "__main__":
    start_alv()
