import os
from ASR.HuggingfaceASR import HuggingfaceASR
from VAD.pyannote import PyannoteVAD
from Recorder.localfile import LocalfileRecorder
from Recorder.sounddevice import SounddeviceRecorder
import sounddevice as sd
import click


@click.command()
@click.option("--list-devices", is_flag=True)
@click.option("--fp16", is_flag=True)
@click.option("--input-device", default=sd.default.device)
@click.option("--local-file")
@click.option("--data-path", default="/tmp/alv")
@click.option("--rec-subpath", default="rec")
@click.option("--vad-subpath", default="vad")
@click.option("--asr-subpath", default="asr")
@click.option("--callword", default="alfred")
def start_alv(
    list_devices,
    fp16,
    input_device,
    local_file,
    data_path,
    rec_subpath,
    vad_subpath,
    asr_subpath,
    callword,
):
    if list_devices:
        devices = [d for d in list(sd.query_devices())]
        for i, device in enumerate(devices):
            print(i, device["name"])
        return

    rec_path = os.path.join(data_path, rec_subpath)
    if local_file is not None:
        rec = LocalfileRecorder(
            local_file, data_path=rec_path, chunk_duration=2
        )
    else:
        rec = SounddeviceRecorder(
            device=input_device, data_path=data_path, chunk_duration=2
        )

    vad = PyannoteVAD(data_path=os.path.join(data_path, vad_subpath))

    asr = HuggingfaceASR(
        "flozi00/wav2vec2-large-xlsr-53-german-with-lm",
        callword=callword,
        data_path=os.path.join(data_path, asr_subpath),
        fp16=fp16,
    )

    for text in asr.recognize(vad, rec):
        print(text)


if __name__ == "__main__":
    start_alv()
