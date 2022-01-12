import os
from ASR.HuggingfaceASR import HuggingfaceASR
from VAD.webrtc import WebrtcVAD
from IntentDetector.sentence_transformer import SentenceTransformerIntents
from Recorder.localfile import LocalfileRecorder
from Recorder.sounddevice import SounddeviceRecorder
from VAD.pyannote import PyannoteVAD
import sounddevice as sd
import click


@click.command()
@click.option("--list-devices", is_flag=True)
@click.option("--input-device", default=sd.default.device)
@click.option("--local-file")
@click.option("--data-path", default="/tmp/alv")
@click.option("--rec-subpath", default="rec")
@click.option("--vad-subpath", default="vad")
@click.option("--intent-subpath", default="int")
@click.option("--asr-subpath", default="asr")
@click.option("--callword", default="alfred")
@click.option("--sampling_rate", default=16_000)
def start_alv(
    list_devices,
    input_device,
    local_file,
    data_path,
    rec_subpath,
    vad_subpath,
    intent_subpath,
    asr_subpath,
    callword,
    sampling_rate,
):
    sr = sampling_rate

    if list_devices:
        devices = [d for d in list(sd.query_devices())]
        for i, device in enumerate(devices):
            print(i, device["name"])
        return

    rec_path = os.path.join(data_path, rec_subpath)
    if local_file is not None:
        rec = LocalfileRecorder(local_file, data_path=rec_path, chunk_duration=2, sr=sr)
    else:
        rec = SounddeviceRecorder(
            data_path=rec_path, chunk_duration=2, sr=sr, to_pcm=True, device=int(input_device)
        )

    vad = PyannoteVAD(data_path=os.path.join(data_path, vad_subpath))

    asr = HuggingfaceASR(
        "flozi00/wav2vec2-large-xlsr-53-german-with-lm",
        callword=callword,
        data_path=os.path.join(data_path, asr_subpath),
    )

    intents = SentenceTransformerIntents(
        "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
        data_path=os.path.join(data_path, intent_subpath),
    )

    intents.add_intent("schalte das licht an", lambda x: print(x), "💡")
    intents.add_intent("spiele die nachrichten ab", lambda x: print(x), "📰")

    print("started alv!")

    for intent in intents.intents(rec, vad, asr):
        print(intent)
        pass


if __name__ == "__main__":
    start_alv()
