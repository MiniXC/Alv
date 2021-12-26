from ASR import ASR
from transformers import AutoModelForCTC, AutoProcessor
import librosa
import torch


class HuggingfaceASR(ASR):
    def __init__(self, model_name, sr=16_000):
        print("loading wav2vec model")
        self.model = AutoModelForCTC.from_pretrained(model_name)
        print("loading wav2vec processor")
        self.processor = AutoProcessor.from_pretrained(model_name)
        print(f"finished loading {model_name}")
        self.sr = sr
        super().__init__()

    def recognize(self, audio_file):
        audio, sr = librosa.load(audio_file, sr=self.sr)
        input_values = self.processor(
            audio, return_tensors="pt", sampling_rate=16_000
        ).input_values
        with torch.no_grad():
            logits = self.model(input_values).logits.numpy()[0]
        decoded = self.processor.decode(logits)
        return decoded.text.lower()
