from ASR import ASR
from transformers import AutoModelForCTC, AutoProcessor
import librosa
import torch


class HuggingfaceASR(ASR):
    def __init__(self, model_name, sr=16_000, fp16=False, **kwargs):
        print("loading wav2vec model")
        self.model = AutoModelForCTC.from_pretrained(model_name)
        self.fp16 = fp16
        if self.fp16:
            self.model.half()
        print("loading wav2vec processor")
        self.processor = AutoProcessor.from_pretrained(model_name)
        print(f"finished loading {model_name}")
        self.sr = sr
        super().__init__(**kwargs)

    def recognize_chunk(self, audio_file):
        audio, _ = librosa.load(audio_file, sr=self.sr)
        input_values = self.processor(
            audio, return_tensors="pt", sampling_rate=16_000
        ).input_values
        with torch.no_grad():
            if self.fp16:
                input_values = input_values.half()
            logits = self.model(input_values).logits.numpy()[0]
        decoded = self.processor.decode(logits)
        return decoded.text
