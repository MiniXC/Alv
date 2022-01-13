from ASR import ASR
from transformers import AutoModelForCTC, AutoProcessor
import librosa

class HuggingfaceASR(ASR):
    def __init__(self, model_name, sr=16_000, **kwargs):
        print("loading wav2vec model")
        self.model = AutoModelForCTC.from_pretrained(model_name, torchscript=True)
        self.model.eval()
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
        logits = self.model(input_values).logits.numpy()[0]
        decoded = self.processor.decode(logits)
        return decoded.text
