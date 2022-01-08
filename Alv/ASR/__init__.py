from abc import ABC, abstractmethod
import os
from pathlib import Path
from utils import class_with_path
from Recorder import Recorder
from VAD import VAD


@class_with_path(delete_file_extension="txt")
class ASR(ABC):
    def __init__(self, callword=None):
        self.callword = callword

    def recognize(self, vad: VAD, recorder: Recorder):
        prev_text = None
        for audio in vad.segment(recorder):
            text = self.recognize_chunk(audio).lower().strip()
            if prev_text is not None:
                text = f"{prev_text} {text}"
                prev_text = None
            if self.callword is not None and text == self.callword:
                prev_text = self.callword
                continue
            if len(text) > 0:
                audio = os.path.basename(audio)
                path = os.path.join(self.data_path, audio.replace(".wav", ".txt"))
                with open(path, "w") as t:
                    t.write(text)
                yield path

    @abstractmethod
    def recognize_chunk(self, audio_file):
        return NotImplemented
