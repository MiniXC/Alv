from typing import List, Tuple
from VAD import VAD
import torch

class PyannoteVAD(VAD):
    def __init__(self, **kwargs):
        self.pipeline = torch.hub.load('pyannote/pyannote-audio', 'sad', pipeline=True)
        super().__init__(**kwargs)

    def detect_activity(self, path, audio) -> List[Tuple[int, int]]:
        vad = self.pipeline({"audio": path})
        result = [(track.start, track.end) for track in vad.get_timeline()]
        if len(result) == 0:
            result = [None]
        return result
