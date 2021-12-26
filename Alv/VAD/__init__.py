from abc import ABC, abstractmethod
from typing import List, Tuple
import librosa
import numpy as np
from Recorder import Recorder


class VAD(ABC):
    def __init__(
        self, hyper_parameters, boundary_treshhold_in_ms=100, padding_in_ms=(100, 20)
    ):
        self.boundary_treshhold = boundary_treshhold_in_ms / 1000
        if isinstance(padding_in_ms, int):
            padding_in_ms = (padding_in_ms, padding_in_ms)
        self.start_padding = padding_in_ms[0] / 1000
        self.end_padding = padding_in_ms[1] / 1000
        self.init_hyper_parameters(hyper_parameters)

    @abstractmethod
    def init_hyper_parameters(self, hyper_parameters):
        raise NotImplementedError()

    @abstractmethod
    def detect_activity(self, audio) -> List[Tuple[int, int]]:
        return NotImplemented

    def segment(self, recorder: Recorder):
        previous_audio = None
        for chunk in recorder.record():
            # TODO: load audio in parallel with detect_activity
            audio, sr = librosa.load(chunk)
            for segment in self.detect_activity(chunk):
                if segment is None:
                    if previous_audio is not None:
                        yield previous_audio, sr
                        previous_audio = None
                    continue
                start = int(max(0, (segment[0] - self.start_padding)) * sr)
                end = int(
                    min((recorder.chunk_duration, segment[1] + self.end_padding)) * sr
                )
                current_audio = audio[start:end]

                if previous_audio is not None:
                    if segment[0] <= self.boundary_treshhold:
                        start = 0
                        current_audio = audio[start:end]
                        current_audio = np.concatenate([previous_audio, current_audio])
                    else:
                        yield previous_audio, sr
                        previous_audio = None
                if segment[1] <= recorder.chunk_duration - self.boundary_treshhold:
                    yield current_audio, sr
                    previous_audio = None
                else:
                    previous_audio = current_audio
