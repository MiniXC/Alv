from abc import ABC, abstractmethod
import os
from typing import List, Tuple
import uuid
import librosa
import numpy as np
from scipy.io import wavfile
from Recorder import Recorder


class VAD(ABC):
    def __init__(
        self,
        hyper_parameters,
        data_path="/tmp/alv",
        boundary_treshhold_in_ms=50,
        padding_in_ms=(100, 20),
    ):
        self.data_path = data_path
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

    def save_audio(self, sr, audio):
        path = os.path.join(
            self.data_path, "segmented", f"{uuid.uuid4().hex}.wav"
        )
        wavfile.write(path, sr, audio)
        return path

    def segment(self, recorder: Recorder):
        previous_audio = None
        for chunk in recorder.record():
            # TODO: check how fast this is, and possibly parallelize
            audio, sr = librosa.load(chunk)
            for segment in self.detect_activity(chunk):
                # yield previous audio if no audio detected this chunk
                if segment is None:
                    if previous_audio is not None:
                        yield self.save_audio(sr, previous_audio)
                        previous_audio = None
                    continue

                # default start/end index
                start = int(max(0, (segment[0] - self.start_padding)) * sr)
                end = int(
                    min((recorder.chunk_duration, segment[1] + self.end_padding)) * sr
                )

                current_audio = audio[start:end]

                # the audio continues from the previous chunk
                cont_prev = (segment[0] <= self.boundary_treshhold) and previous_audio is not None
                # the audio (possibly) continues in the next chunk
                cont_next = (segment[1] > recorder.chunk_duration - self.boundary_treshhold)

                if cont_prev:
                    start = 0
                    current_audio = audio[start:end]
                    current_audio = np.concatenate([previous_audio, current_audio])
                    previous_audio = None
                    # we can't yield yet because the audio might continue
                else:
                    # we yield the previous audio because it has no continuation
                    yield self.save_audio(sr, previous_audio)
                    previous_audio = None
                if cont_next:
                    abs_end = len(audio)
                    if cont_prev:
                        # we concatenate the end that was cut off 
                        current_audio = np.concatenate([current_audio, audio[end:abs_end]])
                    else:
                        current_audio = audio[start:abs_end]
                    previous_audio = current_audio
                else:
                    # we yield as there is no possible continuation in the next chunk
                    yield self.save_audio(sr, current_audio)
