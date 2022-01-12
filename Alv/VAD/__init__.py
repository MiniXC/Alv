from abc import ABC, abstractmethod
import os
from typing import List, Tuple
import uuid
import librosa
import numpy as np
from scipy.io import wavfile
from utils import class_with_path
from Recorder import Recorder


@class_with_path(delete_file_extension="wav")
class VAD(ABC):
    def __init__(
        self,
        boundary_treshhold_in_ms=200,
        padding_in_ms=(200, 200),
    ):
        self.boundary_treshhold = boundary_treshhold_in_ms / 1000
        if isinstance(padding_in_ms, int):
            padding_in_ms = (padding_in_ms, padding_in_ms)
        self.start_padding = padding_in_ms[0] / 1000
        self.end_padding = padding_in_ms[1] / 1000

    @abstractmethod
    def detect_activity(self, path, audio) -> List[Tuple[int, int]]:
        return NotImplemented

    def save_audio(self, sr, audio):
        path = self.generate_path()
        wavfile.write(path, sr, audio)
        return path

    def segment(self, recorder: Recorder):
        self.sr = 16_000 # clean this up
        previous_audio = None
        for chunk in recorder.record():
            # TODO: check how fast this is, and possibly parallelize
            audio, sr = librosa.load(chunk, sr=16_000)
            for segment in self.detect_activity(chunk, audio):
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
                cont_prev = (
                    segment[0] <= self.boundary_treshhold
                ) and previous_audio is not None
                # the audio (possibly) continues in the next chunk
                cont_next = (
                    segment[1] > recorder.chunk_duration - self.boundary_treshhold
                )

                if cont_prev:
                    start = 0
                    current_audio = audio[start:end]
                    current_audio = np.concatenate([previous_audio, current_audio])
                    previous_audio = None
                    # we can't yield yet because the audio might continue
                elif previous_audio is not None:
                    # we yield the previous audio because it has no continuation
                    yield self.save_audio(sr, previous_audio)
                    previous_audio = None
                if cont_next:
                    abs_end = len(audio)
                    if cont_prev:
                        # we concatenate the end that was cut off
                        current_audio = np.concatenate(
                            [current_audio, audio[end:abs_end]]
                        )
                    else:
                        current_audio = audio[start:abs_end]
                    previous_audio = current_audio
                    # we cannot yield yet, as the audio will possibly continue
                else:
                    # we yield as there is no possible continuation in the next chunk
                    yield self.save_audio(sr, current_audio)
