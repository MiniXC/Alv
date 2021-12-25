from abc import ABC, abstractmethod
from pathlib import Path
import os
from typing import Iterable
import sounddevice as sd
from scipy.io import wavfile

class Recorder(ABC):
    def __init__(self, temp_dir="/tmp/Alv", sr=None, chunk_duration=5):
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        self.temp_dir = temp_dir
        if sr is None:
            sr = sd.query_devices(sd.default.device)['default_samplerate']
        self.sr = sr
        self.chunk_duration = chunk_duration
        self.stopped = False

    def record(self):
        while not self.stopped:
            yield self.record_chunk()
        self.stopped = False

    @abstractmethod
    def record_chunk(self):
        raise NotImplementedError()

    def stop(self):
        self.stopped = True


class SounddeviceRecorder(Recorder):
    def record_chunk(self):
        recording = sd.rec(
            int(self.chunk_duration * self.sr), samplerate=self.sr, channels=2
        )
        sd.wait()
        rec_path = os.path.join(self.temp_dir, "temp_recording.wav")
        wavfile.write(rec_path, int(self.sr), recording)
        return rec_path
