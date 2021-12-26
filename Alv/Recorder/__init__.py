from abc import ABC, abstractmethod
from pathlib import Path
import os
import queue
from threading import Thread
from scipy.io import wavfile


class Recorder(ABC):
    def __init__(self, temp_dir="/tmp/Alv", sr=None, chunk_duration=2):
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        self.temp_dir = temp_dir
        self.sr = sr
        self.chunk_duration = chunk_duration
        self.stopped = False
        self.q = queue.Queue()
        self.thread = Thread(target=self.record_chunks, daemon=True)

    def record(self):
        self.thread.start()
        while not self.stopped:
            recording = self.q.get()
            rec_path = os.path.join(self.temp_dir, "temp_recording.wav")
            wavfile.write(rec_path, int(self.sr), recording)
            yield rec_path

    @abstractmethod
    def record_chunks(self):
        # this has to put chunks of audio of length self.chunk_duration on self.q
        # this should respect self.stopped
        raise NotImplementedError()

    def stop(self):
        self.stopped = True
