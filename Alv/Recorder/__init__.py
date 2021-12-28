from abc import ABC, abstractmethod
from pathlib import Path
import os
import queue
from threading import Thread
import uuid
from scipy.io import wavfile

from utils import class_with_path


@class_with_path(delete_file_extension="wav")
class Recorder(ABC):
    def __init__(self, sr=None, chunk_duration=1):
        self.sr = sr
        self.chunk_duration = chunk_duration
        self.stopped = False
        self.q = queue.Queue()
        self.thread = Thread(target=self.record_chunks, daemon=True)

    def record(self):
        self.thread.start()
        while not self.stopped:
            recording = self.q.get()
            rec_path = self.generate_path()
            wavfile.write(rec_path, int(self.sr), recording)
            yield rec_path

    @abstractmethod
    def record_chunks(self):
        # this has to put chunks of audio of length self.chunk_duration on self.q
        # this should respect self.stopped
        raise NotImplementedError()

    def stop(self):
        self.stopped = True
