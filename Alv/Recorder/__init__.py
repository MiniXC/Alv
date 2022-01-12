from abc import ABC, abstractmethod
import queue
from threading import Thread
import numpy as np
from scipy.io import wavfile
import librosa
import noisereduce as nr

from utils import class_with_path


@class_with_path(delete_file_extension="wav")
class Recorder(ABC):
    def __init__(self, sr=None, chunk_duration=1, to_pcm=False, reduce_noise=True):
        self.sr = sr
        self.chunk_duration = chunk_duration
        self.stopped = False
        self.q = queue.Queue()
        self.thread = Thread(target=self.record_chunks, daemon=True)
        self.to_pcm = to_pcm

    def record(self):
        self.thread.start()
        while not self.stopped:
            recording = np.array(self.q.get())
            rec_path = self.generate_path()
            if self.stream_sr != self.sr:
                recording = librosa.resample(recording.squeeze(), self.stream_sr, self.sr, res_type="kaiser_fast")
            recording = nr.reduce_noise(y=recording, sr=self.sr)
            if self.to_pcm:
                recording = (recording * 32767).astype(np.int16)
            wavfile.write(rec_path, int(self.sr), recording)
            yield rec_path

    @abstractmethod
    def record_chunks(self):
        # this has to put chunks of audio of length self.chunk_duration on self.q
        # this should respect self.stopped
        raise NotImplementedError()

    def stop(self):
        self.stopped = True
