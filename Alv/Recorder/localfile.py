from Recorder import Recorder
import librosa


class LocalfileRecorder(Recorder):
    def __init__(self, audio_file, **kwargs):
        sr = None
        if "sr" in kwargs:
            sr = kwargs["sr"]
        self.audio, kwargs["sr"] = librosa.load(audio_file, sr=sr)
        super().__init__(**kwargs)

    def record_chunks(self):
        last_chunk_end = 0
        while not self.stopped:
            start, end = last_chunk_end, last_chunk_end + (
                self.chunk_duration * self.sr
            )
            result = self.audio[start:end]
            if len(result) == 0:
                break
            self.q.put(result)
            last_chunk_end = end
