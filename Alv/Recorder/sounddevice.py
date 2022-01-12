from Recorder import Recorder
import sounddevice as sd
import warnings
import numpy as np
import pyaudio
import time

class SounddeviceRecorder(Recorder):
    def __init__(self, device=0, stream_sr=44100, **kwargs):
        self.last_time = None
        self.stream_sr = stream_sr
        self.device = device
        self.audio = pyaudio.PyAudio()
        self.input_buffer = []
        super().__init__(**kwargs)

    def record_chunks(self):

        def callback(input_data, frame_count, time_info, flags):
            input_data = np.frombuffer(input_data, dtype=np.float32)
            self.input_buffer += list(input_data)
            if len(self.input_buffer) / self.stream_sr >= self.chunk_duration:
                self.q.put(self.input_buffer)
                self.input_buffer = []
            return input_data, pyaudio.paContinue

        stream = self.audio.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=self.stream_sr,
                            input=True,
                            stream_callback=callback,
                            frames_per_buffer=4096,
                            input_device_index=self.device
                            )

        stream.start_stream()

        while stream.is_active():
            time.sleep(0.1)