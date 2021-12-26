from Recorder import Recorder
import sounddevice as sd
import warnings


class SounddeviceRecorder(Recorder):
    def __init__(
        self, temp_dir="/tmp/alv", sr=None, chunk_duration=2, device=sd.default.device
    ):
        if sr is None:
            sr = sd.query_devices(sd.default.device)["default_samplerate"]
        self.device = device
        super().__init__(temp_dir=temp_dir, sr=sr, chunk_duration=chunk_duration)

    def record_chunks(self):
        def chunk_callback(indata, frames, time, status):
            if status:
                warnings.warn(status)
            self.q.put(indata.copy())

        with sd.InputStream(
            samplerate=self.sr,
            device=self.device,
            channels=1,
            blocksize=int(self.chunk_duration * self.sr),
            callback=chunk_callback,
        ):
            try:
                while not self.stopped:
                    sd.sleep(100)
            except (KeyboardInterrupt, SystemExit):
                self.stopped = True
