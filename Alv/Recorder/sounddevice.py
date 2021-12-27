from Recorder import Recorder
import sounddevice as sd
import warnings


class SounddeviceRecorder(Recorder):
    def __init__(self, device=sd.default.device, **kwargs):
        if "sr" not in kwargs:
            kwargs["sr"] = sd.query_devices(device=sd.default.device, kind="input")[
                "default_samplerate"
            ]
        self.device = device
        super().__init__(**kwargs)

    def record_chunks(self):
        def chunk_callback(indata, frames, time, status):
            if status:
                warnings.warn(status)
            self.q.put(indata)

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
