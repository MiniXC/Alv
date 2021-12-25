from VAD.pyannoteVAD import PyannoteVAD
from Recorder import SounddeviceRecorder

for chunk in SounddeviceRecorder().record():
    vad = PyannoteVAD()
    print(chunk)
    print(vad.detect_activity(chunk))
