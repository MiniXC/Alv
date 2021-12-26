from VAD.pyannote import PyannoteVAD
from Recorder.sounddevice import SounddeviceRecorder
from scipy.io import wavfile

rec = SounddeviceRecorder()
vad = PyannoteVAD()

for i, (audio, sr) in enumerate(vad.segment(rec)):
    wavfile.write(f"test{i}.wav", sr, audio)
