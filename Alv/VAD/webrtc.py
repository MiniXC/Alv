import collections
from typing import List, Tuple
from VAD import VAD
import webrtcvad
import wave
import contextlib

class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)

    triggered = False

    result = []

    was_triggered = False

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame.timestamp, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.8 * ring_buffer.maxlen:
                print("triggered")
                was_triggered = True
                triggered = True
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                result.append([ring_buffer[0][0],-1])
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            ring_buffer.append((frame.timestamp, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.8 * ring_buffer.maxlen:
                print("nottriggered")
                triggered = False
                result[-1][1] = ring_buffer[-1][0] + frame.duration
                ring_buffer.clear()
    if triggered:
        result[-1][1] = ring_buffer[-1][0] + frame.duration

    result = [r for r in result if r[1] - r[0] > 0.25]

    return result

class WebrtcVAD(VAD):
    def __init__(self, mode=3, **kwargs):
        self.vad = webrtcvad.Vad(mode)
        super().__init__(**kwargs)

    def detect_activity(self, path, audio) -> List[Tuple[int, int]]:
        audio, sr = read_wave(path)
        frames = frame_generator(30, audio, sr)
        frames = list(frames)
        segments = vad_collector(sr, 30, 150, self.vad, frames)
        return segments
