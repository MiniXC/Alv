from typing import List, Tuple
from VAD import VAD
import librosa

from pyannote.audio.pipelines import VoiceActivityDetection

_DEFAULT_HYPER_PARAMETERS = {
    # onset/offset activation thresholds
    "onset": 0.5,
    "offset": 0.5,
    # remove speech regions shorter than that many seconds.
    "min_duration_on": 0.15,
    # fill non-speech regions shorter than that many seconds.
    "min_duration_off": 0.05,
}


class PyannoteVAD(VAD):
    def __init__(self, hyper_parameters=_DEFAULT_HYPER_PARAMETERS):
        self.pipeline = VoiceActivityDetection(segmentation="pyannote/segmentation")
        super().__init__(hyper_parameters)

    def init_hyper_parameters(self, hyper_parameters):
        self.pipeline.instantiate(hyper_parameters)

    def detect_activity(self, audio) -> List[Tuple[int, int]]:
        vad = self.pipeline(audio)
        return [(track[0].start, track[0].end) for track in vad.itertracks()]
