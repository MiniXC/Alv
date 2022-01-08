from typing import List, Tuple
from VAD import VAD

from pyannote.audio.pipelines import VoiceActivityDetection

_DEFAULT_HYPER_PARAMETERS = {
    # onset/offset activation thresholds
    "onset": 0.4,
    "offset": 0.5,
    # remove speech regions shorter than that many seconds.
    "min_duration_on": 0.15,
    # fill non-speech regions shorter than that many seconds.
    "min_duration_off": 0.15,
}


class PyannoteVAD(VAD):
    def __init__(self, hyper_parameters=_DEFAULT_HYPER_PARAMETERS, **kwargs):
        self.pipeline = VoiceActivityDetection(segmentation="pyannote/segmentation")
        self.pipeline.instantiate(hyper_parameters)
        super().__init__(**kwargs)

    def detect_activity(self, audio) -> List[Tuple[int, int]]:
        vad = self.pipeline(audio)
        result = [(track[0].start, track[0].end) for track in vad.itertracks()]
        if len(result) == 0:
            result = [None]
        return result
