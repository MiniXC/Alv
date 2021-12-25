from abc import ABC, abstractmethod
from typing import List, Tuple


class VAD(ABC):
    def __init__(self, hyper_parameters, boundary_treshhold_in_ms=100, padding_in_ms=20):
        self.init_hyper_parameters(hyper_parameters)

    @abstractmethod
    def init_hyper_parameters(self, hyper_parameters):
        raise NotImplementedError()

    @abstractmethod
    def detect_activity(self, audio) -> List[Tuple[int, int]]:
        return NotImplemented

    def segment(self, recorder):
        for chunk in recorder.record():
            for segment in self.detect_activity(chunk):
                yield NotImplemented
                # do segmention here next, and yield nicely segmented chunks
