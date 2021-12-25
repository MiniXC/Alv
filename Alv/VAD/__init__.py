from abc import ABC, abstractmethod
from typing import List, Tuple


class VAD(ABC):
    def __init__(self, hyper_parameters):
        self.init_hyper_parameters(hyper_parameters)

    @abstractmethod
    def init_hyper_parameters(self, hyper_parameters):
        raise NotImplementedError()

    @abstractmethod
    def detect_activity(self, audio) -> List[Tuple[int, int]]:
        return NotImplemented
