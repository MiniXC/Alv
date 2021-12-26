from abc import ABC, abstractmethod


class ASR(ABC):
    @abstractmethod
    def recognize(self, audio_file):
        return NotImplemented
