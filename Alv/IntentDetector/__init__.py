from abc import ABC, abstractmethod
from typing import List
import os

from utils import class_with_path


@class_with_path(delete_file_extension="txt")
class IntentDetector(ABC):
    def __init__(self, intent_treshhold=0.8):
        self.intent_functions = []
        self.intent_treshhold = intent_treshhold

    def add_intent(self, intent, f, tag):
        self.intent_functions.append([intent, f, tag])

    @abstractmethod
    def recognize_intent(self, text, intents) -> List[float]:
        return NotImplemented

    def intents(self, rec, vad, asr):
        intents = [x[0] for x in self.intent_functions]
        for text_path in asr.recognize(vad, rec):
            probs = self.recognize_intent(open(text_path).read(), intents)
            print(list(zip(intents, probs)))
            for index, (i, p) in enumerate(zip(intents, probs)):
                if p >= self.intent_treshhold:
                    result = self.intent_functions[index]
                    result[1](i)
                    name = os.path.basename(text_path)
                    path = os.path.join(self.data_path, name)
                    with open(path, "a") as f:
                        f.write(f"{result[2]},{result[1]}\n")
                    yield path
