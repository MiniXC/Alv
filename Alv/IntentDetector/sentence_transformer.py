from typing import List
from IntentDetector import IntentDetector
from sentence_transformers import SentenceTransformer, util


class SentenceTransformerIntents(IntentDetector):
    def __init__(self, model_name, **kwargs):
        self.model = SentenceTransformer(model_name)
        super().__init__(**kwargs)

    def recognize_intent(self, text, intents) -> List[float]:
        embeddings1 = self.model.encode([text], convert_to_tensor=True)
        embeddings2 = self.model.encode(intents, convert_to_tensor=True)
        return util.cos_sim(embeddings1, embeddings2)[0]
