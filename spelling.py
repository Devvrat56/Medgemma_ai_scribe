import re
from difflib import get_close_matches

MEDICAL_VOCAB = [
    "fever", "cough", "nausea", "hypertension", "diabetes",
    "paracetamol", "ibuprofen", "mg", "blood", "pressure"
]

class MedicalSpellCorrector:
    def __init__(self, vocab=None):
        self.vocab = vocab if vocab else MEDICAL_VOCAB

    def correct_word(self, word):
        matches = get_close_matches(word, self.vocab, n=1, cutoff=0.8)
        return matches[0] if matches else word

    def correct_text(self, text):
        tokens = re.findall(r"\w+|\S", text)
        corrected = [self.correct_word(t) if t.isalpha() else t for t in tokens]
        return " ".join(corrected)
