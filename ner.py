import torch
from transformers import pipeline
import re


class ClinicalNER:

    def __init__(self):

        self.ner = pipeline(
            "token-classification",
            model="d4data/biomedical-ner-all",
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )

        self.negation_cues = ["no", "denies", "without", "not", "never"]

        # ---------------- DOSAGE PATTERN ----------------
        self.dose_pattern = re.compile(
            r"\b\d+(\.\d+)?\s?(mg|mcg|g|ml)\b",
            re.IGNORECASE
        )

        self.frequency_pattern = re.compile(
            r"\b(once daily|twice daily|thrice daily|daily|bid|tid|qid)\b",
            re.IGNORECASE
        )

        self.route_pattern = re.compile(
            r"\b(oral|iv|intravenous|topical|injection)\b",
            re.IGNORECASE
        )

        # ---------------- THERAPY DETECTION (CRITICAL FIX) ----------------
        self.therapy_pattern = re.compile(
            r"\b(chemotherapy|chemo|radiotherapy|radiation therapy|radio therapy|post[- ]?op chemo|postoperative chemotherapy)\b",
            re.IGNORECASE
        )

    # --------------------------------------------------

    def detect_negation(self, phrase, text):

        idx = text.lower().find(phrase.lower())
        if idx == -1:
            return False

        window = text.lower()[max(0, idx - 25): idx]
        return any(cue in window for cue in self.negation_cues)

    # --------------------------------------------------

    def normalize_units(self, dose):

        value = dose.lower().replace(" ", "")

        try:
            if value.endswith("mcg"):
                num = float(value.replace("mcg", ""))
                return f"{num / 1000:.4g} mg"

            if value.endswith("g"):
                num = float(value.replace("g", ""))
                return f"{num * 1000:.4g} mg"

        except:
            pass

        return dose

    # --------------------------------------------------

    def extract_pattern_nearby(self, pattern, text, index, window=50):

        segment = text[max(0, index - window): index + window]
        match = pattern.search(segment)

        return match.group() if match else None

    # --------------------------------------------------

    def extract_therapies(self, text):

        therapies = []

        for match in self.therapy_pattern.finditer(text):
            therapies.append(match.group())

        return list(set(therapies))

    # --------------------------------------------------

    def extract(self, text):

        ner_results = self.ner(text)

        symptoms = []
        medications = []
        other_entities = []

        # Comprehensive lists to capture more from the NER model
        SYMPTOM_LABELS = ["SIGN_SYMPTOM", "DISEASE", "SIGN", "SYMPTOM", "CONDITION", "FINDING"]
        MEDICATION_LABELS = ["CHEMICAL", "MEDICATION", "DRUG", "THERAPEUTIC_PROCEDURE", "PHARMACOLOGICAL_SUBSTANCE"]

        for ent in ner_results:

            label = ent["entity_group"].upper()
            value = ent["word"]

            if label in SYMPTOM_LABELS:
                symptoms.append(value)

            elif label in MEDICATION_LABELS:

                # Try to extract nearby attributes
                dose = self.extract_pattern_nearby(
                    self.dose_pattern, text, ent["start"]
                )

                freq = self.extract_pattern_nearby(
                    self.frequency_pattern, text, ent["start"]
                )

                route = self.extract_pattern_nearby(
                    self.route_pattern, text, ent["start"]
                )

                med_info = {
                    "name": value,
                    "dosage": self.normalize_units(dose) if dose else None,
                    "frequency": freq,
                    "route": route,
                    "negated": self.detect_negation(value, text)
                }
                
                # Avoid duplicates if possible
                if med_info not in medications:
                    medications.append(med_info)
            else:
                other_entities.append({"label": label, "value": value})

        therapies = self.extract_therapies(text)

        return {
            "symptoms": list(set(symptoms)),
            "medications": medications,
            "therapies": therapies,
            "other_medical_entities": other_entities
        }
