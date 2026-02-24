import os
import gc
import torch

# Force GPU selection if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from asr import WhisperTranscriber
from ner import ClinicalNER
from summarizer import MedGemmaSummarizer
from utils import export_to_excel


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -------------------------------------------------------------

def run_pipeline(audio_path):

    print("\n================ PIPELINE STARTED ================\n")

    # ------------------ ASR ------------------
    print("[1/4] Initializing ASR...")
    asr = WhisperTranscriber()

    print("[ASR] Transcribing audio...")
    transcript = asr.transcribe(audio_path)

    print("\n[DEBUG] Raw Transcript:\n")
    print(transcript)

    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript)

    print("\n[ASR] Transcript saved → transcript.txt")

    print("[ASR] Releasing ASR resources...")
    del asr
    cleanup()

    # ------------------ NER ------------------
    print("\n[2/4] Initializing NER...")
    ner = ClinicalNER()

    print("[NER] Extracting entities...")
    entities = ner.extract(transcript)

    print("\n[DEBUG] Extracted Entities:\n")
    print(entities)

    print("[NER] Releasing NER resources...")
    del ner
    cleanup()

    # ------------------ SUMMARIZER ------------------
    print("\n[3/4] Initializing Summarizer...")
    summarizer = MedGemmaSummarizer()

    print("[SUMMARIZER] Generating clinical summary...")
    summary = summarizer.summarize(transcript)

    print("\n[DEBUG] Generated Summary:\n")
    print(summary)

    print("[SUMMARIZER] Releasing summarizer resources...")
    del summarizer
    cleanup()

    # ------------------ EXPORT ------------------
    print("\n[4/4] Exporting results to Excel...")
    export_to_excel(
        "clinical_output.xlsx",
        transcript,
        entities,
        summary
    )

    print("\n✅ Pipeline completed successfully!")
    print("\n================ PIPELINE FINISHED ================\n")


# -------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline("Call with Devvrat Shukla-20260218_153533-Meeting Recording.wav")
