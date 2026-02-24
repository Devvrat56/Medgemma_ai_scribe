import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class WhisperTranscriber:

    def __init__(self, model_id="openai/whisper-large-v3", device=None):

        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"[ASR] Using device: {self.device}")

        print(f"[ASR] Loading processor from {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)

        print(f"[ASR] Loading model from {model_id}...")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            device_map="auto"
        )

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            # device=0 is removed because device_map handles it
            
            # Smaller chunks → less drift & fewer hallucinations
            chunk_length_s=20,

            # Needed for long recordings
            return_timestamps=True
        )

        print("[ASR] Model loaded successfully!")

    # -------------------------------------------------------------

    def transcribe(self, audio_path):

        print("[ASR] Forcing English decoding...")

        # HARD language constraint (prevents random language switching)


        result = self.pipe(
            audio_path,
            generate_kwargs={

                # CRITICAL FIXES
                "language": "en",
                "task": "transcribe",
                "condition_on_prev_tokens": False,

                # Deterministic decoding → improves numbers & doses
                "temperature": 0.0,

                # Beam search → stabilizes medical dictation
                "num_beams": 5
            }
        )

        text = result["text"]

        print("[ASR] Transcription complete.")
        return text
