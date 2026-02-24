import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def remove_non_ascii(text):
    """
    Hard constraint: keep ONLY ASCII characters
    Prevents multilingual token bleed & decoding artifacts
    """
    return re.sub(r"[^\x00-\x7F]+", "", text)


class MedGemmaSummarizer:

    def __init__(self, model_id="google/medgemma-1.5-4b-it", device=None):

        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"[MedGemma] Loading tokenizer from {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print("[MedGemma] pad_token_id set to eos_token_id")

        print(f"[MedGemma] Loading model from {model_id}...")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            token=HF_TOKEN
        )

        self.model.eval()
        print("[MedGemma] Model loaded successfully!")

    # -------------------------------------------------------------

    def summarize(self, text):

        prompt = f"""You are a medical doctor creating a comprehensive clinical summary for patient records.

CLINICAL TRANSCRIPT:
{text}

Create a detailed clinical summary with the following structure. Write ONLY in English. Do NOT repeat sections.

**PATIENT INFORMATION:**
- Name:
- Age:
- Gender:
- Chief Complaint:

**SYMPTOMS:**
List all symptoms mentioned with severity and duration:
- [Symptom 1]: [severity/description]
- [Symptom 2]: [severity/description]

**MEDICATIONS PRESCRIBED:**
For each medication, provide:
1. Medication Name & Dosage
2. How to Use: [frequency, route, timing]
3. Purpose: [why this medication is prescribed]
4. Side Effects to Watch For: [common side effects]
5. Duration: [how long to take]

**SOCIAL & FAMILY HISTORY:**
- Family History: [relevant family medical history]
- Social Context: [living situation, support system, friends/family mentioned]

**VITAL SIGNS & LAB RESULTS:**
Document all vital signs and laboratory findings mentioned.

**DIAGNOSIS:**
Primary and secondary diagnoses based on clinical findings.

**THERAPY & PROCEDURES:**
- Ongoing Therapies: [oxygen therapy, IV fluids, etc.]
- Surgical Procedures: [if any mentioned or planned]
- Other Interventions: [any other treatments]

**FOLLOW-UP CARE:**
- Follow-up appointments needed
- Monitoring requirements
- Precautions and warnings

**CLINICAL NOTES:**
Document ONLY clinically relevant details explicitly mentioned in the transcript that do not fit the sections above.
Do NOT infer, speculate, or introduce new information.
If no additional details exist, write: None.

Use clear, professional medical language. Be thorough and specific. Write in English only."""

        try:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"[MedGemma] Chat template fallback: {e}")
            formatted_prompt = (
                f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            padding=True
        ).to(self.model.device)

        print("[MedGemma] Generating comprehensive clinical summary...")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1200,
                min_new_tokens=400,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=4,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]

        summary = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        summary = remove_non_ascii(summary)

        # ---------- CLEANUP ----------
        summary = summary.replace("<end_of_turn>", "").replace("<start_of_turn>", "").strip()
        summary = re.sub(r'<unused\d+>', '', summary)
        summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)
        summary = re.sub(r'<[^>]+>', '', summary)

        if summary.count("**CHIEF") > 1:
            parts = re.split(r'\*\*CHI?EF', summary)
            if len(parts) > 1:
                summary = "**CHIEF" + parts[1]

        summary = summary.replace("**CHIEFT COMPLAINTS", "**CHIEF COMPLAINT")
        summary = summary.replace("**CHEF COMPLAINS", "**CHIEF COMPLAINT")

        print(f"[MedGemma] ✓ Summary generated ({len(summary.split())} words)")

        return summary.strip()
