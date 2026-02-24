# 🏥 Clinical Documentation Pipeline

Automated pipeline for generating comprehensive, patient-focused clinical summaries from audio recordings. This tool uses state-of-the-art AI models to transcribe medical conversations and structure them into professional clinical reports.

## 🌟 Features

- **Advanced Speech-to-Text**: Uses `openai/whisper-large-v3` for high-accuracy transcription of medical dialogue, supporting long audio files.
- **Intelligent Summarization**: Powered by `google/medgemma-1.5-4b-it` to generate detailed 8-section clinical reports.
- **Structured Output**: Automatically formats summaries into sections like Patient Info, Symptoms, Medications, Vitals, Diagnosis, and Plan.
- **Excel Export**: Saves the final report, transcript, and extracted entities into a formatted Excel file (`clinical_output.xlsx`) for easy integration with electronic health records.
- **Basic Entity Extraction**: Identifies common symptoms and medications (extensible).

## 🛠️ Components

The pipeline consists of the following modules:

| Component | File | Description |
|-----------|------|-------------|
| **Entry Point** | `main.py` | Orchestrates the entire pipeline (ASR -> NER -> Summarization -> Export). |
| **ASR** | `asr.py` | Handles audio transcription using the Whisper model with timestamp support for long files. |
| **Summarizer** | `summarizer.py` | Generates the structured clinical summary using MedGemma. Includes prompt engineering for specific medical sections. |
| **Utils** | `utils.py` | Handles parsing of the generated summary and exporting data to Excel (`.xlsx`). |
| **NER** | `ner.py` | Basic keyword-based extraction for symptoms and medications. |
| **Spelling** | `spelling.py` | _(Optional)_ Medical spell corrector (currently disabled/commented out in `main.py`). |

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **GPU Support**: A CUDA-capable GPU with at least 12GB VRAM is recommended to run Whisper Large and MedGemma simultaneously.
- **Hugging Face Token**: Required for accessing the gated MedGemma model.

### Installation

1.  **Install Dependencies**:
    Ensure you have the required Python libraries installed:
    ```bash
    pip install torch transformers accelerate openpyxl pandas
    ```
    *(Note: `pandas` is often required by `openpyxl` or for data handling)*

2.  **Configure Access Token**:
    Open `summarizer.py` and ensure your Hugging Face token is set in the `HF_TOKEN` variable:
    ```python
    HF_TOKEN = "your_huggingface_token_here"
    ```

### Usage

1.  **Prepare Audio**:
    Place your medical audio file (e.g., `.wav`, `.mp3`) in a precise location.

2.  **Update input path**:
    Edit `main.py` (line 41) to point to your audio file:
    ```python
    run_pipeline("/path/to/your/audio_file.wav")
    ```

3.  **Run the Pipeline**:
    ```bash
    python main.py
    ```

### Output

After running the pipeline, you will find:
- **`transcript.txt`**: The raw text transcribed from the audio.
- **`clinical_output.xlsx`**: A structured Excel workbook containing:
    - Raw Transcript
    - Extracted Symptoms & Medications
    - **Comprehensive Clinical Summary** broken down into 8 sections.

## 📋 Summary Structure

The generated summary includes the following 8 sections:

1.  **👤 PATIENT INFORMATION**: Demographics and Chief Complaint.
2.  **🩺 SYMPTOMS**: Detailed list with severity and duration.
3.  **💊 MEDICATIONS PRESCRIBED**: Name, dosage, usage instructions, purpose, side effects, and duration.
4.  **📊 VITAL SIGNS & LAB RESULTS**: Recorded vitals and findings.
5.  **🔬 DIAGNOSIS**: Primary and secondary diagnoses.
6.  **🏥 THERAPY & PROCEDURES**: Ongoing therapies, surgeries, and interventions.
7.  **📅 FOLLOW-UP CARE**: Appointments, monitoring, and warnings.
8.  **📝 CLINICAL NOTES**: Additional important information.

## 🔧 Troubleshooting

- **OOM (Out of Memory) Errors**: If you encounter memory errors, try using a smaller Whisper model (e.g., `openai/whisper-medium`) in `asr.py` or running on a GPU with more VRAM.
- **Gated Model Access**: If `MedGemma` fails to load, ensure you have accepted the license agreement on Hugging Face and provided a valid token.
