import streamlit as st
import os
import gc
import torch
import tempfile
from asr import WhisperTranscriber
from ner import ClinicalNER
from summarizer import MedGemmaSummarizer
from utils import export_to_excel, parse_summary_to_dict

# Set page configuration
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide"
)

# Force GPU selection
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_pipeline(audio_path):
    results = {}
    
    # ------------------ ASR ------------------
    with st.status("Initializing ASR and transcribing...", expanded=True) as status:
        st.write("Loading Whisper model...")
        asr = WhisperTranscriber()
        st.write("Transcribing audio file...")
        transcript = asr.transcribe(audio_path)
        results['transcript'] = transcript
        status.update(label="ASR Complete!", state="complete", expanded=False)
        
        # Release ASR resources immediately
        del asr
        cleanup()

    # ------------------ NER ------------------
    with st.status("Initializing NER and extracting entities...", expanded=True) as status:
        st.write("Loading Biomedical NER model...")
        ner = ClinicalNER()
        st.write("Extracting clinical entities...")
        entities = ner.extract(transcript)
        results['entities'] = entities
        status.update(label="NER Complete!", state="complete", expanded=False)
        
        # Release NER resources immediately
        del ner
        cleanup()

    # ------------------ SUMMARIZER ------------------
    with st.status("Initializing Summarizer and generating report...", expanded=True) as status:
        st.write("Loading MedGemma model...")
        summarizer = MedGemmaSummarizer()
        st.write("Generating clinical summary...")
        summary = summarizer.summarize(transcript)
        results['summary'] = summary
        status.update(label="Summarization Complete!", state="complete", expanded=False)
        
        # Release summarizer resources
        del summarizer
        cleanup()

    # ------------------ EXPORT ------------------
    output_file = "clinical_output.xlsx"
    export_to_excel(
        output_file,
        transcript,
        entities,
        summary
    )
    results['output_file'] = output_file
    
    return results

def main():
    st.title("🏥 Medical Clinical Documentation AI")
    st.markdown("""
    This application processes medical audio recordings into structured clinical documentation.
    1. **ASR**: Transcribe audio to text.
    2. **NER**: Extract symptoms, medications, and therapies.
    3. **Summarizer**: Generate a structured medical report using MedGemma.
    """)

    uploaded_file = st.file_uploader("Upload Medical Recording (WAV/MP3)", type=["wav", "mp3", "m4a"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("🚀 Process Clinical Pipeline", type="primary"):
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                with st.spinner("Processing... Please wait (this may take a few minutes)"):
                    results = run_pipeline(tmp_path)
                
                st.success("✅ Pipeline completed successfully!")
                
                # Layout for results
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📄 Transcript", 
                    "🔍 Clinical Entities", 
                    "📝 Medical Summary",
                    "📊 Full Report View"
                ])
                
                with tab1:
                    st.subheader("Transcript")
                    st.text_area("Full Transcript", results['transcript'], height=400)
                
                with tab2:
                    st.subheader("Extracted Entities")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("### Symptoms")
                        if results['entities']['symptoms']:
                            for sym in results['entities']['symptoms']:
                                st.write(f"- {sym}")
                        else:
                            st.write("None detected.")
                            
                        st.write("### Therapies")
                        if results['entities']['therapies']:
                            for therapy in results['entities']['therapies']:
                                st.write(f"- {therapy}")
                        else:
                            st.write("None detected.")
                            
                    with col2:
                        st.write("### Medications")
                        if results['entities']['medications']:
                            for med in results['entities']['medications']:
                                st.info(f"**{med['name']}**\n- Dosage: {med['dosage']}\n- Route: {med['route']}\n- Frequency: {med['frequency']}\n- Negated: {'Yes' if med['negated'] else 'No'}")
                        else:
                            st.write("None detected.")
                
                with tab3:
                    st.subheader("MedGemma Structured Summary")
                    st.markdown(results['summary'])

                with tab4:
                    st.subheader("Excel Sheet Structured View")
                    st.write("This table shows exactly how the data is organized in your downloaded Excel file.")
                    
                    # Prepare the data same as export_to_excel
                    report_data = []
                    report_data.append({"Field": "Transcript", "Value": results['transcript']})
                    
                    # Symptoms
                    symptom_list = []
                    for s in results['entities'].get("symptoms", []):
                        if isinstance(s, dict):
                            text = s.get("text", "")
                            if s.get("negated"): text += " (Negated)"
                            symptom_list.append(text)
                        else:
                            symptom_list.append(str(s))
                    report_data.append({"Field": "Extracted Symptoms", "Value": ", ".join(symptom_list)})
                    
                    # Medications
                    med_list = []
                    for m in results['entities'].get("medications", []):
                        if isinstance(m, dict):
                            text = m.get("name", "")
                            if m.get("negated"): text += " (Negated)"
                            med_list.append(text)
                        else:
                            med_list.append(str(m))
                    report_data.append({"Field": "Extracted Medications", "Value": ", ".join(med_list)})

                    # Therapies
                    therapy_list = results['entities'].get("therapies", [])
                    report_data.append({"Field": "Extracted Therapies", "Value": ", ".join(therapy_list)})
                    
                    # Social/Family (Placeholder as per Excel export logic)
                    report_data.append({"Field": "Family & Social Details", "Value": ", ".join(results['entities'].get("family_friends", []))})
                    
                    # Summary Sections
                    summary_sections = parse_summary_to_dict(results['summary'])
                    for section, content in summary_sections.items():
                        report_data.append({"Field": section, "Value": content})
                    
                    # Display as table
                    st.table(report_data)
                
                # Download button for Excel
                with open(results['output_file'], "rb") as f:
                    st.download_button(
                        label="📥 Download Structured Excel Report",
                        data=f,
                        file_name="clinical_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

if __name__ == "__main__":
    main()
