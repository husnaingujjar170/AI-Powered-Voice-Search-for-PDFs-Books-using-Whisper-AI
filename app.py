import os
os.system("pip install torch==2.0.1 torchaudio ffmpeg")
import streamlit as st
import whisper
import fitz  
from sentence_transformers import SentenceTransformer, util
import numpy as np
import tempfile
import os
from io import BytesIO

def extract_text_from_pdf(pdf_bytes):
    """Extracts text from an uploaded PDF file."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text("text").strip() for page in doc)
        return text
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        return ""

def transcribe_audio_locally(audio_bytes):
    """Transcribes speech using OpenAI's Whisper local model."""
    try:
        model = whisper.load_model("medium") 
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes.read())
            temp_audio_path = temp_audio.name
        
        with st.spinner("Transcribing audio... ⏳"):
            result = model.transcribe(temp_audio_path)
        os.remove(temp_audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"❌ Whisper Transcription Error: {e}")
        return ""

def search_pdf_text(query, pdf_text):
    """Finds the most relevant text in the PDF matching the query using semantic similarity."""
    paragraphs = [p.strip() for p in pdf_text.split("\n\n") if p.strip()]
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    query_embedding = model.encode(query, convert_to_tensor=True)
    para_embeddings = model.encode(paragraphs, convert_to_tensor=True)
    
    cos_scores = util.cos_sim(query_embedding, para_embeddings)[0]
    
    top_results = np.argsort(-cos_scores)[:3]
    
    if len(top_results) > 0:
        highlighted_results = "\n\n".join([f"🔎 **Match {i+1}:**\n {paragraphs[i]}" for i in top_results])
        return highlighted_results
    else:
        return "No relevant match found."

st.title("📚 AI-Powered PDF & Audio Search")

uploaded_pdf = st.file_uploader("📂 Upload your PDF file", type="pdf")
if uploaded_pdf is not None:
    pdf_bytes = BytesIO(uploaded_pdf.read())
    pdf_text = extract_text_from_pdf(pdf_bytes)
    st.success("✅ PDF text extracted successfully!")
    
    uploaded_audio = st.file_uploader("🎤 Upload your audio file", type=["wav", "mp3"])
    if uploaded_audio is not None:
        transcribed_query = transcribe_audio_locally(uploaded_audio)
        if transcribed_query:
            st.write("🎤 **Transcribed Query:**", transcribed_query)
            
            search_results = search_pdf_text(transcribed_query, pdf_text)
            st.write("📄 **Matching Text from PDF:**")
            st.markdown(search_results)
