import streamlit as st
from transformers import pipeline
from datasets import load_dataset
import torch
import soundfile as sf
import io
import numpy as np

# ---------------------- Page Config ----------------------
st.set_page_config(page_title="Arabic Neural TTS", page_icon="🗣️", layout="centered")

# ---------------------- Title and Introduction ----------------------
st.title("🗣️ Arabic Neural TTS Demo")
st.markdown("Welcome to the **Arabic Text-to-Speech** app using `MBZUAI/speecht5_tts_clartts_ar`. Enter Arabic text and listen to the generated audio.")
st.markdown("---")

# ---------------------- Load TTS Pipeline and Speaker Embeddings ----------------------
@st.cache_resource
def get_tts_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-to-speech", "MBZUAI/speecht5_tts_clartts_ar", device=device)

@st.cache_resource
def load_speaker_embeddings():
    return load_dataset("herwoww/arabic_xvector_embeddings", split="validation")

def get_random_speaker_embedding(ds):
    idx = np.random.randint(0, len(ds))
    return torch.tensor(ds[idx]["speaker_embeddings"]).unsqueeze(0)

def get_first_speaker_embedding(ds):
    return torch.tensor(ds[0]["speaker_embeddings"]).unsqueeze(0)

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.header("📘 Project Info")
    st.markdown("**Model**: MBZUAI/speecht5_tts_clartts_ar")
    st.markdown("**Speaker embeddings**: x-vector")
    st.markdown("**Frameworks**: 🤗 Transformers, Streamlit")

    st.markdown("---")
    st.subheader("👤 Developer Info")
    st.markdown("**Name**:BENAICHA Mohamed Etaib")  
    st.markdown("[📊 My Kaggle](https://www.kaggle.com/mohamedtaib)")  
    st.markdown("[🔗 My LinkedIn](https://www.linkedin.com/in/mohamed-etaib-benaicha-757600254/)")  

    st.markdown("---")
    st.markdown("💡 Tip: Try random speaker voices!")


# ---------------------- Input Section ----------------------
st.subheader("✍️ Input Arabic Text")
sample_phrases = [
    "مرحبًا بكم في هذا المشروع الصوتي",
    "الذكاء الاصطناعي يغير العالم من حولنا",
    "تعلم اللغة العربية يمنحك فرصة جديدة",
    "الصوت البشري يحمل الكثير من المشاعر",
    "البيانات الصوتية ضرورية لتطوير الأنظمة الذكية"
]

ds = load_speaker_embeddings()

col1, col2 = st.columns([3, 2])

with col1:
    text_input = st.text_area("اكتب جملة باللغة العربية", height=150)

with col2:
    selected = st.selectbox("أو اختر جملة جاهزة:", [""] + sample_phrases)
    if selected:
        text_input = selected

# ---------------------- Voice Option ----------------------
st.subheader("🎙️ Speaker Voice Options")
use_random_voice = st.checkbox("🎲 Use random speaker voice", value=False)

# ---------------------- Synthesize and Play ----------------------
if st.button("🔊 Generate Audio"):
    if not text_input.strip():
        st.warning("🚫 الرجاء إدخال نص باللغة العربية.")
    else:
        with st.spinner("🎧 Synthesizing speech..."):
            tts = get_tts_pipeline()
            spk_emb = get_random_speaker_embedding(ds) if use_random_voice else get_first_speaker_embedding(ds)
            result = tts(text_input, forward_params={"speaker_embeddings": spk_emb})
            audio, sr = result["audio"], result["sampling_rate"]

            # Save to buffer
            buf = io.BytesIO()
            sf.write(buf, audio, samplerate=sr, format='WAV')
            buf.seek(0)

            st.success("✅ Audio generated!")
            st.audio(buf.read(), format='audio/wav')
            st.download_button("⬇️ Download Audio", data=buf, file_name="arabic_tts.wav", mime="audio/wav")

# ---------------------- Extra Info ----------------------
with st.expander("ℹ️ What is SpeechT5 and x-vectors?"):
    st.markdown("""
    - **SpeechT5** is a model by Microsoft designed for tasks like text-to-speech and automatic speech recognition.
    - It takes both text and a speaker embedding to generate speech that mimics a specific voice.
    - **x-vectors** are compact representations of speaker identity used to personalize TTS output.
    """)

# ---------------------- Feedback ----------------------
st.markdown("---")
st.subheader("⭐ Rate the Voice Quality")
rating = st.slider("كيف تقيم جودة الصوت؟", 1, 5, value=3)
if st.button("📩 Submit Feedback"):
    st.success(f"🎉 شكراً! تم تسجيل تقيمك: {rating}/5")

# ---------------------- Footer ----------------------
st.markdown("---")
st.caption("Built with 🤗 Transformers & Streamlit | © 2025 Université de Blida 1")
