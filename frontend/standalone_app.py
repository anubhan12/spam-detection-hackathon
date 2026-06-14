import os
import re
import joblib
import streamlit as st

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer


# --- PAGE SETUP ---
st.set_page_config(page_title="AI Spam Detector", page_icon="🚫", layout="centered")
st.title("🚫 Smart Spam Detection System")
st.write("This standalone app loads the ML model directly into the interface.")


# --- NLP (NO NLTK DOWNLOADS) ---
@st.cache_resource
def setup_nlp():
    # removed nltk.download completely (fixes SSL crash)
    stop_words = set(ENGLISH_STOP_WORDS)
    stemmer = PorterStemmer()
    return stop_words, stemmer


STOP_WORDS, STEMMER = setup_nlp()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()

    cleaned_words = [
        STEMMER.stem(w) for w in words if w not in STOP_WORDS
    ]

    return " ".join(cleaned_words)


# --- MODEL LOADING ---
@st.cache_resource
def load_ml_model():
    try:
        model = joblib.load("models/model.pkl")
        vectorizer = joblib.load("models/vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        return None, None


model, vectorizer = load_ml_model()


# --- UI ---
if model is None or vectorizer is None:
    st.error("⚠️ Model not found. Run: python3 src/train.py")
else:
    user_input = st.text_area("Enter your message content here:", height=150)

    if st.button("Analyze Text", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a message.")
        else:
            with st.spinner("Processing..."):

                cleaned = clean_text(user_input)
                vectorized_input = vectorizer.transform([cleaned])

                prediction = model.predict(vectorized_input)[0]
                probabilities = model.predict_proba(vectorized_input)[0]

                pred_index = list(model.classes_).index(prediction)
                confidence = float(probabilities[pred_index]) * 100

                st.subheader("Analysis Results:")

                if str(prediction).lower() in ["spam", "1", "true"]:
                    st.error(f"🚨 Spam Detected! ({confidence:.2f}% confidence)")
                else:
                    st.success(f"✅ Legitimate Message (Ham) ({confidence:.2f}% confidence)")