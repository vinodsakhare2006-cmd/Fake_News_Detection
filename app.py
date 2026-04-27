import pickle
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

MODEL_PATH = Path("model.pkl")
VECTORIZER_PATH = Path("vectorizer.pkl")

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def get_confidence(model, x):
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(x)[0])
        return min(abs(score) * 20, 99.9)

    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(x).max() * 100)

    return 75.0

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error("model.pkl or vectorizer.pkl not found. Keep both files in the same folder as app.py.")
    st.stop()

st.title("📰 Fake News Detection")
st.write("Enter news text below and check whether it is **Real** or **Fake**.")

sample_fake = (
    "NASA has unveiled plans to construct the world's first luxury hotel on the Sun by 2035. "
    "Officials claim advanced cooling technology will allow tourists to enjoy solar views."
)

sample_real = (
    "Scientists have developed an AI-powered tool to help doctors detect early signs of disease "
    "from medical scans. Early testing suggests the tool may improve screening speed."
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Try Fake Sample"):
        st.session_state.news_text = sample_fake

with col2:
    if st.button("Try Real Sample"):
        st.session_state.news_text = sample_real

news_text = st.text_area(
    "Enter News Content",
    value=st.session_state.get("news_text", ""),
    height=220
)

if st.button("Predict"):
    if not news_text.strip():
        st.warning("Please enter news content.")
    else:
        vectorized_text = vectorizer.transform([news_text])
        prediction = int(model.predict(vectorized_text)[0])
        confidence = get_confidence(model, vectorized_text)

        if prediction == 1:
            st.success("✅ Prediction: Real News")
        else:
            st.error("❌ Prediction: Fake News")

        st.progress(int(confidence))
        st.write(f"Confidence: **{confidence:.2f}%**")

st.markdown("---")
st.caption("Educational project. Always verify important news from trusted sources.")
