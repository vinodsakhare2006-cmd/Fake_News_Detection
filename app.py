import pickle
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="Fake News Detection Pro",
    page_icon="📰",
    layout="wide"
)

page_bg = """
<style>
.stApp {
    background-image:
        linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.75)),
        url("https://images.unsplash.com/photo-1504711434969-e33886168f5c");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.main-card {
    background: rgba(255, 255, 255, 0.13);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 24px;
    padding: 35px;
    border: 1px solid rgba(255,255,255,0.25);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    animation: fadeIn 1s ease-in-out;
}

.title {
    font-size: 48px;
    font-weight: 800;
    color: white;
    text-align: center;
}

.subtitle {
    color: #e5e7eb;
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
}

.stTextArea textarea {
    background: rgba(255,255,255,0.92) !important;
    color: #111827 !important;
    border-radius: 15px !important;
}

.stButton button {
    width: 100%;
    border-radius: 14px;
    padding: 12px;
    font-weight: 700;
    background: linear-gradient(90deg, #ef4444, #f97316);
    color: white;
    border: none;
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 18px rgba(249,115,22,0.8);
}

.result-box {
    background: rgba(255,255,255,0.16);
    backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: 22px;
    margin-top: 20px;
    border: 1px solid rgba(255,255,255,0.25);
    color: white;
    text-align: center;
}

.footer {
    color: #d1d5db;
    text-align: center;
    margin-top: 30px;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(25px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

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
except Exception:
    st.error("❌ model.pkl or vectorizer.pkl not found. Keep both files with app.py.")
    st.stop()

st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown('<div class="title">📰 Fake News Detection Pro</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered news authenticity checker using NLP and Machine Learning</div>',
    unsafe_allow_html=True
)

sample_fake = "NASA plans to build a luxury hotel on the Sun by 2035."
sample_real = "Scientists develop AI tool for early disease detection."

col1, col2 = st.columns(2)

with col1:
    if st.button("❌ Try Fake Sample"):
        st.session_state.news_text = sample_fake

with col2:
    if st.button("✅ Try Real Sample"):
        st.session_state.news_text = sample_real

news_text = st.text_area(
    "Enter News Content",
    value=st.session_state.get("news_text", ""),
    height=230
)

if st.button("🚀 Predict News"):
    if not news_text.strip():
        st.warning("Please enter news content.")
    else:
        vectorized_text = vectorizer.transform([news_text])
        prediction = int(model.predict(vectorized_text)[0])
        confidence = get_confidence(model, vectorized_text)

        if prediction == 1:
            result = "✅ Real News"
        else:
            result = "❌ Fake News"

        st.markdown(
            f"""
            <div class="result-box">
                <h2>{result}</h2>
                <h3>Confidence: {confidence:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(int(confidence))

st.markdown(
    '<div class="footer">⚠️ Educational project. Always verify important news from trusted sources.</div>',
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)
