"""
Customer Sentiment Analysis — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import joblib
import re
import string
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎭",
    layout="centered",
)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model      = joblib.load("models/sentiment_model.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    scaler     = joblib.load("models/standard_scaler.joblib")
    return model, vectorizer, scaler

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

EMOJI = {"positive": "😊", "negative": "😠", "neutral": "😐"}
COLOR = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#3498db"}

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎭 Customer Sentiment Analyzer")
st.markdown("Paste a tweet or customer review below to classify its sentiment.")

if not os.path.exists("models/sentiment_model.joblib"):
    st.error("Model files not found. Please run the notebook first to generate `models/`.")
    st.stop()

model, vectorizer, scaler = load_models()

user_input = st.text_area("Enter text:", placeholder="e.g. I love this product!", height=120)

if st.button("Analyze Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        cleaned = clean_text(user_input)
        vec     = vectorizer.transform([cleaned])
        scaled  = scaler.transform(vec)
        label   = model.predict(scaled)[0]
        probs   = model.predict_proba(scaled)[0]
        classes = model.classes_

        st.markdown("---")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(
                f"<div style='text-align:center; font-size:64px'>{EMOJI[label]}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<h3 style='text-align:center; color:{COLOR[label]}'>{label.upper()}</h3>",
                unsafe_allow_html=True,
            )

        with col2:
            st.subheader("Confidence Scores")
            for cls, prob in sorted(zip(classes, probs), key=lambda x: -x[1]):
                st.progress(float(prob), text=f"{cls}: {prob:.1%}")

        with st.expander("Cleaned text (what the model sees)"):
            st.code(cleaned)

# ── Batch mode ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📂 Batch Analysis")
uploaded = st.file_uploader("Upload a CSV with a `text` column", type="csv")

if uploaded:
    import pandas as pd
    df = pd.read_csv(uploaded)
    if "text" not in df.columns:
        st.error("CSV must have a `text` column.")
    else:
        df["cleaned_text"] = df["text"].apply(clean_text)
        vecs   = vectorizer.transform(df["cleaned_text"])
        scaled = scaler.transform(vecs)
        df["predicted_sentiment"] = model.predict(scaled)

        st.dataframe(df[["text", "predicted_sentiment"]].head(20))

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results", csv, "predictions.csv", "text/csv")
