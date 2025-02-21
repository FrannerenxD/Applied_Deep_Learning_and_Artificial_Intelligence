# streamlit_app.py

import streamlit as st
from transformers import pipeline

# Load Hugging Face text classification pipeline
@st.cache_resource  # Cache the model to avoid reloading it on every refresh
def load_classification_pipeline():
    classifier = pipeline("text-classification", model="Driisa/finbert-finetuned-github")
    return classifier

classifier = load_classification_pipeline()

# Streamlit app UI
st.title("📊 Assignment 2: Multi-class Text Classification")
st.markdown(
    """
    This app uses **finbert-finetuned-github** to classify text into relevant categories.
    Just type your financial text in the box below and click **Classify** to see the results.
    """
)

# User input
st.text_area("📝 Enter your text below:", placeholder="Type your text here...", key="user_input")

# Classify button
if st.button("🔍 Classify"):
    user_input = st.session_state.user_input.strip()
    
    if user_input:
        with st.spinner("🚀 Classifying..."):
            results = classifier(user_input)
            label = results[0]["label"]
            score = results[0]["score"]

        # Display results with progress bar and formatting
        st.subheader("🎯 Classification Result:")
        st.write(f"**Prediction:** `{label}`")
        st.write(f"**Confidence Score:** `{score:.4f}`")

        # Add a progress bar to visualize confidence
        st.progress(score)

    else:
        st.warning("⚠️ Please enter some text before classifying.")

