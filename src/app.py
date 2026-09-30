import streamlit as st
import joblib
import re

st.set_page_config(page_title="MailGuard | Spam Detection", layout="centered")

# Load your exported files
try:
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    model = joblib.load('spam_classifier.pkl')
except Exception as e:
    st.error("Error: Could not find the .joblib or .pkl files. Make sure you exported them!")

# The Data Engineering rules
def clean_and_validate_sms(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "ERROR"
    if len(text.strip()) <= 2:
        return "BYPASS_HAM"

    text = re.sub(r'\+?\d{10,15}', '[PHONE]', text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\[\]]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# Build the Webpage
st.title("🛡️ MailGuard")
st.subheader("Deterministic SMS Spam Detection")

user_input = st.text_area("Enter SMS text below:", height=150)

if st.button("Classify Message", type="primary"):
    if user_input:
        cleaned_text = clean_and_validate_sms(user_input)

        if cleaned_text == "ERROR":
            st.error("Invalid input.")
        elif cleaned_text == "BYPASS_HAM":
            st.success("✅ **HAM (Legitimate)** - Message too short for spam.")
        else:
            # Transform text and predict
            vectorized_input = vectorizer.transform([cleaned_text])
            probabilities = model.predict_proba(vectorized_input)[0]
            spam_prob = probabilities[1]

            if spam_prob > 0.50:
                st.error(f"🚨 **SPAM (Blocked)** - {spam_prob * 100:.1f}% Confidence")
            else:
                st.success(f"✅ **HAM (Legitimate)** - {spam_prob * 100:.1f}% Confidence")
