import os
import joblib

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "spam_classifier.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR, "data", "processed", "tfidf_vectorizer.joblib"
)

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_message(message: str):
    """
    Predict whether an SMS message is Ham or Spam.

    Returns:
        label: "Ham" or "Spam"
        spam_probability: probability that the message is Spam
    """
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)[0]
    probabilities = model.predict_proba(message_tfidf)[0]

    if prediction == 1:
        label = "Spam"
    else:
        label = "Ham"

    spam_probability = probabilities[1]

    return label, spam_probability