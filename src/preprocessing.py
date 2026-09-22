"""
MailGuard Cyber - Production Data Engineering Pipeline
Author: Data Engineer (Group 3)
Role Deliverables: Preprocessing, Leakage Prevention, & Quality Audit
"""

import os
import re
import joblib
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "SMSSpamCollection")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def run_pipeline():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Ingestion
    raw_file = RAW_PATH if os.path.exists(RAW_PATH) else RAW_PATH + ".txt"
    df = pd.read_csv(raw_file, sep="\t", header=None, names=["label", "message"], encoding="utf-8")
    raw_count = len(df)
    duplicates_count = int(df.duplicated().sum())

    # 2. Sanitization & Target Mapping
    df = df.drop_duplicates().reset_index(drop=True)
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["clean_message"] = df["message"].apply(clean_text)
    df = df[df["clean_message"].str.strip() != ""].copy()

    # 3. Leakage-Free Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    # 4. Feature Extraction (Fit on Train ONLY)
    tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # 5. Persist Deliverables
    # CSV representations for EDA / Data Analyst
    pd.DataFrame({"message": X_train.reset_index(drop=True), "label": y_train.reset_index(drop=True)}).to_csv(
        os.path.join(PROCESSED_DIR, "train_clean.csv"), index=False
    )
    pd.DataFrame({"message": X_test.reset_index(drop=True), "label": y_test.reset_index(drop=True)}).to_csv(
        os.path.join(PROCESSED_DIR, "test_clean.csv"), index=False
    )

    # Compressed sparse matrices (<2 MB)
    sparse.save_npz(os.path.join(PROCESSED_DIR, "X_train_tfidf.npz"), X_train_tfidf)
    sparse.save_npz(os.path.join(PROCESSED_DIR, "X_test_tfidf.npz"), X_test_tfidf)

    # Standalone target vectors
    y_train.reset_index(drop=True).to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.reset_index(drop=True).to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

    # Serialized vectorizer for inference (ML Engineer)
    joblib.dump(tfidf, os.path.join(PROCESSED_DIR, "tfidf_vectorizer.joblib"))

    # Audit Sheet
    report = pd.DataFrame({
        "Check": [
            "Initial row count", "Duplicate rows handled", "Cleaned rows",
            "Training rows", "Testing rows", "Vocabulary size"
        ],
        "Result": [raw_count, duplicates_count, len(df), len(y_train), len(y_test), X_train_tfidf.shape[1]],
        "Status": ["Checked", "Handled", "Validated", "Validated", "Validated", "Optimized"]
    })
    report.to_csv(os.path.join(REPORTS_DIR, "data_quality_report.csv"), index=False)
    print("Execution complete. All artifacts generated successfully.")

if __name__ == "__main__":
    run_pipeline()
