import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(file_path):
    """Load the raw SMS Spam Collection dataset."""

    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    return df


def validate_data(df):
    """Validate the dataset schema and check for missing values."""

    expected_columns = ["label", "message"]

    if list(df.columns) != expected_columns:
        raise ValueError(
            f"Invalid schema. Expected columns: {expected_columns}"
        )

    if df["label"].isna().any():
        raise ValueError("Missing values found in label column.")

    if df["message"].isna().any():
        raise ValueError("Missing values found in message column.")

    # Check whether the same message has conflicting labels
    conflicting_labels = (
        df.groupby("message")["label"]
        .nunique()
        .gt(1)
        .sum()
    )

    if conflicting_labels > 0:
        raise ValueError(
            f"Found {conflicting_labels} messages with conflicting labels."
        )

    return df


def clean_text(text):
    """Clean SMS text while preserving useful spam indicators."""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_duplicates(df):
    """Remove exact duplicate records."""

    df = df.drop_duplicates().reset_index(drop=True)

    return df


def prepare_data(df):
    """Validate, clean, and prepare the dataset."""

    df = df.copy()

    # Validate raw data before transformation
    df = validate_data(df)

    # Remove exact duplicate records
    df = remove_duplicates(df)

    # Clean message text
    df["message"] = df["message"].apply(clean_text)

    return df


def split_data(df):
    """Create a reproducible stratified train-test split."""

    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def vectorize_data(X_train, X_test):
    """Apply leakage-safe TF-IDF vectorization."""

    tfidf = TfidfVectorizer()

    # Learn vocabulary and IDF only from training data
    X_train_tfidf = tfidf.fit_transform(X_train)

    # Transform test data using training vocabulary and IDF
    X_test_tfidf = tfidf.transform(X_test)

    return X_train_tfidf, X_test_tfidf, tfidf
