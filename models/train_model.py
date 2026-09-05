import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from preprocessing.text_preprocessor import preprocess


DATA_PATH = "data/raw/training_data.csv"

MODEL_PATH = "models/sentiment_model.pkl"

VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


def train():

    df = pd.read_csv(DATA_PATH)

    if "review" not in df.columns:
        raise ValueError(
            "Dataset must contain 'review' column"
        )

    if "sentiment" not in df.columns:
        raise ValueError(
            "Dataset must contain 'sentiment' column"
        )

    # Remove missing values
    df = df.dropna(
        subset=["review", "sentiment"]
    )

    # Convert sentiment to 0/1
    df["sentiment"] = (
        df["sentiment"]
        .astype(str)
        .str.lower()
        .map({
            "positive": 1,
            "negative": 0,
            "1": 1,
            "0": 0
        })
    )

    df = df.dropna(
        subset=["sentiment"]
    )

    df["sentiment"] = df[
        "sentiment"
    ].astype(int)

    # NLP preprocessing
    df["clean_review"] = (
        df["review"]
        .apply(preprocess)
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            df["clean_review"],
            df["sentiment"],
            test_size=0.2,
            random_state=42,
            stratify=df["sentiment"]
        )
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )

    # Logistic Regression
    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train_tfidf,
        y_train
    )

    # Prediction
    predictions = model.predict(
        X_test_tfidf
    )

    print("\n===== MODEL RESULTS =====")

    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            predictions
        )
    )

    print(
        "Precision:",
        precision_score(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print(
        "Recall:",
        recall_score(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print(
        "F1:",
        f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    # Save model
    joblib.dump(
        model,
        MODEL_PATH
    )

    # Save vectorizer
    joblib.dump(
        vectorizer,
        VECTORIZER_PATH
    )

    print("\nModel saved.")
    

if __name__ == "__main__":
    train()