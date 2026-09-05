import joblib

from preprocessing.text_preprocessor import preprocess


MODEL_PATH = "models/sentiment_model.pkl"

VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


model = joblib.load(
    MODEL_PATH
)

vectorizer = joblib.load(
    VECTORIZER_PATH
)


def predict_sentiment(review):

    cleaned_review = preprocess(
        review
    )

    features = vectorizer.transform(
        [cleaned_review]
    )

    prediction = model.predict(
        features
    )[0]

    probabilities = model.predict_proba(
        features
    )[0]

    confidence = max(
        probabilities
    )

    return {
        "sentiment": (
            "positive"
            if prediction == 1
            else "negative"
        ),
        "confidence": float(
            confidence
        )
    }