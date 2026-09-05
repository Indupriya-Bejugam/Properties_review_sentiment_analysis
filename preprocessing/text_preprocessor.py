import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer


# Run once if resources are not available
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


STEMMER = SnowballStemmer("english")


def preprocess(text):

    text = str(text)

    # Remove HTML
    text = re.sub(r"<.*?>", " ", text)

    # Lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    words = [
        word for word in words
        if word not in STOP_WORDS
    ]

    # Stemming
    words = [
        STEMMER.stem(word)
        for word in words
    ]

    return " ".join(words)