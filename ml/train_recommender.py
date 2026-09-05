import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.sparse import hstack


DATA_PATH = "data/processed/properties_cleaned.csv"

MODEL_DIR = "models"

TFIDF_PATH = os.path.join(
    MODEL_DIR,
    "property_tfidf.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "property_scaler.pkl"
)

KMEANS_PATH = os.path.join(
    MODEL_DIR,
    "property_kmeans.pkl"
)


# ---------------------------------------------
# 1. LOAD PROCESSED PROPERTY DATA
# ---------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ---------------------------------------------
# 2. PREPARE TEXT
# ---------------------------------------------

text_columns = [
    "title",
    "location",
    "description",
    "developer",
    "nearby"
]

for column in text_columns:

    if column in df.columns:
        df[column] = df[column].fillna("")


df["combined_text"] = (
    df["title"] + " "
    + df["location"] + " "
    + df["description"] + " "
    + df["developer"] + " "
    + df["nearby"]
)


# ---------------------------------------------
# 3. TF-IDF
# ---------------------------------------------

tfidf = TfidfVectorizer(
    max_features=3000,
    stop_words="english",
    ngram_range=(1, 2)
)

text_features = tfidf.fit_transform(
    df["combined_text"]
)

print(
    "TF-IDF shape:",
    text_features.shape
)


# ---------------------------------------------
# 4. NUMERICAL FEATURES
# ---------------------------------------------

numeric_columns = [
    "area_sqft",
    "price_lakhs",
    "price_per_sqft_numeric"
]

numeric_data = df[
    numeric_columns
].fillna(0)


# Standardize numerical features
scaler = StandardScaler()

numeric_features = scaler.fit_transform(
    numeric_data
)


# ---------------------------------------------
# 5. COMBINE FEATURES
# ---------------------------------------------

combined_features = hstack([
    text_features,
    numeric_features
])


print(
    "Combined feature shape:",
    combined_features.shape
)


# ---------------------------------------------
# 6. K-MEANS
# ---------------------------------------------

# Number of property groups
n_clusters = min(5, len(df))

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(
    combined_features
)

df["cluster"] = clusters


# ---------------------------------------------
# 7. SAVE MODELS
# ---------------------------------------------

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    tfidf,
    TFIDF_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)

joblib.dump(
    kmeans,
    KMEANS_PATH
)


# Save clustered dataset
df.to_csv(
    DATA_PATH,
    index=False
)


print("\nTraining completed.")

print(
    "TF-IDF saved:",
    TFIDF_PATH
)

print(
    "Scaler saved:",
    SCALER_PATH
)

print(
    "K-Means saved:",
    KMEANS_PATH
)

print("\nCluster distribution:")

print(
    df["cluster"].value_counts()
)