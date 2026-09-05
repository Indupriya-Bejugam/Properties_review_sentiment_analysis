import os
import joblib
import pandas as pd
import numpy as np

from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------
# PATHS
# -------------------------------------------------

DATA_PATH = "data/processed/properties_cleaned.csv"
MODEL_DIR = "models"


# -------------------------------------------------
# LOAD TRAINED MODELS
# -------------------------------------------------

tfidf = joblib.load(
    os.path.join(
        MODEL_DIR,
        "property_tfidf.pkl"
    )
)

scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "property_scaler.pkl"
    )
)

kmeans = joblib.load(
    os.path.join(
        MODEL_DIR,
        "property_kmeans.pkl"
    )
)


# -------------------------------------------------
# LOAD PROCESSED DATA
# -------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Loaded properties:", len(df))


# -------------------------------------------------
# PREPARE TEXT
# -------------------------------------------------

text_columns = [
    "title",
    "location",
    "description",
    "developer",
    "nearby"
]

for column in text_columns:

    if column in df.columns:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
        )


df["combined_text"] = (
    df["title"] + " "
    + df["location"] + " "
    + df["description"] + " "
    + df["developer"] + " "
    + df["nearby"]
)


# -------------------------------------------------
# CREATE TF-IDF FEATURES
# -------------------------------------------------

text_features = tfidf.transform(
    df["combined_text"]
)


# -------------------------------------------------
# CREATE NUMERICAL FEATURES
#
# IMPORTANT:
# These must be the same 3 features used
# during training.
# -------------------------------------------------

numeric_columns = [
    "area_sqft",
    "price_lakhs",
    "price_per_sqft_numeric"
]

numeric_data = df[
    numeric_columns
].fillna(0)


numeric_features = scaler.transform(
    numeric_data
)


# -------------------------------------------------
# COMBINE FEATURES
#
# TF-IDF = 1134 features
# Numeric = 3 features
# Total = 1137 features
# -------------------------------------------------

combined_features = hstack([
    text_features,
    numeric_features
])


# -------------------------------------------------
# RECOMMENDATION FUNCTION
# -------------------------------------------------

def recommend_properties(
    location=None,
    min_area=None,
    max_price=None,
    top_n=5
):

    candidates = df.copy()

    # ---------------------------------------------
    # 1. FILTER BY MINIMUM AREA
    # ---------------------------------------------

    if min_area is not None:

        candidates = candidates[
            candidates["area_sqft"] >= min_area
        ]


    # ---------------------------------------------
    # 2. FILTER BY MAXIMUM PRICE
    #
    # price_lakhs = 0 means price was unavailable.
    # We don't reject such properties.
    # ---------------------------------------------

    if max_price is not None:

        candidates = candidates[
            (candidates["price_lakhs"] == 0)
            |
            (candidates["price_lakhs"] <= max_price)
        ]


    # ---------------------------------------------
    # 3. PREFER USER'S LOCATION
    # ---------------------------------------------

    if location:

        location_matches = candidates[
            candidates["location"]
            .str.lower()
            .str.contains(
                str(location).lower(),
                na=False
            )
        ]

        if not location_matches.empty:

            candidates = location_matches


    # ---------------------------------------------
    # 4. CHECK IF ANY CANDIDATES EXIST
    # ---------------------------------------------

    if candidates.empty:

        return pd.DataFrame()


    # ---------------------------------------------
    # 5. GET ORIGINAL DATAFRAME INDICES
    # ---------------------------------------------

    candidate_indices = candidates.index.tolist()


    # ---------------------------------------------
    # 6. GET CANDIDATE FEATURES
    # ---------------------------------------------

    candidate_text_features = text_features[
        candidate_indices
    ]

    candidate_numeric_features = numeric_features[
        candidate_indices
    ]


    # IMPORTANT:
    # K-Means was trained using 1137 features,
    # so prediction must also receive 1137.
    
    candidate_features = hstack([
        candidate_text_features,
        candidate_numeric_features
    ])


    # ---------------------------------------------
    # 7. BUILD USER QUERY
    # ---------------------------------------------

    query_parts = []

    if location:

        query_parts.append(
            str(location)
        )

    query_parts.append(
        "residential plot"
    )

    if min_area:

        query_parts.append(
            f"{min_area} sqft"
        )


    query_text = " ".join(
        query_parts
    )


    # ---------------------------------------------
    # 8. CONVERT USER QUERY INTO TF-IDF
    # ---------------------------------------------

    query_vector = tfidf.transform(
        [query_text]
    )


    # ---------------------------------------------
    # 9. CALCULATE TEXT SIMILARITY
    # ---------------------------------------------

    similarities = cosine_similarity(
        query_vector,
        candidate_text_features
    ).flatten()


    # ---------------------------------------------
    # 10. PREDICT PROPERTY CLUSTERS
    # ---------------------------------------------

    candidate_clusters = kmeans.predict(
        candidate_features
    )


    # ---------------------------------------------
    # 11. FIND MOST COMMON CLUSTER
    # ---------------------------------------------

    if len(candidate_clusters) > 0:

        cluster_counts = np.bincount(
            candidate_clusters
        )

        preferred_cluster = np.argmax(
            cluster_counts
        )

    else:

        preferred_cluster = None


    # ---------------------------------------------
    # 12. CALCULATE FINAL SCORE
    # ---------------------------------------------

    scores = []


    for i, index in enumerate(
        candidate_indices
    ):

        score = 0.0


        # -----------------------------------------
        # TEXT SIMILARITY
        # Maximum contribution = 60
        # -----------------------------------------

        score += (
            similarities[i] * 60
        )


        # -----------------------------------------
        # LOCATION MATCH
        # Maximum contribution = 20
        # -----------------------------------------

        if location:

            property_location = str(
                df.loc[index, "location"]
            ).lower()

            if str(location).lower() in property_location:

                score += 20


        # -----------------------------------------
        # AREA MATCH
        # Maximum contribution = 10
        # -----------------------------------------

        if min_area:

            property_area = float(
                df.loc[index, "area_sqft"]
            )

            if property_area >= min_area:

                score += 10


        # -----------------------------------------
        # CLUSTER MATCH
        # Maximum contribution = 10
        # -----------------------------------------

        if (
            preferred_cluster is not None
            and candidate_clusters[i]
            == preferred_cluster
        ):

            score += 10


        scores.append(score)


    # ---------------------------------------------
    # 13. ADD SCORE TO CANDIDATES
    # ---------------------------------------------

    candidates = candidates.copy()

    candidates[
        "recommendation_score"
    ] = scores


    # ---------------------------------------------
    # 14. SORT BY SCORE
    # ---------------------------------------------

    recommendations = (
        candidates
        .sort_values(
            "recommendation_score",
            ascending=False
        )
        .head(top_n)
    )


    # ---------------------------------------------
    # 15. RETURN USEFUL COLUMNS
    # ---------------------------------------------

    return recommendations[
        [
            "title",
            "location",
            "area",
            "price",
            "price_per_sqft",
            "description",
            "nearby",
            "url",
            "recommendation_score"
        ]
    ]


# -------------------------------------------------
# TEST THE RECOMMENDER
# -------------------------------------------------

if __name__ == "__main__":

    results = recommend_properties(
        location="Maheshwaram",
        min_area=1500,
        max_price=80,
        top_n=5
    )

    print(
        "\n=============================="
    )

    print(
        "Recommended Properties"
    )

    print(
        "==============================\n"
    )


    if results.empty:

        print(
            "No properties found."
        )

    else:

        print(
            results.to_string(
                index=False
            )
        )