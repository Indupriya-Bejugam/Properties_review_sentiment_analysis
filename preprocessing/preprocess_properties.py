import pandas as pd
import re
import os


INPUT_PATH = "data/raw/magicbricks_properties.csv"
OUTPUT_PATH = "data/processed/properties_cleaned.csv"


def extract_number(value):

    if pd.isna(value):
        return 0.0

    text = str(value)

    match = re.search(r"[\d,.]+", text)

    if not match:
        return 0.0

    return float(
        match.group().replace(",", "")
    )


def convert_price_to_lakhs(value):

    if pd.isna(value):
        return 0.0

    text = str(value).lower()

    number = extract_number(text)

    if "crore" in text or " cr" in text:
        return number * 100

    if "lakh" in text or "lac" in text:
        return number

    return number


def clean_text(value):

    if pd.isna(value):
        return ""

    value = str(value)

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def preprocess_properties():

    # -----------------------------------------
    # 1. LOAD RAW SCRAPED DATA
    # -----------------------------------------

    df = pd.read_csv(INPUT_PATH)

    print("Original shape:", df.shape)

    # -----------------------------------------
    # 2. CLEAN TEXT
    # -----------------------------------------

    text_columns = [
        "title",
        "location",
        "property_type",
        "description",
        "developer",
        "nearby",
        "url"
    ]

    for column in text_columns:

        if column in df.columns:

            df[column] = df[column].apply(
                clean_text
            )

    # -----------------------------------------
    # 3. NUMERIC FEATURES
    # -----------------------------------------

    df["area_sqft"] = df["area"].apply(
        extract_number
    )

    df["price_lakhs"] = df["price"].apply(
        convert_price_to_lakhs
    )

    df["price_per_sqft_numeric"] = (
        df["price_per_sqft"].apply(
            extract_number
        )
    )

    # -----------------------------------------
    # 4. REMOVE DUPLICATES
    # -----------------------------------------

    before = len(df)

    # Only remove duplicate URLs when URL exists
    df = df.drop_duplicates(
        subset=["url"],
        keep="first"
    )

    print(
        "Duplicates removed:",
        before - len(df)
    )

    # -----------------------------------------
    # 5. KEEP PROPERTIES WITH AREA OR TEXT
    # -----------------------------------------

    # Do NOT require price because the scraped
    # page may not expose price information.

    df = df[
        (df["area_sqft"] > 0)
        | (df["title"].str.len() > 0)
        | (df["description"].str.len() > 0)
    ].copy()

    # -----------------------------------------
    # 6. RESET INDEX
    # -----------------------------------------

    df.reset_index(
        drop=True,
        inplace=True
    )

    # -----------------------------------------
    # 7. SAVE
    # -----------------------------------------

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nProcessed shape:",
        df.shape
    )

    print(
        "\nSaved to:",
        OUTPUT_PATH
    )

    print(
        "\nSample:"
    )

    print(
        df[
            [
                "title",
                "location",
                "area_sqft",
                "price_lakhs",
                "price_per_sqft_numeric"
            ]
        ].head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":

    preprocess_properties()