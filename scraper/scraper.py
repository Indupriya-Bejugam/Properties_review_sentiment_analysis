import requests
from bs4 import BeautifulSoup
import pandas as pd


def scrape_reviews(url):

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=15
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    reviews = []

    # Generic selectors. Adjust for the permitted
    # website/page being scraped.
    selectors = [
        ".review",
        ".review-text",
        ".reviewText",
        "[class*='review']"
    ]

    elements = []

    for selector in selectors:
        elements.extend(
            soup.select(selector)
        )

    seen = set()

    for element in elements:

        text = element.get_text(
            " ",
            strip=True
        )

        if (
            text
            and len(text) >= 20
            and text not in seen
        ):
            reviews.append(text)
            seen.add(text)

    return reviews


def save_scraped_reviews(
    reviews,
    output_path
):

    df = pd.DataFrame({
        "review": reviews
    })

    df.to_csv(
        output_path,
        index=False
    )

    return df