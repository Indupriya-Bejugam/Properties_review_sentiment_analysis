import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from urllib.parse import urljoin


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9"
}


def clean_text(text):
    if not text:
        return ""

    return re.sub(r"\s+", " ", text).strip()


def scrape_magicbricks(url):

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # Actual class found in your debug.html
    cards = soup.select(
        "div.mb-srp__card__container"
    )

    print("Cards found:", len(cards))

    properties = []

    for card in cards:

        data = {
            "title": "",
            "location": "",
            "property_type": "",
            "area": "",
            "price": "",
            "price_per_sqft": "",
            "description": "",
            "developer": "",
            "url": "",
            "nearby": ""
        }

        # --------------------------------
        # TITLE
        # --------------------------------

        title = card.select_one(
            "h2.mb-srp__card--title"
        )

        if title:
            data["title"] = clean_text(
                title.get_text()
            )

        # --------------------------------
        # DEVELOPER / PROJECT
        # --------------------------------

        developer = card.select_one(
            ".mb-srp__card__developer--name"
        )

        if developer:
            data["developer"] = clean_text(
                developer.get_text()
            )

        # --------------------------------
        # PROPERTY AREA
        # --------------------------------

        area_item = card.select_one(
            '[data-summary="plot-area"]'
        )

        if area_item:

            value = area_item.select_one(
                ".mb-srp__card__summary--value"
            )

            if value:
                data["area"] = clean_text(
                    value.get_text()
                )

        # --------------------------------
        # PRICE
        # --------------------------------

        price = card.select_one(
            ".mb-srp__card__price--amount"
        )

        if price:
            data["price"] = clean_text(
                price.get_text()
            )

        # --------------------------------
        # PRICE PER SQFT
        # --------------------------------

        price_sqft = card.select_one(
            ".mb-srp__card__price--size"
        )

        if price_sqft:
            data["price_per_sqft"] = clean_text(
                price_sqft.get_text()
            )

        # --------------------------------
        # DESCRIPTION
        # --------------------------------

        description = card.select_one(
            ".mb-srp__card--desc--text"
        )

        if description:
            data["description"] = clean_text(
                description.get_text()
            )

        # --------------------------------
        # PROPERTY URL
        # --------------------------------

        link = card.select_one(
            "a.view-property-link"
        )

        if link and link.get("href"):

            data["url"] = urljoin(
                url,
                link["href"]
            )

        # --------------------------------
        # NEARBY LOCATIONS
        # --------------------------------

        nearby = card.select(
            ".mb-srp-m__card__nearby__tag--item"
        )

        if nearby:

            data["nearby"] = ", ".join(
                clean_text(x.get_text())
                for x in nearby
            )

        # --------------------------------
        # LOCATION
        # --------------------------------

        # Location is also contained in title,
        # e.g. "Residential Land / Plot in
        # Maheshwaram, Hyderabad"

        title_text = data["title"]

        if " in " in title_text:

            data["location"] = (
                title_text.split(
                    " in ",
                    1
                )[1]
            )

        # Keep records with useful information
        if (
            data["title"]
            or data["location"]
            or data["price"]
        ):

            properties.append(data)

    return properties


def scrape_pages(
    urls,
    output_file="data/raw/magicbricks_properties.csv"
):

    all_properties = []

    seen_urls = set()

    for url in urls:

        print("\nScraping:")
        print(url)

        try:

            properties = scrape_magicbricks(
                url
            )

            for property_data in properties:

                property_url = property_data[
                    "url"
                ]

                # Deduplicate
                if (
                    property_url
                    and property_url in seen_urls
                ):
                    continue

                if property_url:
                    seen_urls.add(
                        property_url
                    )

                all_properties.append(
                    property_data
                )

            print(
                "Records collected:",
                len(properties)
            )

        except Exception as error:

            print(
                "Error:",
                error
            )

        # Delay between requests
        time.sleep(2)

    df = pd.DataFrame(
        all_properties
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        "\n=========================="
    )

    print(
        "Total records:",
        len(df)
    )

    print(
        "Saved to:",
        output_file
    )

    print(
        "=========================="
    )

    return df


if __name__ == "__main__":

    urls = [
        "https://www.magicbricks.com/property-for-sale/residential-real-estate?bedroom=&proptype=Residential-Plot&cityName=Hyderabad"
    ]

    scrape_pages(urls)