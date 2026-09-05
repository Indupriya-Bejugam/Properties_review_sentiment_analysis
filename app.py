from flask import Flask, render_template, request

from recommendation.recommender import recommend_properties


app = Flask(__name__)


# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return render_template(
        "index.html"
    )


# -------------------------------------------------
# RECOMMENDATION
# -------------------------------------------------

@app.route(
    "/recommend",
    methods=["POST"]
)
def recommend():

    # Get values from the form
    location = request.form.get(
        "location",
        ""
    ).strip()

    min_area_value = request.form.get(
        "min_area",
        ""
    ).strip()

    max_price_value = request.form.get(
        "max_price",
        ""
    ).strip()


    # Convert numeric inputs
    min_area = None

    if min_area_value:

        try:
            min_area = float(
                min_area_value
            )

        except ValueError:

            min_area = None


    max_price = None

    if max_price_value:

        try:
            max_price = float(
                max_price_value
            )

        except ValueError:

            max_price = None


    # Call recommendation engine
    recommendations = recommend_properties(
        location=location
        if location
        else None,

        min_area=min_area,

        max_price=max_price,

        top_n=5
    )


    # Convert DataFrame to records
    properties = (
        recommendations
        .to_dict(
            orient="records"
        )
    )


    return render_template(
        "results.html",
        properties=properties,
        location=location,
        min_area=min_area,
        max_price=max_price
    )


# -------------------------------------------------
# RUN APPLICATION
# -------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )