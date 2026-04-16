from flask import Flask, jsonify, render_template
from flask import request

from aqi_service import (
    build_prediction_payload,
    get_aqi_page_context,
    get_contact_page_context,
    get_home_context,
    get_policies_page_context,
    get_temperature_page_context,
)


app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static",
)

@app.route("/")
def home():
    selected_station = request.args.get("station")
    return render_template(
        "home.html",
        active_page="home",
        **get_home_context(selected_station),
    )


@app.route("/aqi")
def aqi_page():
    return render_template(
        "aqi.html",
        active_page="aqi",
        **get_aqi_page_context(),
    )


@app.route("/temperature")
def temperature_page():
    selected_station = request.args.get("station")
    return render_template(
        "temperature.html",
        active_page="temperature",
        **get_temperature_page_context(selected_station),
    )


@app.route("/policies")
def policies_page():
    return render_template(
        "policies.html",
        active_page="policies",
        **get_policies_page_context(),
    )


@app.route("/contact")
def contact_page():
    return render_template(
        "contact.html",
        active_page="contact",
        **get_contact_page_context(),
    )


@app.route("/api/aqi")
def get_aqi_api():
    return jsonify(build_prediction_payload())


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
