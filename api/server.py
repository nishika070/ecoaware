import json
from flask import Flask, jsonify, g, redirect, render_template, request, url_for
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.aqi_service import (
    build_prediction_payload,
    get_aqi_page_context,
    get_available_stations,
    get_analysis_context,
    get_contact_page_context,
    get_home_context,
    get_policies_page_context,
)

from utils.station_config import DEFAULT_STATION, STATION_COORDINATES


app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static",
)


@app.context_processor
def inject_station_coordinates():
    return {"station_coordinates": STATION_COORDINATES}


@app.after_request
def inject_analysis_js(response):
    if request.path == "/analysis" and response.content_type and "text/html" in response.content_type:
        analysis_payload = getattr(g, "analysis_js", None)
        if analysis_payload:
            script = (
                "<script>"
                f"var analysis_data = {json.dumps(analysis_payload)};"
                "window.analysis_data = analysis_data;"
                "</script>"
            )
            body = response.get_data(as_text=True)
            marker = "<!-- ================= DATA FROM FLASK ================= -->"
            if marker in body:
                body = body.replace(marker, f"{script}{marker}", 1)
                response.set_data(body)
    return response


# HOME
@app.route("/")
def home():
    selected_station = request.args.get("station") or DEFAULT_STATION
    return render_template(
        "home.html",
        active_page="home",
        **get_home_context(selected_station),
    )


# AQI PAGE
@app.route("/aqi")
def aqi_page():
    station = request.args.get("station", "all")
    return render_template(
        "aqi.html",
        active_page="aqi",
        **get_aqi_page_context(station),
    )


# TEMPERATURE
@app.route("/temperature")
def temperature_page():
    selected_station = request.args.get("station") or DEFAULT_STATION
    return redirect(url_for("home", station=selected_station))


# POLICY PAGE
@app.route("/policies")
def policies_page():
    return render_template(
        "policy_insights.html",
        active_page="policies",
        **get_policies_page_context(),
    )


# CONTACT PAGE
@app.route("/contact")
def contact_page():
    return render_template(
        "contact.html",
        **get_contact_page_context(),
    )


# ANALYSIS PAGE
@app.route("/analysis")
def analysis_page():
    context = get_analysis_context("all")
    g.analysis_js = context.get("analysis_data", {})
    return render_template(
        "analysis.html",
        active_page="analysis",
        **context,
    )


# API
@app.route("/api/aqi")
def get_aqi_api():
    return jsonify(build_prediction_payload())


# RUN
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
