import argparse
import json
import os
import pickle
import sys
import time
from typing import Optional
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter


from station_map import STATION_COORDINATES


WAQI_GEO_BASE_URL = "https://api.waqi.info/feed/geo:{lat};{lng}/?token={token}"
FORECAST_BASE_URL = "https://api.open-meteo.com/v1/forecast"

MODEL_PATH = "aqi_policy_model.pkl"
MLB_PATH = "aqi_policy_mlb.pkl"        # MultiLabelBinarizer for policy encoding
OUTPUT_JSON = "predictions.json"

FEATURES = ["T2M", "PRECTOTCORR", "WS10M", "PS", "AQI",
            "Location", "hasSprinkler", "isIndustrial"]

POLICY_MAP = {
    0:  "No immediate action required",
    1:  "Water sprinklers",
    2:  "Odd-even vehicle scheme",
    3:  "GRAP 3/4",
    4:  "Run industrial checks for fire, ash and dust control measures + fine imposition",
    5:  "Suspend outdoor activities in schools + online / work-from-home for schools, colleges and offices",
    6:  "Suspend construction activities temporarily",
    7:  "Temporary water distribution points (Heatwave)",
    8:  "Pre-emptive closure of vulnerable roads and metro stations (Flood)",
    9:  "Controlled release from reservoirs to prevent overflow (Flood)",
    10: "Distribution of blankets and warm clothing for unhoused populations (Cold Wave)",
    11: "Suspension of port / airport operations (Storm / Cyclone)",
    12: "Deployment of emergency power backup systems (Storm / Cyclone)",
}

LOCATION_MAP = {
    'Anand_Vihar_Delhi_DPCC': 0,       'Ashok_Vihar_Delhi_DPCC': 1,
    'Bawana_Delhi_DPCC': 2,             'CRRI_Mathura_Road_Delhi_IMD': 3,
    'DTU_Delhi_CPCB': 4,                'IGI_Airport_(T3)_Delhi_IMD': 5,
    'ITO_Delhi_CPCB': 6,                'Jahangirpuri_Delhi_DPCC': 7,
    'Jawaharlal_Nehru_Stadium_Delhi_DPCC': 8, 'Lodhi_Road_Delhi_IMD': 9,
    'Narela_Delhi_DPCC': 10,            'North_Campus_DU_Delhi_IMD': 11,
    'NSIT_Dwarka_Delhi_CPCB': 12,       'Okhla_Phase-2_Delhi_DPCC': 13,
    'Patparganj_Delhi_DPCC': 14,        'Punjabi_Bagh_Delhi_DPCC': 15,
    'Rohini_Delhi_DPCC': 16,            'Sirifort_Delhi_CPCB': 17,
    'Vivek_Vihar_Delhi_DPCC': 18,       'Wazirpur_Delhi_DPCC': 19,
}

STATION_TO_LOCATION_ID = {
    "Anand Vihar": 0,          "Ashok Vihar": 1,
    "Bawana": 2,                "CRRI Mathura Road": 3,
    "DTU": 4,                   "IGI Airport T3": 5,
    "ITO": 6,                   "Jahangirpuri": 7,
    "Jawaharlal Nehru Stadium": 8, "Lodhi Road": 9,
    "Narela": 10,               "North Campus DU": 11,
    "NSIT Dwarka": 12,          "Okhla Phase-2": 13,
    "Patparganj": 14,           "Punjabi Bagh": 15,
    "Rohini": 16,               "Sirifort": 17,
    "Vivek Vihar": 18,          "Wazirpur": 19,
}

INDUSTRIAL_IDS  = {2, 7, 10, 13, 19}   # Bawana, Jahangirpuri, Narela, Okhla, Wazirpur
SPRINKLER_IDS   = {5, 8, 9, 11, 17}    # IGI, JN Stadium, Lodhi, N-Campus DU, Sirifort


def build_multilabel_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, MultiLabelBinarizer]:

    def expand(row):
        base = int(row["policy_suggestion"])
        labels = {base}
        aqi = row["AQI"]
        is_ind = int(row["isIndustrial"])

        if base in (8, 9):          labels |= {8, 9}
        if base in (11, 12):        labels |= {11, 12}
        if aqi > 400:               labels |= {3, 5}
        elif aqi > 300 and is_ind:  labels |= {4, 2}
        elif aqi > 300:             labels |= {3, 6}

        return sorted(labels)

    df = df.copy()
    df["labels"] = df.apply(expand, axis=1)

    mlb = MultiLabelBinarizer(classes=sorted(POLICY_MAP.keys()))
    Y = mlb.fit_transform(df["labels"])
    Y_df = pd.DataFrame(Y, columns=mlb.classes_)
    return Y_df, mlb


def train(csv_path: str) -> None:
    print(f"[train] Loading dataset from {csv_path} …")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=FEATURES + ["policy_suggestion"])

    X = df[FEATURES].astype(float)
    Y_df, mlb = build_multilabel_targets(df)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_df, test_size=0.2, random_state=42
    )

    print("[train] Fitting Multi-output Random Forest …")
    base_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model = MultiOutputClassifier(base_rf, n_jobs=-1)
    model.fit(X_train, Y_train)

    # Evaluation
    Y_pred = model.predict(X_test)
    print("\n[train] Per-label classification report:")
    for i, col in enumerate(mlb.classes_):
        report = classification_report(
            Y_test.iloc[:, i], Y_pred[:, i],
            target_names=["absent", "present"],
            zero_division=0,
            output_dict=True,
        )
        f1 = report["present"]["f1-score"]
        print(f"  Policy {col:>2} | {POLICY_MAP[col][:55]:<55} | F1={f1:.2f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(MLB_PATH, "wb") as f:
        pickle.dump(mlb, f)

    print(f"\n[train] Model saved → {MODEL_PATH}")
    print(f"[train] Binarizer saved → {MLB_PATH}")


def fetch_waqi(lat: float, lng: float, token: str) -> Optional[float]:
    """Fetch current AQI from WAQI geo endpoint. Returns None on failure."""
    url = WAQI_GEO_BASE_URL.format(lat=lat, lng=lng, token=token)
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("status") == "ok":
            aqi_val = data["data"].get("aqi")
            if isinstance(aqi_val, (int, float)):
                return float(aqi_val)
    except Exception as e:
        print(f"  [WAQI] error for ({lat},{lng}): {e}")
    return None


def fetch_weather(lat: float, lng: float) -> Optional[dict]:
    """
    Fetch current weather from Open-Meteo.
    Returns dict with keys: T2M, PRECTOTCORR, WS10M, PS
    """
    params = {
        "latitude": lat,
        "longitude": lng,
        "current": [
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "surface_pressure",
        ],
        "timezone": "Asia/Kolkata",
    }
    try:
        resp = requests.get(FORECAST_BASE_URL, params=params, timeout=10)
        data = resp.json()
        current = data.get("current", {})
        return {
            "T2M":         current.get("temperature_2m", 25.0),
            "PRECTOTCORR": current.get("precipitation", 0.0),
            "WS10M":       current.get("wind_speed_10m", 2.0),
            "PS":          current.get("surface_pressure", 101.0),
        }
    except Exception as e:
        print(f"  [Open-Meteo] error for ({lat},{lng}): {e}")
    return None


def predict_all(waqi_token: str) -> None:
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"[predict] Model not found at {MODEL_PATH}. Run with --train first.")

    with open(MODEL_PATH, "rb") as f:
        model: MultiOutputClassifier = pickle.load(f)
    with open(MLB_PATH, "rb") as f:
        mlb: MultiLabelBinarizer = pickle.load(f)

    results = {}
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    for station_name, coords in STATION_COORDINATES.items():
        lat, lng = coords["lat"], coords["lng"]
        loc_id = STATION_TO_LOCATION_ID.get(station_name, 0)
        print(f"[predict] {station_name} ({lat}, {lng}) …")

        # Fetch live data
        weather = fetch_weather(lat, lng)
        aqi_val = fetch_waqi(lat, lng, waqi_token)

        # Fallbacks if APIs fail
        if weather is None:
            weather = {"T2M": 25.0, "PRECTOTCORR": 0.0, "WS10M": 2.0, "PS": 101.0}
            print(f"  [warn] Using fallback weather for {station_name}")
        if aqi_val is None:
            aqi_val = 200.0
            print(f"  [warn] Using fallback AQI for {station_name}")

        is_industrial = 1 if loc_id in INDUSTRIAL_IDS else 0
        has_sprinkler  = 1 if loc_id in SPRINKLER_IDS  else 0

        feature_row = pd.DataFrame([{
            "T2M":         weather["T2M"],
            "PRECTOTCORR": weather["PRECTOTCORR"],
            "WS10M":       weather["WS10M"],
            "PS":          weather["PS"],
            "AQI":         aqi_val,
            "Location":    loc_id,
            "hasSprinkler": has_sprinkler,
            "isIndustrial": is_industrial,
        }])

        # Predict multi-label binary vector
        pred_binary = model.predict(feature_row)[0]       # shape: (n_policies,)
        pred_proba  = model.predict_proba(feature_row)    # list of per-label probas

        # Confidence per label
        label_proba = {}
        for i, cls in enumerate(mlb.classes_):
            prob_arr = pred_proba[i][0]   # [prob_class0, prob_class1]
            label_proba[int(cls)] = round(float(prob_arr[1]), 3)

        # Collect activated policies, sorted by confidence descending
        activated = [
            int(cls)
            for cls, active in zip(mlb.classes_, pred_binary)
            if active == 1
        ]
        activated_sorted = sorted(activated, key=lambda c: label_proba[c], reverse=True)

        results[station_name] = {
            "lat":           lat,
            "lng":           lng,
            "location_id":   loc_id,
            "is_industrial": bool(is_industrial),
            "has_sprinkler": bool(has_sprinkler),
            "live_data": {
                "AQI":         aqi_val,
                "T2M":         weather["T2M"],
                "PRECTOTCORR": weather["PRECTOTCORR"],
                "WS10M":       weather["WS10M"],
                "PS":          weather["PS"],
            },
            "policies": [
                {
                    "id":          p,
                    "label":       POLICY_MAP[p],
                    "confidence":  label_proba[p],
                }
                for p in activated_sorted
            ],
            "all_confidences": {
                POLICY_MAP[k]: v for k, v in label_proba.items()
            },
        }

        time.sleep(0.3)   # polite rate-limiting

    # ─────────────────────────────────────────────
    # AGGREGATE CITY-WIDE POLICY SIGNAL
    # ─────────────────────────────────────────────
    policy_counter = Counter()
    policy_confidence_sum = Counter()

    for station, info in results.items():
        for p in info["policies"]:
            pid = p["id"]
            policy_counter[pid] += 1
            policy_confidence_sum[pid] += p["confidence"]

    total_stations = len(results)

    city_policies = []
    for pid, count in policy_counter.items():
        avg_conf = policy_confidence_sum[pid] / count

        city_policies.append({
            "id": pid,
            "label": POLICY_MAP[pid],
            "station_support": count,
            "support_ratio": round(count / total_stations, 2),
            "avg_confidence": round(avg_conf, 3),
        })

    # Sort by importance:
    # 1. most stations supporting
    # 2. then highest confidence
    city_policies = sorted(
        city_policies,
        key=lambda x: (x["station_support"], x["avg_confidence"]),
        reverse=True
    )

    majority_policy = city_policies[0] if city_policies else None

    # ─────────────────────────────────────────────
    # MINIMAL STATION OUTPUT (for UI)
    # ─────────────────────────────────────────────
    simplified_stations = {}

    for station, info in results.items():
        simplified_stations[station] = {
            "AQI": info["live_data"]["AQI"],
            "top_policy": info["policies"][0]["label"] if info["policies"] else "None"
        }


    output = {
        "generated_at": timestamp,
        "majority_policy": majority_policy,
        "city_policy_ranking": city_policies,
        "stations": simplified_stations
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[predict] Results written → {OUTPUT_JSON}")

    # summary maybe...
    print("\nCITY-WIDE POLICY RECOMMENDATION")
    if majority_policy:
        print(f"➡ {majority_policy['label']} "
              f"(supported by {majority_policy['station_support']}/{total_stations} stations)")


def _print_summary(results: dict) -> None:
    print("\n" + "=" * 72)
    print(f"{'Station':<35} {'AQI':>5}  Suggested Policies")
    print("=" * 72)
    for name, info in results.items():
        aqi = info["live_data"]["AQI"]
        policies = ", ".join(
            f"[{p['id']}]" for p in info["policies"]
        ) or "—"
        print(f"{name:<35} {aqi:>5.0f}  {policies}")
    print("=" * 72)
    print("Policy legend:")
    for pid, label in POLICY_MAP.items():
        print(f"  [{pid:>2}] {label}")



def main():
    parser = argparse.ArgumentParser(
        description="Delhi AQI Policy Recommendation Engine"
    )
    parser.add_argument("--train", action="store_true",
                        help="Train the model on the CSV dataset")
    parser.add_argument("--predict", action="store_true",
                        help="Fetch live data and generate predictions.json")
    parser.add_argument("--data", default="delhi_aqi_policy_dataset.csv",
                        help="Path to training CSV (default: delhi_aqi_policy_dataset.csv)")
    parser.add_argument("--waqi-token", default=os.environ.get("WAQI_TOKEN", ""),
                        help="WAQI API token (or set WAQI_TOKEN env var)")
    parser.add_argument("--output", default=OUTPUT_JSON,
                        help=f"Output JSON path (default: {OUTPUT_JSON})")

    args = parser.parse_args()

    if not args.train and not args.predict:
        parser.print_help()
        sys.exit(0)

    if args.output != OUTPUT_JSON:
        globals()["OUTPUT_JSON"] = args.output

    if args.train:
        train(args.data)

    if args.predict:
        if not args.waqi_token:
            sys.exit("[predict] --waqi-token is required (or set WAQI_TOKEN env var)")
        predict_all(args.waqi_token)


if __name__ == "__main__":
    main()
