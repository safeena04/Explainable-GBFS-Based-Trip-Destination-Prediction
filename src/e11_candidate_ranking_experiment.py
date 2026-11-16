"""
E11 Candidate Ranking Experiment

Compare:
A. Nearest predicted-coordinate distance
B. Weighted distance + time + battery
C. Weighted distance + time

Weights are selected using validation data only.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from geopy.distance import geodesic

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    DATA_PROCESSED_DIR,
    METRICS_DIR,
    MODEL_PATH,
    SCALER_PATH,
    MODEL_FEATURES,
    SCALED_FEATURES,
    VAL_DATES,
    MAX_BATTERY_DIFF,
    MAX_SPEED_KMH,
    
)
BASELINE_RENTAL_FILE = METRICS_DIR / "baseline_geographic_coverage.csv"
# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MAX_DISTANCE = 5000
MAX_TIME = 1800
TOLERANCE_M = 1000

ID = "spoofed_id_rental"

# Candidate ranking weights to investigate.
# These are experimental values, not learned model weights.

WEIGHT_CONFIGS = [
    {"name": "W1", "distance": 0.80, "time": 0.15, "battery": 0.05},
    {"name": "W2", "distance": 0.70, "time": 0.20, "battery": 0.10},
    {"name": "W3", "distance": 0.60, "time": 0.30, "battery": 0.10},
    {"name": "W4", "distance": 0.80, "time": 0.20, "battery": 0.00},
    {"name": "W5", "distance": 0.90, "time": 0.10, "battery": 0.00},
]


# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------

def load_data():

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    candidates = pd.read_csv(
        DATA_PROCESSED_DIR / "expanded_candidates.csv"
    )

    rentals = pd.read_csv(BASELINE_RENTAL_FILE)

    candidates["date"] = candidates["date"].astype(str)
    rentals["date"] = rentals["date"].astype(str)

    candidates = candidates[
        candidates["date"].isin(VAL_DATES)
    ].copy()

    rentals = rentals[
        rentals["date"].isin(VAL_DATES)
    ].copy()

    print("Validation rental events:", len(rentals))
    print("Validation candidate records:", len(candidates))

    return model, scaler, candidates, rentals


# --------------------------------------------------
# APPLY E11 FILTERING
# --------------------------------------------------

def apply_e11(candidates):

    filtered = candidates[
        (candidates["distance"] <= MAX_DISTANCE)
        & (candidates["time_diff"] <= MAX_TIME)
        & (candidates["battery_diff"] <= MAX_BATTERY_DIFF)
        & (candidates["speed"] <= MAX_SPEED_KMH)
    ].copy()

    return filtered


# --------------------------------------------------
# GENERATE XGBOOST PREDICTIONS
# --------------------------------------------------

def generate_predictions(filtered, model, scaler):

    data = filtered.copy()

    data = data.rename(columns={
        "lat_rental": "lat_lend",
        "lng_rental": "lng_lend",
        "speed": "mean_speed_distance_based",
        "speed_range_based": "mean_speed_range_based"
    })

    X = data[MODEL_FEATURES].copy().astype(float)

    X[SCALED_FEATURES] = scaler.transform(
        X[SCALED_FEATURES]
    )

    predictions = model.predict(X)

    data["predicted_latitude"] = predictions[:, 0]
    data["predicted_longitude"] = predictions[:, 1]

    data["prediction_distance_m"] = [
        geodesic(
            (pred_lat, pred_lng),
            (return_lat, return_lng)
        ).meters
        for pred_lat, pred_lng, return_lat, return_lng
        in zip(
            data["predicted_latitude"],
            data["predicted_longitude"],
            data["lat_return"],
            data["lng_return"]
        )
    ]

    return data


# --------------------------------------------------
# CALCULATE TRUE DESTINATION ERROR
# --------------------------------------------------

def calculate_true_error(data):

    data = data.copy()

    data["true_destination_error_m"] = [
        geodesic(
            (return_lat, return_lng),
            (true_lat, true_lng)
        ).meters
        for return_lat, return_lng, true_lat, true_lng
        in zip(
            data["lat_return"],
            data["lng_return"],
            data["lat_returned_true"],
            data["lng_returned_true"]
        )
    ]

    return data


# --------------------------------------------------
# NORMALIZE RANKING FEATURES
# --------------------------------------------------

def normalize_features(data):

    data = data.copy()

    # Normalize within each rental candidate group.
    # This avoids comparing candidates from unrelated rentals.

    for feature in [
        "prediction_distance_m",
        "time_diff",
        "battery_diff"
    ]:

        minimum = data.groupby(ID)[feature].transform("min")
        maximum = data.groupby(ID)[feature].transform("max")

        difference = maximum - minimum

        data[f"{feature}_normalized"] = np.where(
            difference > 0,
            (data[feature] - minimum) / difference,
            0.0
        )

    return data


# --------------------------------------------------
# SELECT BEST CANDIDATE
# --------------------------------------------------

def select_candidates(data, strategy):

    data = data.copy()

    if strategy == "Nearest":

        data["ranking_score"] = data["prediction_distance_m"]

    else:

        weights = next(
            item for item in WEIGHT_CONFIGS
            if item["name"] == strategy
        )

        data["ranking_score"] = (
            weights["distance"]
            * data["prediction_distance_m_normalized"]
            + weights["time"]
            * data["time_diff_normalized"]
            + weights["battery"]
            * data["battery_diff_normalized"]
        )

    selected_indices = data.groupby(ID)["ranking_score"].idxmin()

    return data.loc[selected_indices].copy()


# --------------------------------------------------
# EVALUATE STRATEGY
# --------------------------------------------------

def evaluate_strategy(selected, strategy, total_rentals):

    correct = int(
        (selected["true_destination_error_m"] <= TOLERANCE_M).sum()
    )

    available = selected[ID].nunique()

    end_to_end_accuracy = correct / total_rentals * 100

    conditional_accuracy = (
        correct / available * 100 if available else 0
    )

    return {
        "strategy": strategy,
        "total_rentals": total_rentals,
        "available_rentals": available,
        "correct_predictions_1km": correct,
        "end_to_end_accuracy_1km_percent": end_to_end_accuracy,
        "conditional_accuracy_1km_percent": conditional_accuracy
    }


# --------------------------------------------------
# RUN EXPERIMENT
# --------------------------------------------------

def main():

    model, scaler, candidates, rentals = load_data()

    total_rentals = len(rentals)

    filtered = apply_e11(candidates)

    print("\nE11 candidate pairs:", len(filtered))
    print("Available rentals:", filtered[ID].nunique())

    predicted = generate_predictions(
        filtered,
        model,
        scaler
    )

    predicted = calculate_true_error(predicted)

    predicted = normalize_features(predicted)

    strategies = ["Nearest"] + [
        item["name"] for item in WEIGHT_CONFIGS
    ]

    summaries = []

    for strategy in strategies:

        selected = select_candidates(predicted, strategy)

        result = evaluate_strategy(
            selected,
            strategy,
            total_rentals
        )

        summaries.append(result)

        print(
            f"{strategy}: "
            f"{result['end_to_end_accuracy_1km_percent']:.2f}%"
        )

    summary_df = pd.DataFrame(summaries)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        METRICS_DIR / "e11_ranking_validation_results.csv"
    )

    summary_df.to_csv(output_path, index=False)

    predicted.to_csv(
        METRICS_DIR / "e11_ranking_candidate_details.csv",
        index=False
    )

    print("\n========== VALIDATION RANKING RESULTS ==========")

    print(summary_df.round(3).to_string(index=False))

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()