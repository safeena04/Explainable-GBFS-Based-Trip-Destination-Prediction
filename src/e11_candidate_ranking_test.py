"""
Final Test Evaluation:
1. Baseline + Nearest
2. E11 + Nearest
3. E11 + W3

W3 weights were selected using validation data only.
Test data is used only for final evaluation.
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
    TEST_DATES,
    MAX_BATTERY_DIFF,
    MAX_SPEED_KMH,
)

BASELINE_RENTAL_FILE = METRICS_DIR / "baseline_geographic_coverage.csv"

ID = "spoofed_id_rental"

MAX_DISTANCE = 5000
MAX_TIME = 1800

TOLERANCES = [100, 250, 500, 1000]

# Frozen using validation results
W3 = {
    "distance": 0.60,
    "time": 0.30,
    "battery": 0.10
}


def load_data():

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    expanded = pd.read_csv(
        DATA_PROCESSED_DIR / "expanded_candidates.csv"
    )

    baseline = pd.read_csv(
        DATA_PROCESSED_DIR / "baseline_candidates.csv"
    )

    rentals = pd.read_csv(BASELINE_RENTAL_FILE)

    for df in [expanded, baseline, rentals]:
        df["date"] = df["date"].astype(str)

    expanded = expanded[
        expanded["date"].isin(TEST_DATES)
    ].copy()

    baseline = baseline[
        baseline["date"].isin(TEST_DATES)
    ].copy()

    rentals = rentals[
        rentals["date"].isin(TEST_DATES)
    ].copy()
    rentals = rentals.rename(
        columns={"rental_id": ID}
    )

    print("Test rental events:", len(rentals))
    print("Expanded candidate records:", len(expanded))
    print("Baseline candidate records:", len(baseline))

    return model, scaler, expanded, baseline, rentals


def apply_e11(candidates):

    return candidates[
        (candidates["distance"] <= MAX_DISTANCE)
        & (candidates["time_diff"] <= MAX_TIME)
        & (candidates["battery_diff"] <= MAX_BATTERY_DIFF)
        & (candidates["speed"] <= MAX_SPEED_KMH)
    ].copy()


def generate_predictions(data, model, scaler):

    data = data.copy()

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


def normalize_features(data):

    data = data.copy()

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


def select_candidates(data, strategy):

    data = data.copy()

    if strategy in ["Baseline + Nearest", "E11 + Nearest"]:

        data["ranking_score"] = data["prediction_distance_m"]

    elif strategy == "E11 + W3":

        data["ranking_score"] = (
            W3["distance"] * data["prediction_distance_m_normalized"]
            + W3["time"] * data["time_diff_normalized"]
            + W3["battery"] * data["battery_diff_normalized"]
        )

    selected_indices = data.groupby(ID)["ranking_score"].idxmin()

    return data.loc[selected_indices].copy()


def evaluate_strategy(selected, strategy, total_rentals):

    available = selected[ID].nunique()

    result = {
        "strategy": strategy,
        "total_rentals": total_rentals,
        "available_rentals": available,
        "availability_percent": available / total_rentals * 100
    }

    for tolerance in TOLERANCES:

        correct = int(
            (selected["true_destination_error_m"] <= tolerance).sum()
        )

        result[f"correct_{tolerance}m"] = correct

        result[f"end_to_end_accuracy_{tolerance}m_percent"] = (
            correct / total_rentals * 100
        )

        result[f"conditional_accuracy_{tolerance}m_percent"] = (
            correct / available * 100 if available else 0
        )

    return result


def main():

    model, scaler, expanded, baseline, rentals = load_data()

    total_rentals = rentals[ID].nunique()

    if total_rentals != len(rentals):
        raise ValueError(
            "Duplicate rental IDs found in the test rental records."
        )

    # Baseline candidates are already generated using the original method.
    baseline_data = generate_predictions(
        baseline, model, scaler
    )

    # Apply E11 filtering to expanded candidates.
    e11_candidates = apply_e11(expanded)

    e11_data = generate_predictions(
        e11_candidates, model, scaler
    )

    e11_data = normalize_features(e11_data)

    print("\nBaseline candidate pairs:", len(baseline_data))
    print("E11 candidate pairs:", len(e11_data))

    print(
        "Baseline available rentals:",
        baseline_data[ID].nunique()
    )

    print(
        "E11 available rentals:",
        e11_data[ID].nunique()
    )

    strategies = {
        "Baseline + Nearest": baseline_data,
        "E11 + Nearest": e11_data,
        "E11 + W3": e11_data
    }

    summaries = []
    selected_results = {}

    for strategy, data in strategies.items():

        selected = select_candidates(data, strategy)

        selected_results[strategy] = selected

        result = evaluate_strategy(
            selected,
            strategy,
            total_rentals
        )

        summaries.append(result)

    summary_df = pd.DataFrame(summaries)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = (
        METRICS_DIR / "final_test_ranking_results.csv"
    )

    details_path = (
        METRICS_DIR / "final_test_selected_candidates.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    details = []

    for strategy, selected in selected_results.items():

        temp = selected.copy()
        temp["strategy"] = strategy
        details.append(temp)

    pd.concat(details, ignore_index=True).to_csv(
        details_path,
        index=False
    )

    print("\n========== FINAL TEST RESULTS ==========")

    print(summary_df.round(3).to_string(index=False))

    print("\nSaved:", summary_path)
    print("Saved:", details_path)


if __name__ == "__main__":
    main()