import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MODEL_PATH,
    SCALER_PATH,
    MODEL_FEATURES,
    SCALED_FEATURES,
    METRICS_DIR,
    PREDICTIONS_DIR,
    GEO_TOLERANCES_M,
)

from geopy.distance import geodesic


def load_model_and_scaler():
    """Load the saved XGBoost model and scaler."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


def prepare_features(candidates, scaler):
    """Prepare and scale candidate features."""

    candidates = candidates.rename(columns={
        "lat_rental": "lat_lend",
        "lng_rental": "lng_lend",
        "speed": "mean_speed_distance_based",
        "speed_range_based": "mean_speed_range_based"
    })

    features = candidates[MODEL_FEATURES].copy()
    features = features.astype(float)

    features.loc[:, SCALED_FEATURES] = scaler.transform(
        features[SCALED_FEATURES]
    )

    return features


def predict_destinations(candidates, model, scaler):
    """Predict destination coordinates for candidate records."""

    features = prepare_features(candidates, scaler)

    predictions = model.predict(features)

    results = candidates.copy()

    results["predicted_latitude"] = predictions[:, 0]
    results["predicted_longitude"] = predictions[:, 1]

    return results


def calculate_prediction_distances(results):
    """Calculate distance between predicted and candidate return coordinates."""

    distances = []

    for _, row in results.iterrows():

        predicted_location = (
            row["predicted_latitude"],
            row["predicted_longitude"]
        )

        candidate_location = (
            row["lat_return"],
            row["lng_return"]
        )

        distance = geodesic(
            predicted_location,
            candidate_location
        ).meters

        distances.append(distance)

    results["prediction_distance_m"] = distances

    return results


def select_best_destination(results):
    """Select the candidate closest to predicted coordinates."""

    selected = (
        results.sort_values("prediction_distance_m")
        .groupby("spoofed_id_rental", as_index=False)
        .first()
    )

    return selected


if __name__ == "__main__":

    print("Loading candidate dataset...")

    candidate_path = (
        PROJECT_ROOT / "data" / "processed" / "baseline_candidates.csv"
    )

    candidates = pd.read_csv(candidate_path)

    print("Candidate records:", len(candidates))

    print("\nLoading model and scaler...")

    model, scaler = load_model_and_scaler()

    print("\nPredicting destination coordinates...")

    predictions = predict_destinations(
        candidates,
        model,
        scaler
    )

    print("Predictions generated:", len(predictions))

    print("\nCalculating geographical distances...")

    predictions = calculate_prediction_distances(predictions)

    print("\nSelecting best destination...")

    selected = select_best_destination(predictions)

    print("\n========== PREDICTION SUMMARY ==========")

    print("Total candidate records:", len(candidates))
    print("Selected destinations:", len(selected))

    print("\nSample selected destinations:")

    print(
        selected[
            [
                "spoofed_id_rental",
                "predicted_latitude",
                "predicted_longitude",
                "lat_return",
                "lng_return",
                "prediction_distance_m"
            ]
        ].head()
    )

    output_path = PREDICTIONS_DIR / "baseline_predictions.csv"

    selected.to_csv(output_path, index=False)

    print("\nPredictions saved to:", output_path) 