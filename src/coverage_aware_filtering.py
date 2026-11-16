"""
Coverage-Aware Candidate Filtering Experiment

Purpose:
Evaluate candidate distance and time thresholds using
geographical candidate coverage and candidate availability.

Validation only. Test data is not used for selection.
"""

import sys
from pathlib import Path

import pandas as pd
from geopy.distance import geodesic

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_PROCESSED_DIR,
    METRICS_DIR,
    MAX_BATTERY_DIFF,
    MAX_SPEED_KMH,
    VAL_DATES,
)

# ---------------------------------------------------------
# CONFIGURATIONS
# ---------------------------------------------------------

EXPERIMENT_CONFIGS = [
    # Existing experiments
    {"experiment": "E1", "max_distance": 1000, "max_time": 600},
    {"experiment": "E2", "max_distance": 1000, "max_time": 1200},
    {"experiment": "E3", "max_distance": 2000, "max_time": 600},
    {"experiment": "E4", "max_distance": 2000, "max_time": 1200},
    {"experiment": "E5", "max_distance": 3000, "max_time": 600},
    {"experiment": "E6", "max_distance": 3000, "max_time": 1200},
    {"experiment": "Baseline", "max_distance": 3000, "max_time": 1800},

    # New experiments: extended distance and time
    {"experiment": "E10", "max_distance": 4000, "max_time": 1800},
    {"experiment": "E11", "max_distance": 5000, "max_time": 1800},
    {"experiment": "E12", "max_distance": 3000, "max_time": 2400},
    {"experiment": "E13", "max_distance": 3000, "max_time": 3000},
    {"experiment": "E14", "max_distance": 4000, "max_time": 2400},
    {"experiment": "E15", "max_distance": 5000, "max_time": 3000},
]

TOLERANCES = [100, 250, 500, 1000]


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_experiment_data():

    rental_records = pd.read_csv(
        METRICS_DIR / "baseline_geographic_coverage.csv"
    )

    candidates = pd.read_csv(
        DATA_PROCESSED_DIR / "expanded_candidates.csv"
    )

    print("\nData loaded successfully!")
    print("Total rental events:", len(rental_records))
    print("Total candidate pairs:", len(candidates))

    return rental_records, candidates


# ---------------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------------

def apply_candidate_filters(candidates, max_distance, max_time):

    return candidates[
        (candidates["distance"] <= max_distance)
        & (candidates["time_diff"] <= max_time)
        & (candidates["battery_diff"] <= MAX_BATTERY_DIFF)
        & (candidates["speed"] <= MAX_SPEED_KMH)
    ].copy()


# ---------------------------------------------------------
# CALCULATE GEOGRAPHIC DISTANCES
# ---------------------------------------------------------

def calculate_geographic_distances(candidates):

    candidates = candidates.copy()

    coordinate_columns = [
        "lat_return",
        "lng_return",
        "lat_returned_true",
        "lng_returned_true",
    ]

    for column in coordinate_columns:
        candidates[column] = pd.to_numeric(
            candidates[column],
            errors="coerce"
        )

    valid = candidates[coordinate_columns].notna().all(axis=1)

    candidates["candidate_distance_to_true_m"] = float("nan")

    for index in candidates.index[valid]:

        row = candidates.loc[index]

        candidate_location = (
            row["lat_return"],
            row["lng_return"]
        )

        true_destination = (
            row["lat_returned_true"],
            row["lng_returned_true"]
        )

        candidates.at[index, "candidate_distance_to_true_m"] = (
            geodesic(
                candidate_location,
                true_destination
            ).meters
        )

    return candidates


# ---------------------------------------------------------
# EVALUATE ONE CONFIGURATION
# ---------------------------------------------------------

def evaluate_experiment(
    rental_records,
    filtered_candidates,
    experiment
):

    total_rentals = len(rental_records)

    candidate_counts = (
        filtered_candidates
        .groupby("spoofed_id_rental")
        .size()
    )

    rentals_with_candidates = candidate_counts.index

    minimum_distances = (
        filtered_candidates
        .groupby("spoofed_id_rental")[
            "candidate_distance_to_true_m"
        ]
        .min()
    )

    result = {
        "experiment": experiment["experiment"],
        "max_distance": experiment["max_distance"],
        "max_time_minutes": experiment["max_time"] / 60,
        "total_rentals": total_rentals,
        "rentals_with_candidates": len(rentals_with_candidates),
        "candidate_availability_percent": (
            len(rentals_with_candidates) / total_rentals * 100
            if total_rentals else 0
        ),
        "average_candidates_per_rental": (
            len(filtered_candidates) / total_rentals
            if total_rentals else 0
        ),
    }

    for tolerance in TOLERANCES:

        covered_count = (
            minimum_distances.le(tolerance).sum()
        )

        coverage = (
            covered_count / total_rentals * 100
            if total_rentals else 0
        )

        result[f"covered_rentals_{tolerance}m"] = int(
            covered_count
        )

        result[f"candidate_coverage_{tolerance}m_percent"] = (
            round(coverage, 2)
        )

    # Old vehicle-ID proxy retained for comparison only.
    proxy_correct = filtered_candidates[
        filtered_candidates["vehicle_id_rental"]
        == filtered_candidates["vehicle_id_return"]
    ]["spoofed_id_rental"].nunique()

    result["vehicle_id_match_proxy_count"] = int(proxy_correct)

    result["vehicle_id_match_proxy_percent"] = round(
        proxy_correct / total_rentals * 100
        if total_rentals else 0,
        2
    )

    return result


# ---------------------------------------------------------
# RUN VALIDATION EXPERIMENTS
# ---------------------------------------------------------

def run_experiments():

    rental_records, candidates = load_experiment_data()

    validation_rentals = rental_records[
        rental_records["date"].astype(str).isin(VAL_DATES)
    ].copy()

    validation_candidates = candidates[
        candidates["date"].astype(str).isin(VAL_DATES)
    ].copy()

    print("\n========== VALIDATION DATA ==========")

    print("Validation rentals:", len(validation_rentals))
    print("Validation candidate pairs:", len(validation_candidates))

    # Calculate geographic distances once before filtering.
    validation_candidates = calculate_geographic_distances(
        validation_candidates
    )

    validation_results = []

    for experiment in EXPERIMENT_CONFIGS:

        print("\nEvaluating:", experiment["experiment"])

        filtered_candidates = apply_candidate_filters(
            validation_candidates,
            experiment["max_distance"],
            experiment["max_time"]
        )

        metrics = evaluate_experiment(
            validation_rentals,
            filtered_candidates,
            experiment
        )

        validation_results.append(metrics)

        print(
            "Availability: {:.2f}%".format(
                metrics["candidate_availability_percent"]
            )
        )

        print(
            "Coverage at 1000 m: {:.2f}%".format(
                metrics["candidate_coverage_1000m_percent"]
            )
        )

    validation_df = pd.DataFrame(validation_results)

    output_path = (
        METRICS_DIR / "validation_expanded_geographic_filtering_results.csv"
    )

    validation_df.to_csv(output_path, index=False)

    print("\n========== GEOGRAPHIC VALIDATION RESULTS ==========")

    print(validation_df.to_string(index=False))

    print("\nResults saved to:")
    print(output_path)


if __name__ == "__main__":
    run_experiments()