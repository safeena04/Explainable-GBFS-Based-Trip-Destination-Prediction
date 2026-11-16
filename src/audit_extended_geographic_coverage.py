import sys
from pathlib import Path

import pandas as pd
from geopy.distance import geodesic

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_PROCESSED_DIR,
    METRICS_DIR,
    VAL_DATES,
    MAX_BATTERY_DIFF,
    MAX_SPEED_KMH
)

# E7, E8 and E9 configurations
EXPERIMENT_CONFIGS = {
    "E7": {"max_distance": 3000, "max_time": 1350},
    "E8": {"max_distance": 3000, "max_time": 1500},
    "E9": {"max_distance": 3000, "max_time": 1650},
}

TOLERANCES = [100, 250, 500, 1000]


def calculate_distances(data):

    data = data.copy()

    columns = [
        "lat_return",
        "lng_return",
        "lat_returned_true",
        "lng_returned_true"
    ]

    valid = data[columns].notna().all(axis=1)

    data["candidate_distance_to_true_m"] = float("nan")

    for index in data.index[valid]:

        row = data.loc[index]

        candidate_location = (
            row["lat_return"],
            row["lng_return"]
        )

        true_location = (
            row["lat_returned_true"],
            row["lng_returned_true"]
        )

        data.at[index, "candidate_distance_to_true_m"] = (
            geodesic(
                candidate_location,
                true_location
            ).meters
        )

    return data


def run_audit():

    # Load all eligible rental records
    rentals = pd.read_csv(
        METRICS_DIR / "baseline_geographic_coverage.csv"
    )

    # Load original baseline candidates
    candidates = pd.read_csv(
        DATA_PROCESSED_DIR / "baseline_candidates.csv"
    )

    # Validation split only
    rentals = rentals[
        rentals["date"].astype(str).isin(VAL_DATES)
    ].copy()

    candidates = candidates[
        candidates["date"].astype(str).isin(VAL_DATES)
    ].copy()

    total_rentals = len(rentals)

    print("Validation rentals:", total_rentals)
    print("Validation baseline candidates:", len(candidates))

    # Calculate geographic distances once
    candidates = calculate_distances(candidates)

    all_results = []

    for name, config in EXPERIMENT_CONFIGS.items():

        print("\nEvaluating:", name)

        filtered = candidates[
            (candidates["distance"] <= config["max_distance"])
            & (candidates["time_diff"] <= config["max_time"])
            & (candidates["battery_diff"] <= MAX_BATTERY_DIFF)
            & (candidates["speed"] <= MAX_SPEED_KMH)
        ].copy()

        candidate_counts = (
            filtered.groupby("spoofed_id_rental").size()
        )

        available = len(candidate_counts)

        average_candidates = (
            len(filtered) / total_rentals
        )

        minimum_distances = (
            filtered.groupby("spoofed_id_rental")[
                "candidate_distance_to_true_m"
            ].min()
        )

        result = {
            "experiment": name,
            "max_distance": config["max_distance"],
            "max_time_minutes": config["max_time"] / 60,
            "total_rentals": total_rentals,
            "candidate_pairs": len(filtered),
            "available_rentals": available,
            "availability_percent": round(
                available / total_rentals * 100, 2
            ),
            "average_candidates": round(
                average_candidates, 3
            )
        }

        for tolerance in TOLERANCES:

            covered = int(
                minimum_distances.le(tolerance).sum()
            )

            coverage = (
                covered / total_rentals * 100
            )

            result[f"covered_{tolerance}m"] = covered

            result[f"coverage_{tolerance}m_percent"] = round(
                coverage, 2
            )

        all_results.append(result)

        print("Candidate pairs:", len(filtered))
        print("Availability:", result["availability_percent"])
        print(
            "Geographic coverage at 1000m:",
            result["coverage_1000m_percent"]
        )
        print("Average candidates:", result["average_candidates"])

    results_df = pd.DataFrame(all_results)

    output_path = (
        METRICS_DIR /
        "extended_validation_geographic_coverage.csv"
    )

    results_df.to_csv(output_path, index=False)

    print("\n========== E7-E9 GEOGRAPHIC COVERAGE ==========")

    print(results_df.to_string(index=False))

    print("\nSaved to:", output_path)


if __name__ == "__main__":
    run_audit()
    