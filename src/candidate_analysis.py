import sys
from pathlib import Path

import pandas as pd
from geopy.distance import geodesic

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from candidate_generation import load_processed_data, create_events


# Geographic tolerances in metres
TOLERANCES = [100, 250, 500, 1000]


def analyze_candidates():

    # --------------------------------------------------
    # STEP 1: LOAD ORIGINAL PROCESSED DATA
    # --------------------------------------------------

    trips = load_processed_data()

    # Recreate eligible rental events
    all_rentals, all_returns = create_events(trips)

    rental_records = []

    for date, rentals in all_rentals.items():

        for rental in rentals:

            rental_records.append({
                "rental_id": rental["spoofed_id"],
                "vehicle_id": rental["vehicle_id"],
                "date": date
            })

    rentals_df = pd.DataFrame(rental_records)

    # --------------------------------------------------
    # STEP 2: LOAD GENERATED CANDIDATES
    # --------------------------------------------------

    candidates_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "baseline_candidates.csv"
    )

    candidates = pd.read_csv(candidates_path)

    print("\nLoaded candidate records:", len(candidates))

    # --------------------------------------------------
    # STEP 3: CALCULATE GEOGRAPHIC ERROR
    # --------------------------------------------------

    required_columns = [
        "lat_return",
        "lng_return",
        "lat_returned_true",
        "lng_returned_true"
    ]

    for column in required_columns:
        candidates[column] = pd.to_numeric(
            candidates[column],
            errors="coerce"
        )

    valid_coordinates = candidates[required_columns].notna().all(axis=1)

    candidates["candidate_distance_to_true_m"] = float("nan")

    for index in candidates.index[valid_coordinates]:

        candidate = candidates.loc[index]

        candidate_location = (
            candidate["lat_return"],
            candidate["lng_return"]
        )

        true_destination = (
            candidate["lat_returned_true"],
            candidate["lng_returned_true"]
        )

        candidates.at[index, "candidate_distance_to_true_m"] = (
            geodesic(
                candidate_location,
                true_destination
            ).meters
        )

    print(
        "Candidates with valid geographic coordinates:",
        valid_coordinates.sum()
    )

    # --------------------------------------------------
    # STEP 4: KEEP VEHICLE-ID MATCHING AS A DIAGNOSTIC
    # --------------------------------------------------

    candidates["vehicle_id_match_proxy"] = (
        candidates["vehicle_id_rental"]
        == candidates["vehicle_id_return"]
    )

    # --------------------------------------------------
    # STEP 5: COUNT CANDIDATES PER RENTAL
    # --------------------------------------------------

    candidate_counts = (
        candidates.groupby("spoofed_id_rental")
        .size()
        .rename("candidate_count")
    )

    proxy_correct_counts = (
        candidates[candidates["vehicle_id_match_proxy"]]
        .groupby("spoofed_id_rental")
        .size()
        .rename("vehicle_id_match_count")
    )

    # --------------------------------------------------
    # STEP 6: GEOGRAPHIC COVERAGE AT EACH TOLERANCE
    # --------------------------------------------------

    results = rentals_df.set_index("rental_id")

    results = results.join(candidate_counts)
    results = results.join(proxy_correct_counts)

    results["candidate_count"] = (
        results["candidate_count"]
        .fillna(0)
        .astype(int)
    )

    results["vehicle_id_match_count"] = (
        results["vehicle_id_match_count"]
        .fillna(0)
        .astype(int)
    )

    results["has_candidate"] = (
        results["candidate_count"] > 0
    )

    # Calculate the minimum candidate-to-destination
    # geographic distance for each rental.
    minimum_distances = (
        candidates.groupby("spoofed_id_rental")[
            "candidate_distance_to_true_m"
        ]
        .min()
        .rename("minimum_candidate_distance_m")
    )

    results = results.join(minimum_distances)

    # Calculate geographic coverage at each tolerance.
    for tolerance in TOLERANCES:

        column_name = f"covered_{tolerance}m"

        results[column_name] = (
            results["minimum_candidate_distance_m"]
            <= tolerance
        )

        results[column_name] = (
            results[column_name].fillna(False)
        )

    # Use 1,000 metres as the primary coverage tolerance.
    results["correct_destination_retained"] = (
        results["covered_1000m"]
    )

    # --------------------------------------------------
    # STEP 7: CALCULATE METRICS
    # --------------------------------------------------

    total_rentals = len(results)

    available_rentals = results["has_candidate"].sum()

    candidate_availability = (
        available_rentals / total_rentals * 100
        if total_rentals > 0 else 0
    )

    average_candidates = (
        results["candidate_count"].mean()
        if total_rentals > 0 else 0
    )

    print("\n========== BASELINE GEOGRAPHIC EVALUATION ==========")

    print("Total rental events:", total_rentals)

    print("Rentals with candidates:", available_rentals)

    print(
        "Rentals without candidates:",
        total_rentals - available_rentals
    )

    print(
        "Candidate availability: {:.2f}%".format(
            candidate_availability
        )
    )

    print(
        "Average candidates per rental: {:.2f}".format(
            average_candidates
        )
    )

    print("\nGeographic candidate coverage:")

    for tolerance in TOLERANCES:

        covered_count = results[
            f"covered_{tolerance}m"
        ].sum()

        coverage = (
            covered_count / total_rentals * 100
            if total_rentals > 0 else 0
        )

        print(
            f"{tolerance} m: "
            f"{covered_count}/{total_rentals} "
            f"({coverage:.2f}%)"
        )

    # Old vehicle-ID proxy for comparison only.
    proxy_count = (
        results["vehicle_id_match_count"] > 0
    ).sum()

    proxy_coverage = (
        proxy_count / total_rentals * 100
        if total_rentals > 0 else 0
    )

    print("\nVehicle-ID matching proxy (old method):")

    print(
        f"{proxy_count}/{total_rentals} "
        f"({proxy_coverage:.2f}%)"
    )

    print("\nCandidate count distribution:")

    print(results["candidate_count"].describe())

    # --------------------------------------------------
    # STEP 8: SAVE RESULTS
    # --------------------------------------------------

    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rental_output = (
        metrics_dir / "baseline_geographic_coverage.csv"
    )

    candidate_output = (
        metrics_dir / "baseline_candidate_geographic_distances.csv"
    )

    results.to_csv(rental_output)

    candidates.to_csv(candidate_output, index=False)

    print("\nRental-level geographic results saved to:")
    print(rental_output)

    print("\nCandidate-level geographic results saved to:")
    print(candidate_output)

    print("\n====================================================")


if __name__ == "__main__":
    analyze_candidates()