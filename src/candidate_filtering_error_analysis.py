import sys
from pathlib import Path

import pandas as pd

# Add project root to Python import path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    DATA_PROCESSED_DIR,
    METRICS_DIR,
    VAL_DATES,
)

# --------------------------------------------------
# 1. Load baseline candidates and rental information
# --------------------------------------------------

candidate_path = DATA_PROCESSED_DIR / "baseline_candidates.csv"

coverage_path = METRICS_DIR / "baseline_rental_coverage.csv"

candidates = pd.read_csv(candidate_path)
rentals = pd.read_csv(coverage_path)

# Use validation dates only
candidates["date"] = candidates["date"].astype(str)
rentals["date"] = rentals["date"].astype(str)

candidates = candidates[
    candidates["date"].isin(VAL_DATES)
].copy()

rentals = rentals[
    rentals["date"].isin(VAL_DATES)
].copy()

# Identify correct destination candidates
candidates["is_correct"] = (
    candidates["vehicle_id_rental"]
    == candidates["vehicle_id_return"]
)

# Rental identifier
RENTAL_ID = "spoofed_id_rental"

total_rentals = rentals["rental_id"].nunique()

baseline_correct = set(
    candidates.loc[
        candidates["is_correct"], RENTAL_ID
    ].unique()
)

print("\n========== VALIDATION DATA ==========")
print("Total rentals:", total_rentals)
print("Baseline candidate pairs:", len(candidates))
print("Rentals with correct baseline candidate:",
      len(baseline_correct))


# --------------------------------------------------
# 2. Define filtering configurations
# --------------------------------------------------

CONFIGS = {
    "E1": (1000, 600),
    "E2": (1000, 1200),
    "E3": (2000, 600),
    "E4": (2000, 1200),
    "E5": (3000, 600),
    "E6": (3000, 1200),
    "Baseline": (3000, 1800),
}

summary_results = []
removal_results = []
rental_results = []


# --------------------------------------------------
# 3. Analyze each configuration
# --------------------------------------------------

for name, (max_distance, max_time) in CONFIGS.items():

    filtered = candidates[
        (candidates["distance"] <= max_distance)
        & (candidates["time_diff"] <= max_time)
    ].copy()

    # Rentals having at least one candidate
    available = set(filtered[RENTAL_ID].unique())

    # Rentals retaining at least one correct candidate
    correct_retained = set(
        filtered.loc[
            filtered["is_correct"], RENTAL_ID
        ].unique()
    )

    # Correct baseline rentals lost after filtering
    lost_correct = baseline_correct - correct_retained

    # Candidate removal categories
    distance_fail = candidates["distance"] > max_distance
    time_fail = candidates["time_diff"] > max_time

    distance_only = distance_fail & ~time_fail
    time_only = time_fail & ~distance_fail
    both = distance_fail & time_fail

    # Correct candidate removal categories
    correct_distance_only = candidates[
        distance_only & candidates["is_correct"]
    ]

    correct_time_only = candidates[
        time_only & candidates["is_correct"]
    ]

    correct_both = candidates[
        both & candidates["is_correct"]
    ]

    # Metrics
    candidate_count = len(filtered)

    coverage = (
        len(correct_retained) / total_rentals * 100
        if total_rentals else 0
    )

    availability = (
        len(available) / total_rentals * 100
        if total_rentals else 0
    )

    average_candidates = (
        candidate_count / total_rentals
        if total_rentals else 0
    )

    summary_results.append({
        "configuration": name,
        "max_distance_m": max_distance,
        "max_time_s": max_time,
        "candidate_pairs": candidate_count,
        "available_rentals": len(available),
        "correct_rentals_retained": len(correct_retained),
        "correct_rentals_lost": len(lost_correct),
        "candidate_coverage_percent": coverage,
        "availability_percent": availability,
        "average_candidates": average_candidates,
    })

    removal_results.append({
        "configuration": name,
        "distance_only_removed_pairs": int(distance_only.sum()),
        "time_only_removed_pairs": int(time_only.sum()),
        "both_conditions_failed_pairs": int(both.sum()),
        "correct_pairs_removed_by_distance_only":
            len(correct_distance_only),
        "correct_pairs_removed_by_time_only":
            len(correct_time_only),
        "correct_pairs_removed_by_both":
            len(correct_both),
    })

    # Per-rental analysis
    for rental_id in rentals["rental_id"].unique():

        original = candidates[
            candidates[RENTAL_ID] == rental_id
        ]

        remaining = filtered[
            filtered[RENTAL_ID] == rental_id
        ]

        had_correct = (
            original["is_correct"].any()
        )

        retained_correct = (
            remaining["is_correct"].any()
        )

        if not had_correct:
            status = "No correct candidate in baseline"
        elif retained_correct:
            status = "Correct candidate retained"
        else:
            status = "Correct candidate removed"

        rental_results.append({
            "configuration": name,
            "rental_id": rental_id,
            "original_candidate_count": len(original),
            "remaining_candidate_count": len(remaining),
            "had_correct_baseline_candidate": had_correct,
            "correct_candidate_retained": retained_correct,
            "status": status,
        })


# --------------------------------------------------
# 4. Save results
# --------------------------------------------------

summary_df = pd.DataFrame(summary_results)
removal_df = pd.DataFrame(removal_results)
rental_df = pd.DataFrame(rental_results)

summary_path = METRICS_DIR / "candidate_filtering_error_summary.csv"
removal_path = METRICS_DIR / "candidate_removal_reasons.csv"
rental_path = METRICS_DIR / "candidate_filtering_rental_details.csv"

summary_df.to_csv(summary_path, index=False)
removal_df.to_csv(removal_path, index=False)
rental_df.to_csv(rental_path, index=False)


# --------------------------------------------------
# 5. Display results
# --------------------------------------------------

print("\n========== CANDIDATE FILTERING SUMMARY ==========")
print(summary_df.round(2).to_string(index=False))

print("\n========== CANDIDATE REMOVAL REASONS ==========")
print(removal_df.to_string(index=False))

print("\n========== E6 CORRECT CANDIDATE ANALYSIS ==========")

e6 = rental_df[
    rental_df["configuration"] == "E6"
]

print(e6["status"].value_counts().to_string())

print("\n========== COMPLETED ==========")
print("Saved:", summary_path)
print("Saved:", removal_path)
print("Saved:", rental_path)