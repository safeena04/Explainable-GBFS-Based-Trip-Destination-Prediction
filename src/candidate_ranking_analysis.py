import sys
from pathlib import Path

import pandas as pd
from geopy.distance import geodesic
import joblib

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
)

# --------------------------------------------------
# 1. Load model, scaler and candidate data
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

candidates = pd.read_csv(
    DATA_PROCESSED_DIR / "baseline_candidates.csv"
)

candidates["date"] = candidates["date"].astype(str)

candidates = candidates[
    candidates["date"].isin(VAL_DATES)
].copy()

ID = "spoofed_id_rental"

# Identify correct candidates using vehicle ID matching
candidates["is_correct"] = (
    candidates["vehicle_id_rental"]
    == candidates["vehicle_id_return"]
)

# --------------------------------------------------
# 2. Define configurations
# --------------------------------------------------

CONFIGS = {
    "Baseline": (3000, 1800),
    "E6": (3000, 1200),
}

all_results = []

# --------------------------------------------------
# 3. Predict and rank candidates
# --------------------------------------------------

for name, (max_distance, max_time) in CONFIGS.items():

    print(f"\nProcessing {name}...")

    filtered = candidates[
        (candidates["distance"] <= max_distance)
        & (candidates["time_diff"] <= max_time)
        & (candidates["battery_diff"] <= 10)
        & (candidates["speed"] <= 20)
    ].copy()

    if filtered.empty:
        continue

    # Match model feature names
    filtered = filtered.rename(columns={
        "lat_rental": "lat_lend",
        "lng_rental": "lng_lend",
        "speed": "mean_speed_distance_based",
        "speed_range_based": "mean_speed_range_based",
    })

    # Prepare model input
    X = filtered[MODEL_FEATURES].copy()

    # Scale the six numerical features
    X[SCALED_FEATURES] = scaler.transform(
        X[SCALED_FEATURES].astype(float)
    )

    # Predict destination coordinates
    predictions = model.predict(X)

    filtered["predicted_latitude"] = predictions[:, 0]
    filtered["predicted_longitude"] = predictions[:, 1]

    # Calculate distance between predicted coordinates
    # and each candidate return location
    filtered["prediction_distance_m"] = [
        geodesic(
            (pred_lat, pred_lng),
            (return_lat, return_lng)
        ).meters
        for pred_lat, pred_lng, return_lat, return_lng
        in zip(
            filtered["predicted_latitude"],
            filtered["predicted_longitude"],
            filtered["lat_return"],
            filtered["lng_return"]
        )
    ]

    # Rank candidates within each rental
    # method="min" gives tied candidates the same rank
    filtered["candidate_rank"] = (
        filtered.groupby(ID)["prediction_distance_m"]
        .rank(method="min", ascending=True)
    )

    filtered["configuration"] = name

    all_results.append(filtered)

# --------------------------------------------------
# 4. Combine results
# --------------------------------------------------

if not all_results:
    raise ValueError("No candidate records found for the configurations.")

results = pd.concat(all_results, ignore_index=True)

# --------------------------------------------------
# 5. Calculate correct candidate ranking per rental
# --------------------------------------------------

correct_candidates = results[
    results["is_correct"]
].copy()

# Keep the best rank of the correct candidate per rental.
# This prevents multiple correct candidate rows from
# counting the same rental more than once.

correct_rank = correct_candidates.groupby(
    ["configuration", ID]
)["candidate_rank"].min()

ranking_summary = correct_rank.groupby(
    level="configuration"
).agg(
    rentals_with_correct_candidate="count",
    average_correct_rank="mean",
    median_correct_rank="median",
)

# Top-1 and Top-3 rental counts
top1 = correct_rank.eq(1).groupby(
    level="configuration"
).sum()

top3 = correct_rank.le(3).groupby(
    level="configuration"
).sum()

rank2 = correct_rank.eq(2).groupby(
    level="configuration"
).sum()

rank3 = correct_rank.eq(3).groupby(
    level="configuration"
).sum()

ranking_summary["correct_rank_1"] = top1
ranking_summary["correct_rank_2"] = rank2
ranking_summary["correct_rank_3"] = rank3

ranking_summary["top1_accuracy_percent"] = (
    top1
    / ranking_summary["rentals_with_correct_candidate"]
    * 100
)

ranking_summary["top3_accuracy_percent"] = (
    top3
    / ranking_summary["rentals_with_correct_candidate"]
    * 100
)

ranking_summary = ranking_summary.reset_index()

# --------------------------------------------------
# 6. Display ranking summary
# --------------------------------------------------

print("\n========== CORRECT CANDIDATE RANKING ==========")

print(
    ranking_summary.round(2).to_string(index=False)
)

# --------------------------------------------------
# 7. Candidate count and candidate-level rank distribution
# --------------------------------------------------

results["candidate_count"] = (
    results.groupby(ID)[ID].transform("size")
)

print("\n========== CANDIDATE-LEVEL RANK DISTRIBUTION ==========")

print(
    correct_candidates.groupby(
        ["configuration", "candidate_rank"]
    ).size().to_string()
)

# --------------------------------------------------
# 8. Save results
# --------------------------------------------------

METRICS_DIR.mkdir(parents=True, exist_ok=True)

ranking_summary.to_csv(
    METRICS_DIR / "candidate_ranking_summary.csv",
    index=False
)

results.to_csv(
    METRICS_DIR / "candidate_level_ranking_details.csv",
    index=False
)

print("\n========== COMPLETED ==========")

print("Saved candidate_ranking_summary.csv")
print("Saved candidate_level_ranking_details.csv")
