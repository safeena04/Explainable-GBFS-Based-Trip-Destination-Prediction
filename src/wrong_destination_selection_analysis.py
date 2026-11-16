import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import METRICS_DIR, PREDICTIONS_DIR

# --------------------------------------------------
# 1. Load E6 validation predictions and rental details
# --------------------------------------------------

predictions = pd.read_csv(
    PREDICTIONS_DIR / "validation_e6_predictions.csv"
)

filter_details = pd.read_csv(
    METRICS_DIR / "candidate_filtering_rental_details.csv"
)

# Keep E6 filtering details only
filter_details = filter_details[
    filter_details["configuration"] == "E6"
].copy()

# Match rental identifier
filter_details = filter_details.rename(
    columns={"rental_id": "spoofed_id_rental"}
)

ID = "spoofed_id_rental"

# --------------------------------------------------
# 2. Merge candidate counts
# --------------------------------------------------

data = predictions.merge(
    filter_details[
        [
            ID,
            "remaining_candidate_count",
            "correct_candidate_retained"
        ]
    ],
    on=ID,
    how="left",
    validate="one_to_one"
)

assert data["remaining_candidate_count"].notna().all()

# --------------------------------------------------
# 3. Separate correct and incorrect selections
# --------------------------------------------------

correct = data[
    data["is_correct_destination"] == True
].copy()

wrong = data[
    data["is_correct_destination"] == False
].copy()

print("\n========== E6 SELECTION SUMMARY ==========")

print("Total predictions:", len(data))
print("Correct selections:", len(correct))
print("Wrong selections:", len(wrong))

print(
    "Vehicle matching accuracy:",
    round(len(correct) / len(data) * 100, 2),
    "%"
)

# --------------------------------------------------
# 4. Compare input feature characteristics
# --------------------------------------------------

features = [
    "remaining_candidate_count",
    "distance",
    "time_diff",
    "battery_diff",
    "range_diff",
    "speed",
    "speed_range_based"
]

print("\n========== FEATURE COMPARISON ==========")

comparison = pd.DataFrame({
    "Correct_selection_mean": correct[features].mean(),
    "Wrong_selection_mean": wrong[features].mean(),
    "Correct_selection_median": correct[features].median(),
    "Wrong_selection_median": wrong[features].median()
})

print(comparison.round(2).to_string())

# --------------------------------------------------
# 5. Analyze errors by candidate count
# --------------------------------------------------

print("\n========== ACCURACY BY CANDIDATE COUNT ==========")

data["candidate_count_group"] = pd.cut(
    data["remaining_candidate_count"],
    bins=[0, 1, 2, 3, 5, float("inf")],
    labels=["1", "2", "3", "4-5", "6+"]
)

count_analysis = data.groupby(
    "candidate_count_group",
    observed=False
).agg(
    total_predictions=("is_correct_destination", "size"),
    correct_predictions=("is_correct_destination", "sum")
)

count_analysis["accuracy_percent"] = (
    count_analysis["correct_predictions"]
    / count_analysis["total_predictions"]
    * 100
)

print(count_analysis.round(2).to_string())

# --------------------------------------------------
# 6. Analyze errors by time difference
# --------------------------------------------------

print("\n========== ACCURACY BY TIME DIFFERENCE ==========")

data["time_group"] = pd.cut(
    data["time_diff"],
    bins=[0, 300, 600, 900, 1200, float("inf")],
    labels=[
        "0-5 min",
        "5-10 min",
        "10-15 min",
        "15-20 min",
        "Above 20 min"
    ],
    include_lowest=True
)

time_analysis = data.groupby(
    "time_group",
    observed=False
).agg(
    total_predictions=("is_correct_destination", "size"),
    correct_predictions=("is_correct_destination", "sum")
)

time_analysis["accuracy_percent"] = (
    time_analysis["correct_predictions"]
    / time_analysis["total_predictions"]
    * 100
)

print(time_analysis.round(2).to_string())

# --------------------------------------------------
# 7. Save detailed results
# --------------------------------------------------

comparison.to_csv(
    METRICS_DIR / "wrong_selection_feature_comparison.csv"
)

count_analysis.to_csv(
    METRICS_DIR / "accuracy_by_candidate_count.csv"
)

time_analysis.to_csv(
    METRICS_DIR / "accuracy_by_time_difference.csv"
)

data.to_csv(
    METRICS_DIR / "e6_wrong_selection_details.csv",
    index=False
)

print("\n========== COMPLETED ==========")

print("Saved feature comparison.")
print("Saved candidate count analysis.")
print("Saved time difference analysis.")
print("Saved detailed E6 predictions.")