import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import METRICS_DIR, PREDICTIONS_DIR, VAL_DATES

# --------------------------------------------------
# 1. Load validation results
# --------------------------------------------------

coverage = pd.read_csv(
    METRICS_DIR / "baseline_rental_coverage.csv"
)

filter_details = pd.read_csv(
    METRICS_DIR / "candidate_filtering_rental_details.csv"
)

baseline = pd.read_csv(
    PREDICTIONS_DIR / "validation_baseline_predictions.csv"
)

e6 = pd.read_csv(
    PREDICTIONS_DIR / "validation_e6_predictions.csv"
)

# Use validation data only
coverage = coverage[
    coverage["date"].astype(str).isin(VAL_DATES)
].copy()

# Rename rental identifier
coverage = coverage.rename(
    columns={"rental_id": "spoofed_id_rental"}
)

ID = "spoofed_id_rental"

# Select E6 candidate filtering details
e6_filter = filter_details[
    filter_details["configuration"] == "E6"
].copy()

e6_filter = e6_filter.rename(
    columns={"rental_id": ID}
)

# --------------------------------------------------
# 2. Prepare selected predictions
# --------------------------------------------------

baseline_info = baseline[
    [ID, "is_correct_destination"]
].copy()

baseline_info = baseline_info.rename(
    columns={
        "is_correct_destination": "baseline_selected_correct"
    }
)

e6_info = e6[
    [ID, "is_correct_destination"]
].copy()

e6_info = e6_info.rename(
    columns={
        "is_correct_destination": "e6_selected_correct"
    }
)

# --------------------------------------------------
# 3. Merge results
# --------------------------------------------------

result = coverage.merge(
    e6_filter[
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

result = result.merge(
    baseline_info,
    on=ID,
    how="left",
    validate="one_to_one"
)

result = result.merge(
    e6_info,
    on=ID,
    how="left",
    validate="one_to_one"
)

# Check that all rental identifiers matched
assert len(result) == len(coverage)
assert result["remaining_candidate_count"].notna().all()

# --------------------------------------------------
# 4. Classify E6 errors
# --------------------------------------------------

def classify(row):

    if not bool(row["has_candidate"]):
        return "No baseline candidate"

    if not bool(row["correct_destination_retained"]):
        return "Correct candidate absent in baseline"

    if not bool(row["correct_candidate_retained"]):
        return "Correct candidate removed by E6"

    if pd.isna(row["e6_selected_correct"]):
        return "Unexpected: retained but no prediction"

    if bool(row["e6_selected_correct"]):
        return "Correct candidate selected by E6"

    return "Correct candidate retained but wrong selection"


result["error_category"] = result.apply(
    classify,
    axis=1
)

# --------------------------------------------------
# 5. Display error categories
# --------------------------------------------------

print("\n========== VALIDATION ERROR ANALYSIS ==========")

print("Total rentals:", len(result))

print("\nE6 error categories:")

print(
    result["error_category"]
    .value_counts()
    .to_string()
)

# --------------------------------------------------
# 6. Compare selected vehicle correctness
# --------------------------------------------------

print("\n========== VEHICLE MATCHING ACCURACY ==========")

for name, column in [
    ("Baseline", "baseline_selected_correct"),
    ("E6", "e6_selected_correct")
]:

    predictions = result[column].dropna()

    correct = int(predictions.astype(bool).sum())

    total = len(predictions)

    accuracy = correct / total * 100 if total else 0

    print(f"\n{name}")
    print("Predictions:", total)
    print("Correct vehicle selections:", correct)
    print("Incorrect vehicle selections:", total - correct)
    print(f"Vehicle matching accuracy: {accuracy:.2f}%")

# --------------------------------------------------
# 7. Compare common rentals
# --------------------------------------------------

common = result[
    result["baseline_selected_correct"].notna()
    & result["e6_selected_correct"].notna()
].copy()

b = common["baseline_selected_correct"].astype(bool)
e = common["e6_selected_correct"].astype(bool)

print("\n========== BASELINE VS E6 ==========")

print("Rentals predicted by both:", len(common))

print("Both correct:", int((b & e).sum()))

print("Both wrong:", int((~b & ~e).sum()))

print("Baseline correct, E6 wrong:", int((b & ~e).sum()))

print("Baseline wrong, E6 correct:", int((~b & e).sum()))

# --------------------------------------------------
# 8. Save results
# --------------------------------------------------

output = METRICS_DIR / "prediction_error_analysis.csv"

result.to_csv(output, index=False)

print("\nAnalysis completed.")
print("Saved:", output)