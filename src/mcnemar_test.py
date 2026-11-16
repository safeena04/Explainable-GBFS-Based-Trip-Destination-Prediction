import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# --------------------------------------------------
# 1. Load validation predictions
# --------------------------------------------------

PREDICTION_DIR = ROOT_DIR / "results" / "predictions"

baseline = pd.read_csv(
    PREDICTION_DIR / "validation_baseline_predictions.csv"
)

e6 = pd.read_csv(
    PREDICTION_DIR / "validation_e6_predictions.csv"
)

ID = "spoofed_id_rental"
CORRECT = "is_correct_destination"

# --------------------------------------------------
# 2. Keep required columns
# --------------------------------------------------

baseline = baseline[[ID, CORRECT]].copy()

e6 = e6[[ID, CORRECT]].copy()

# Check for duplicate rental IDs
if baseline[ID].duplicated().any():
    raise ValueError("Duplicate rental IDs found in Baseline.")

if e6[ID].duplicated().any():
    raise ValueError("Duplicate rental IDs found in E6.")

# Rename correctness columns
baseline = baseline.rename(
    columns={CORRECT: "baseline_correct"}
)

e6 = e6.rename(
    columns={CORRECT: "e6_correct"}
)

# --------------------------------------------------
# 3. Match common rentals
# --------------------------------------------------

paired = pd.merge(
    baseline,
    e6,
    on=ID,
    how="inner"
)

paired["baseline_correct"] = (
    paired["baseline_correct"].astype(bool)
)

paired["e6_correct"] = (
    paired["e6_correct"].astype(bool)
)

print("\n========== PAIRED VALIDATION DATA ==========")

print("Baseline predictions:", len(baseline))
print("E6 predictions:", len(e6))
print("Common rentals:", len(paired))

# --------------------------------------------------
# 4. Calculate paired outcomes
# --------------------------------------------------

both_correct = (
    paired["baseline_correct"]
    & paired["e6_correct"]
).sum()

both_wrong = (
    ~paired["baseline_correct"]
    & ~paired["e6_correct"]
).sum()

baseline_only_correct = (
    paired["baseline_correct"]
    & ~paired["e6_correct"]
).sum()

e6_only_correct = (
    ~paired["baseline_correct"]
    & paired["e6_correct"]
).sum()

print("\n========== PAIRED OUTCOMES ==========")

print("Both correct:", both_correct)
print("Both incorrect:", both_wrong)
print("Baseline correct, E6 incorrect:", baseline_only_correct)
print("Baseline incorrect, E6 correct:", e6_only_correct)

# --------------------------------------------------
# 5. Calculate accuracy on common rentals
# --------------------------------------------------

total = len(paired)

baseline_accuracy = (
    paired["baseline_correct"].mean() * 100
)

e6_accuracy = (
    paired["e6_correct"].mean() * 100
)

difference = e6_accuracy - baseline_accuracy

print("\n========== COMMON-SET ACCURACY ==========")

print("Baseline accuracy:", round(baseline_accuracy, 2), "%")
print("E6 accuracy:", round(e6_accuracy, 2), "%")
print("Accuracy improvement:", round(difference, 2), "percentage points")

# --------------------------------------------------
# 6. Exact McNemar test
# --------------------------------------------------

discordant_pairs = (
    baseline_only_correct + e6_only_correct
)

if discordant_pairs == 0:
    print("\nNo discordant pairs. McNemar test is not informative.")
else:

    test_result = binomtest(
        k=e6_only_correct,
        n=discordant_pairs,
        p=0.5,
        alternative="two-sided"
    )

    print("\n========== EXACT McNEMAR TEST ==========")

    print("Discordant pairs:", discordant_pairs)
    print("E6-only correct:", e6_only_correct)
    print("Baseline-only correct:", baseline_only_correct)

    print("Exact p-value:", test_result.pvalue)

    alpha = 0.05

    if test_result.pvalue < alpha:
        print("\nResult: Statistically significant difference.")
    else:
        print("\nResult: No statistically significant difference.")

    print("Significance level:", alpha)

# --------------------------------------------------
# 7. Save paired results
# --------------------------------------------------

output_path = (
    ROOT_DIR / "results" / "metrics" / "mcnemar_test_results.csv"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

summary = pd.DataFrame([{
    "common_rentals": total,
    "both_correct": both_correct,
    "both_wrong": both_wrong,
    "baseline_only_correct": baseline_only_correct,
    "e6_only_correct": e6_only_correct,
    "baseline_accuracy_percent": baseline_accuracy,
    "e6_accuracy_percent": e6_accuracy,
    "accuracy_difference_percentage_points": difference,
    "mcnemar_exact_p_value": (
        test_result.pvalue if discordant_pairs > 0 else None
    )
}])

summary.to_csv(output_path, index=False)

print("\nSaved:", output_path)
