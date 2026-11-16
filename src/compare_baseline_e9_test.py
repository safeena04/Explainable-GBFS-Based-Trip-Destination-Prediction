import sys
from pathlib import Path

import pandas as pd
from geopy.distance import geodesic
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import METRICS_DIR, PREDICTIONS_DIR, TEST_DATES

TOLERANCE = 1000


def calculate_geographic_error(df):
    df = df.copy()

    errors = []

    for _, row in df.iterrows():
        selected_location = (
            row["lat_return"],
            row["lng_return"]
        )

        actual_location = (
            row["lat_returned_true"],
            row["lng_returned_true"]
        )

        errors.append(
            geodesic(
                selected_location,
                actual_location
            ).meters
        )

    df["geographic_error_m"] = errors

    return df


def main():

    # 1. Load all test rentals
    rentals = pd.read_csv(
        METRICS_DIR / "baseline_rental_coverage.csv"
    )

    rentals = rentals[
        rentals["date"].astype(str).isin(TEST_DATES)
    ].copy()

    rentals["rental_id"] = rentals["rental_id"].astype(str)

    if rentals["rental_id"].duplicated().any():
        raise ValueError("Duplicate rental IDs in test rental dataset.")

    total_rentals = len(rentals)

    # 2. Load Baseline predictions
    baseline = pd.read_csv(
        PREDICTIONS_DIR / "test_baseline_predictions.csv"
    )

    baseline["spoofed_id_rental"] = (
        baseline["spoofed_id_rental"].astype(str)
    )

    baseline = calculate_geographic_error(baseline)

    # 3. Load E9 predictions
    e9 = pd.read_csv(
        PREDICTIONS_DIR / "test_e9_corrected_predictions.csv"
    )

    e9["spoofed_id_rental"] = (
        e9["spoofed_id_rental"].astype(str)
    )

    e9 = calculate_geographic_error(e9)

    # 4. Check duplicate prediction IDs
    if baseline["spoofed_id_rental"].duplicated().any():
        raise ValueError("Duplicate Baseline rental predictions found.")

    if e9["spoofed_id_rental"].duplicated().any():
        raise ValueError("Duplicate E9 rental predictions found.")

    # 5. Calculate correctness independently
    baseline["baseline_correct"] = (
        baseline["geographic_error_m"] <= TOLERANCE
    )

    e9["e9_correct"] = (
        e9["geographic_error_m"] <= TOLERANCE
    )

    baseline_results = baseline[
        ["spoofed_id_rental", "geographic_error_m", "baseline_correct"]
    ].rename(
        columns={
            "spoofed_id_rental": "rental_id",
            "geographic_error_m": "baseline_error_m"
        }
    )

    e9_results = e9[
        ["spoofed_id_rental", "geographic_error_m", "e9_correct"]
    ].rename(
        columns={
            "spoofed_id_rental": "rental_id",
            "geographic_error_m": "e9_error_m"
        }
    )

    # 6. Independently match both methods to all test rentals
    comparison = rentals[["rental_id"]].merge(
        baseline_results,
        on="rental_id",
        how="left",
        validate="one_to_one"
    )

    comparison = comparison.merge(
        e9_results,
        on="rental_id",
        how="left",
        validate="one_to_one"
    )

    # Record prediction availability before filling missing values
    comparison["baseline_prediction_available"] = (
        comparison["baseline_correct"].notna()
    )

    comparison["e9_prediction_available"] = (
        comparison["e9_correct"].notna()
    )

    baseline_available = int(
        comparison["baseline_prediction_available"].sum()
    )

    e9_available = int(
        comparison["e9_prediction_available"].sum()
    )

    # Missing predictions count as incorrect
    comparison["baseline_correct"] = (
        comparison["baseline_correct"].fillna(False).astype(bool)
    )

    comparison["e9_correct"] = (
        comparison["e9_correct"].fillna(False).astype(bool)
    )

    # 7. End-to-end accuracy
    baseline_correct = int(comparison["baseline_correct"].sum())
    e9_correct = int(comparison["e9_correct"].sum())

    baseline_accuracy = baseline_correct / total_rentals * 100
    e9_accuracy = e9_correct / total_rentals * 100

    print("\n========== DATASET INFORMATION ==========")
    print("Total test rentals:", total_rentals)
    print("Baseline predictions matched:", baseline_available)
    print("E9 predictions matched:", e9_available)

    print("\n========== END-TO-END TEST ACCURACY ==========")

    print(
        f"Baseline: {baseline_correct}/{total_rentals} "
        f"({baseline_accuracy:.2f}%)"
    )

    print(
        f"E9: {e9_correct}/{total_rentals} "
        f"({e9_accuracy:.2f}%)"
    )

    print(
        f"Accuracy difference: "
        f"{e9_accuracy - baseline_accuracy:+.2f} percentage points"
    )

    # 8. Paired comparison
    both_correct = int(
        (
            comparison["baseline_correct"]
            & comparison["e9_correct"]
        ).sum()
    )

    both_incorrect = int(
        (
            ~comparison["baseline_correct"]
            & ~comparison["e9_correct"]
        ).sum()
    )

    baseline_only = int(
        (
            comparison["baseline_correct"]
            & ~comparison["e9_correct"]
        ).sum()
    )

    e9_only = int(
        (
            ~comparison["baseline_correct"]
            & comparison["e9_correct"]
        ).sum()
    )

    discordant = baseline_only + e9_only

    # 9. Exact McNemar test
    if discordant > 0:
        test_result = binomtest(
            e9_only,
            n=discordant,
            p=0.5,
            alternative="two-sided"
        )

        p_value = test_result.pvalue

    else:
        p_value = 1.0

    print("\n========== PAIRED COMPARISON ==========")
    print("Both correct:", both_correct)
    print("Both incorrect:", both_incorrect)
    print("Baseline correct, E9 incorrect:", baseline_only)
    print("Baseline incorrect, E9 correct:", e9_only)
    print("Discordant pairs:", discordant)

    print(f"McNemar exact-test p-value: {p_value:.6f}")

    if p_value < 0.05:
        print("Result: Statistically significant difference.")
    else:
        print("Result: No statistically significant difference.")

    # 10. Save comparison
    output_path = (
        METRICS_DIR / "baseline_e9_paired_test_comparison.csv"
    )

    comparison.to_csv(output_path, index=False)

    print("\nCorrected paired comparison saved to:")
    print(output_path)


if __name__ == "__main__":
    main()