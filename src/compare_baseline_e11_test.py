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

    # 3. Load E11 predictions
    e11 = pd.read_csv(
        PREDICTIONS_DIR / "test_e11_predictions.csv"
    )

    e11["spoofed_id_rental"] = (
        e11["spoofed_id_rental"].astype(str)
    )

    e11 = calculate_geographic_error(e11)

    # 4. Check duplicate prediction IDs
    if baseline["spoofed_id_rental"].duplicated().any():
        raise ValueError("Duplicate Baseline rental predictions found.")

    if e11["spoofed_id_rental"].duplicated().any():
        raise ValueError("Duplicate E11 rental predictions found.")

    # 5. Calculate correctness independently
    baseline["baseline_correct"] = (
        baseline["geographic_error_m"] <= TOLERANCE
    )

    e11["e11_correct"] = (
        e11["geographic_error_m"] <= TOLERANCE
    )

    baseline_results = baseline[
        ["spoofed_id_rental", "geographic_error_m", "baseline_correct"]
    ].rename(
        columns={
            "spoofed_id_rental": "rental_id",
            "geographic_error_m": "baseline_error_m"
        }
    )

    e11_results = e11[
        ["spoofed_id_rental", "geographic_error_m", "e11_correct"]
    ].rename(
        columns={
            "spoofed_id_rental": "rental_id",
            "geographic_error_m": "e11_error_m"
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
        e11_results,
        on="rental_id",
        how="left",
        validate="one_to_one"
    )

    # Record prediction availability before filling missing values
    comparison["baseline_prediction_available"] = (
        comparison["baseline_correct"].notna()
    )

    comparison["e11_prediction_available"] = (
        comparison["e11_correct"].notna()
    )

    baseline_available = int(
        comparison["baseline_prediction_available"].sum()
    )

    e11_available = int(
        comparison["e11_prediction_available"].sum()
    )

    # Missing predictions count as incorrect
    comparison["baseline_correct"] = (
        comparison["baseline_correct"].fillna(False).astype(bool)
    )

    comparison["e11_correct"] = (
        comparison["e11_correct"].fillna(False).astype(bool)
    )

    # 7. End-to-end accuracy
    baseline_correct = int(comparison["baseline_correct"].sum())
    e11_correct = int(comparison["e11_correct"].sum())

    baseline_accuracy = baseline_correct / total_rentals * 100
    e11_accuracy = e11_correct / total_rentals * 100

    print("\n========== DATASET INFORMATION ==========")
    print("Total test rentals:", total_rentals)
    print("Baseline predictions matched:", baseline_available)
    print("E11 predictions matched:", e11_available)

    print("\n========== END-TO-END TEST ACCURACY ==========")

    print(
        f"Baseline: {baseline_correct}/{total_rentals} "
        f"({baseline_accuracy:.2f}%)"
    )

    print(
        f"E11: {e11_correct}/{total_rentals} "
        f"({e11_accuracy:.2f}%)"
    )

    print(
        f"Accuracy difference: "
        f"{e11_accuracy - baseline_accuracy:+.2f} percentage points"
    )

    # 8. Paired comparison
    both_correct = int(
        (
            comparison["baseline_correct"]
            & comparison["e11_correct"]
        ).sum()
    )

    both_incorrect = int(
        (
            ~comparison["baseline_correct"]
            & ~comparison["e11_correct"]
        ).sum()
    )

    baseline_only = int(
        (
            comparison["baseline_correct"]
            & ~comparison["e11_correct"]
        ).sum()
    )

    e11_only = int(
        (
            ~comparison["baseline_correct"]
            & comparison["e11_correct"]
        ).sum()
    )

    discordant = baseline_only + e11_only

    # 9. Exact McNemar test
    if discordant > 0:
        test_result = binomtest(
            e11_only,
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
    print("Baseline correct, E11 incorrect:", baseline_only)
    print("Baseline incorrect, E11 correct:", e11_only)
    print("Discordant pairs:", discordant)

    print(f"McNemar exact-test p-value: {p_value:.6f}")

    if p_value < 0.05:
        print("Result: Statistically significant difference.")
    else:
        print("Result: No statistically significant difference.")

    # 10. Save comparison
    output_path = (
        METRICS_DIR / "baseline_e11_paired_test_comparison.csv"
    )

    comparison.to_csv(output_path, index=False)

    print("\nPaired comparison saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
