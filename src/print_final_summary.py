from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "results" / "metrics"

files = [
    "final_selected_configuration_corrected.csv",
    "all_configurations_corrected.csv",
    "validation_geographic_filtering_results.csv",
    "extended_validation_geographic_coverage.csv",
    "validation_geographic_accuracy.csv",
    "extended_validation_results.csv",
    "test_e9_corrected_metrics.csv",
]

for filename in files:
    path = METRICS / filename

    print("\n" + "=" * 75)
    print(filename)
    print("=" * 75)

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Display at most 10 rows
    print(df.head(10).to_string(index=False))


# Summarize paired statistical comparison
path = METRICS / "baseline_e9_paired_test_comparison.csv"

if path.exists():
    df = pd.read_csv(path)

    print("\n" + "=" * 75)
    print("PAIRED TEST COMPARISON SUMMARY")
    print("=" * 75)

    print("Total rentals:", len(df))
    print("Both correct:", (
        df["baseline_correct"] & df["e9_correct"]
    ).sum())

    print("Both incorrect:", (
        ~df["baseline_correct"] & ~df["e9_correct"]
    ).sum())

    print("Baseline only correct:", (
        df["baseline_correct"] & ~df["e9_correct"]
    ).sum())

    print("E9 only correct:", (
        ~df["baseline_correct"] & df["e9_correct"]
    ).sum())