from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PREDICTION_DIR = ROOT / "results" / "predictions"

FILES = {
    "Baseline": "test_baseline_predictions.csv",
    "E6": "test_e6_predictions.csv",
}

for name, filename in FILES.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    path = PREDICTION_DIR / filename
    df = pd.read_csv(path)

    print("Total prediction rows:", len(df))
    print("Columns:")
    print(df.columns.tolist())

    rental_id = "spoofed_id_rental"

    if rental_id in df.columns:
        print(
            "Unique rentals:",
            df[rental_id].nunique()
        )

        print(
            "Duplicate rental IDs:",
            df[rental_id].duplicated().sum()
        )

    if "is_correct_destination" in df.columns:
        print(
            "Correct destination count:",
            df["is_correct_destination"].sum()
        )

    if "prediction_error_m" in df.columns:
        for tolerance in [100, 250, 500, 1000]:
            count = (
                df["prediction_error_m"] <= tolerance
            ).sum()

            print(
                f"Correct within {tolerance}m:",
                count
            )

    print("\nFirst three rows:")
    print(df.head(3).to_string(index=False))

print("\n" + "=" * 60)
print("INSPECTION COMPLETED")
print("=" * 60)