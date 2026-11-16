from pathlib import Path
import pandas as pd
from geopy.distance import geodesic

ROOT = Path(__file__).resolve().parent.parent
PREDICTION_DIR = ROOT / "results" / "predictions"
METRICS_DIR = ROOT / "results" / "metrics"

FILES = {
    "Baseline": "test_baseline_predictions.csv",
    "E6": "test_e6_predictions.csv",
}

TOLERANCES = [100, 250, 500, 1000]

# Total number of test rentals, including rentals
# for which no candidate was available.
TOTAL_TEST_RENTALS = 2282

all_results = []

for configuration, filename in FILES.items():

    print("\n" + "=" * 65)
    print(f"GEOGRAPHIC ACCURACY: {configuration}")
    print("=" * 65)

    df = pd.read_csv(PREDICTION_DIR / filename)

    # Error between the model's predicted coordinates
    # and the actual destination coordinates.
    df["predicted_coordinate_error_m"] = [
        geodesic(
            (pred_lat, pred_lng),
            (true_lat, true_lng)
        ).meters
        for pred_lat, pred_lng, true_lat, true_lng
        in zip(
            df["predicted_latitude"],
            df["predicted_longitude"],
            df["lat_returned_true"],
            df["lng_returned_true"]
        )
    ]

    # Error between the selected candidate's return location
    # and the actual destination coordinates.
    df["selected_candidate_error_m"] = [
        geodesic(
            (selected_lat, selected_lng),
            (true_lat, true_lng)
        ).meters
        for selected_lat, selected_lng, true_lat, true_lng
        in zip(
            df["lat_return"],
            df["lng_return"],
            df["lat_returned_true"],
            df["lng_returned_true"]
        )
    ]

    print("Prediction rows:", len(df))
    print("Unique rentals:", df["spoofed_id_rental"].nunique())
    print("Vehicle matching correct:", df["is_correct"].sum())

    print("\n--- Geographic accuracy ---")

    for tolerance in TOLERANCES:

        predicted_correct = (
            df["predicted_coordinate_error_m"] <= tolerance
        ).sum()

        selected_correct = (
            df["selected_candidate_error_m"] <= tolerance
        ).sum()

        # End-to-end accuracy includes rentals without predictions.
        predicted_end_to_end = (
            predicted_correct / TOTAL_TEST_RENTALS * 100
        )

        selected_end_to_end = (
            selected_correct / TOTAL_TEST_RENTALS * 100
        )

        print(f"\nTolerance: {tolerance} metres")

        print(
            "Model coordinate error:",
            predicted_correct,
            "/",
            len(df),
            f"({predicted_correct / len(df) * 100:.2f}%)"
        )

        print(
            "Selected candidate error:",
            selected_correct,
            "/",
            len(df),
            f"({selected_correct / len(df) * 100:.2f}%)"
        )

        print(
            "Model coordinate end-to-end accuracy:",
            f"{predicted_end_to_end:.2f}%"
        )

        print(
            "Selected candidate end-to-end accuracy:",
            f"{selected_end_to_end:.2f}%"
        )

        all_results.append({
            "configuration": configuration,
            "tolerance_m": tolerance,
            "prediction_rows": len(df),
            "model_coordinate_correct": predicted_correct,
            "model_coordinate_conditional_accuracy_percent":
                predicted_correct / len(df) * 100,
            "model_coordinate_end_to_end_accuracy_percent":
                predicted_end_to_end,
            "selected_candidate_correct": selected_correct,
            "selected_candidate_conditional_accuracy_percent":
                selected_correct / len(df) * 100,
            "selected_candidate_end_to_end_accuracy_percent":
                selected_end_to_end,
        })

    print("\n--- Error distribution ---")

    print(
        df[
            [
                "predicted_coordinate_error_m",
                "selected_candidate_error_m"
            ]
        ].describe().round(2)
    )

    # Save detailed rows for further inspection.
    output_file = (
        PREDICTION_DIR /
        f"test_{configuration.lower()}_geographic_errors.csv"
    )

    df.to_csv(output_file, index=False)

    print("\nSaved:", output_file)


# Save summary
summary = pd.DataFrame(all_results)

summary_path = (
    METRICS_DIR / "geographic_accuracy_audit.csv"
)

summary.to_csv(summary_path, index=False)

print("\n" + "=" * 65)
print("GEOGRAPHIC ACCURACY AUDIT COMPLETED")
print("=" * 65)

print(summary.round(3).to_string(index=False))
print("\nSaved:", summary_path)