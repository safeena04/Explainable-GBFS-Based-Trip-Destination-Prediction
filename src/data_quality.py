import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED_DIR


def investigate_anomalies():

    file_path = DATA_PROCESSED_DIR / "processed_trips.csv"

    df = pd.read_csv(file_path)

    # Identify unusual records
    long_trips = df[
        df["trip_duration_minutes"] > 180
    ]

    battery_increases = df[
        df["battery_diff"] < 0
    ]

    range_increases = df[
        df["range_diff"] < 0
    ]

    long_distances = df[
        df["straight_line_distance_m"] > 3000
    ]

    zero_distances = df[
        df["straight_line_distance_m"] == 0
    ]

    print("\n========== DATA QUALITY INVESTIGATION ==========")

    print("Trips longer than 3 hours:", len(long_trips))

    print("Battery increases:", len(battery_increases))

    print("Range increases:", len(range_increases))

    print("Distances above 3000 metres:", len(long_distances))

    print("Zero-distance trips:", len(zero_distances))

    print("\nLongest 10 trips:")

    print(
        long_trips[
            [
                "vehicle_id",
                "source_date",
                "trip_duration_minutes",
                "straight_line_distance_m",
                "battery_diff",
                "range_diff"
            ]
        ]
        .sort_values(
            "trip_duration_minutes",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nBattery increase examples:")

    print(
        battery_increases[
            [
                "vehicle_id",
                "pedelec_battery_lend",
                "pedelec_battery_returned",
                "battery_diff"
            ]
        ].head(10).to_string(index=False)
    )

    # Save anomaly records for inspection
    anomaly_records = df[
        (df["trip_duration_minutes"] > 180)
        | (df["battery_diff"] < 0)
        | (df["range_diff"] < 0)
        | (df["straight_line_distance_m"] > 3000)
    ]

    output_path = (
        DATA_PROCESSED_DIR / "anomaly_records.csv"
    )

    anomaly_records.to_csv(
        output_path,
        index=False
    )

    print("\nAnomaly records saved to:", output_path)

    print("================================================")


if __name__ == "__main__":

    investigate_anomalies()