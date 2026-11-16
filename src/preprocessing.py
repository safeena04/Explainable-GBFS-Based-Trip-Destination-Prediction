"""
preprocessing.py

Prepares trip records for destination prediction.
Calculates trip duration, geographic distance,
battery consumption and range consumption.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED_DIR
from data_loader import load_all_datasets


def calculate_geographic_distance(df):
    """
    Calculate straight-line distance between
    rental and return coordinates using Haversine formula.
    """

    lat1 = np.radians(df["lat_lend"])
    lon1 = np.radians(df["lng_lend"])

    lat2 = np.radians(df["lat_returned"])
    lon2 = np.radians(df["lng_returned"])

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    a = np.clip(a, 0, 1)

    earth_radius = 6371000

    distance = (
        2 * earth_radius
        * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    )

    return distance


def preprocess_dataset(df):

    df = df.copy()

    # Convert timestamps to datetime
    df["timestamp_lend"] = pd.to_datetime(
        df["timestamp_lend"], utc=True, errors="coerce"
    )

    df["timestamp_returned"] = pd.to_datetime(
        df["timestamp_returned"], utc=True, errors="coerce"
    )

    # Convert numeric columns
    numeric_columns = [
        "lat_lend",
        "lng_lend",
        "lat_returned",
        "lng_returned",
        "current_range_meters_lend",
        "current_range_meters_returned",
        "pedelec_battery_lend",
        "pedelec_battery_returned"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Calculate trip duration in seconds
    df["trip_duration_seconds"] = (
        df["timestamp_returned"]
        - df["timestamp_lend"]
    ).dt.total_seconds()

    # Calculate duration in minutes
    df["trip_duration_minutes"] = (
        df["trip_duration_seconds"] / 60
    )

    # Calculate geographic distance in metres
    df["straight_line_distance_m"] = (
        calculate_geographic_distance(df)
    )

    # Calculate battery consumption
    df["battery_diff"] = (
        df["pedelec_battery_lend"]
        - df["pedelec_battery_returned"]
    )

    # Calculate range consumption
    df["range_diff"] = (
        df["current_range_meters_lend"]
        - df["current_range_meters_returned"]
    )

    # Identify valid geographic coordinates
    df["valid_coordinates"] = (
        df["lat_lend"].between(-90, 90)
        & df["lat_returned"].between(-90, 90)
        & df["lng_lend"].between(-180, 180)
        & df["lng_returned"].between(-180, 180)
    )

    # Identify invalid trip durations
    df["valid_duration"] = (
        df["trip_duration_seconds"] > 0
    )

    # Identify negative battery consumption
    df["negative_battery_diff"] = (
        df["battery_diff"] < 0
    )

    return df


def generate_data_report(df):

    print("\n========== PREPROCESSING REPORT ==========")

    print("Total trips:", len(df))

    print(
        "Valid coordinates:",
        df["valid_coordinates"].sum()
    )

    print(
        "Invalid coordinates:",
        (~df["valid_coordinates"]).sum()
    )

    print(
        "Valid trip durations:",
        df["valid_duration"].sum()
    )

    print(
        "Non-positive trip durations:",
        (~df["valid_duration"]).sum()
    )

    print(
        "Negative battery differences:",
        df["negative_battery_diff"].sum()
    )

    print("\nTrip Duration (minutes):")

    print(
        df["trip_duration_minutes"].describe()
    )

    print("\nStraight-Line Distance (metres):")

    print(
        df["straight_line_distance_m"].describe()
    )

    print("\nBattery Difference:")

    print(
        df["battery_diff"].describe()
    )

    print("\nRange Difference:")

    print(
        df["range_diff"].describe()
    )

    print("\nTrips per day:")

    print(
        df["source_date"].value_counts().sort_index()
    )

    print("==========================================\n")


if __name__ == "__main__":

    # Load original dataset
    original_df = load_all_datasets()

    # Preprocess dataset
    processed_df = preprocess_dataset(original_df)

    # Generate report
    generate_data_report(processed_df)

    # Create processed data directory
    DATA_PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save processed dataset
    output_path = (
        DATA_PROCESSED_DIR / "processed_trips.csv"
    )

    processed_df.to_csv(
        output_path,
        index=False
    )

    print("Processed dataset saved successfully!")
    print("Location:", output_path)