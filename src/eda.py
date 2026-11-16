import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED_DIR, FIGURES_DIR


def load_processed_data():

    file_path = DATA_PROCESSED_DIR / "processed_trips.csv"

    if not file_path.exists():
        raise FileNotFoundError("Processed dataset not found")

    return pd.read_csv(file_path)


def plot_trip_duration(df):

    # Limit the plot to 180 minutes for visibility.
    # The original data remains unchanged.
    plot_data = df[
        df["trip_duration_minutes"].between(0, 180)
    ]

    plt.figure(figsize=(10, 5))

    sns.histplot(
        plot_data["trip_duration_minutes"],
        bins=40,
        kde=True
    )

    plt.title("Distribution of Trip Duration")
    plt.xlabel("Trip Duration (minutes)")
    plt.ylabel("Number of Trips")

    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "trip_duration.png")

    plt.show()


def plot_trip_distance(df):

    plot_data = df[
        df["straight_line_distance_m"] <= 5000
    ]

    plt.figure(figsize=(10, 5))

    sns.histplot(
        plot_data["straight_line_distance_m"],
        bins=40,
        kde=True
    )

    plt.title("Distribution of Geographic Trip Distance")
    plt.xlabel("Straight-Line Distance (metres)")
    plt.ylabel("Number of Trips")

    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "trip_distance.png")

    plt.show()


def plot_daily_trips(df):

    daily_counts = (
        df.groupby("source_date")
        .size()
        .reset_index(name="trip_count")
    )

    plt.figure(figsize=(10, 5))

    sns.barplot(
        data=daily_counts,
        x="source_date",
        y="trip_count"
    )

    plt.title("Number of Trips Per Day")
    plt.xlabel("Date")
    plt.ylabel("Number of Trips")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "daily_trips.png")

    plt.show()


def plot_battery_difference(df):

    plt.figure(figsize=(10, 5))

    sns.histplot(
        df["battery_diff"],
        bins=40,
        kde=True
    )

    plt.title("Battery Difference Distribution")
    plt.xlabel("Rental Battery - Return Battery")
    plt.ylabel("Number of Trips")

    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "battery_difference.png")

    plt.show()


if __name__ == "__main__":

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_processed_data()

    print("Generating exploratory data analysis plots...")

    plot_trip_duration(df)

    plot_trip_distance(df)

    plot_daily_trips(df)

    plot_battery_difference(df)

    print("\nEDA completed successfully!")

    print("Figures saved in:", FIGURES_DIR)