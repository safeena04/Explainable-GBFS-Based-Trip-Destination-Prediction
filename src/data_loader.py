"""
data_loader.py

Loads the seven daily GBFS trip datasets
and combines them into one DataFrame.
"""

import sys
from pathlib import Path

import pandas as pd

# Add the project root directory to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_RAW_DIR, RAW_COLUMNS


def load_all_datasets():

    csv_files = sorted(DATA_RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in data/raw"
        )

    dataframes = []

    for file in csv_files:

        print(f"Loading: {file.name}")

        df = pd.read_csv(file)

        df["source_date"] = file.stem

        dataframes.append(df)

    combined_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    return combined_df


def validate_dataset(df):

    missing_columns = [
        col for col in RAW_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print("\nDataset Validation")
    print("------------------")

    print("Total records:", len(df))

    print("Total columns:", len(df.columns))

    print("Missing values:", df.isnull().sum().sum())

    print("Unique vehicles:", df["vehicle_id"].nunique())

    return True


if __name__ == "__main__":

    df = load_all_datasets()

    validate_dataset(df)

    print("\nFirst five records:")

    print(df.head())