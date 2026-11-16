from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "expanded_candidates.csv"
)

df = pd.read_csv(path)

print("\n========== EXPANDED DATASET ==========")

print("Total candidate pairs:", len(df))
print("Unique rental events:", df["spoofed_id_rental"].nunique())

print("\n========== DISTANCE ANALYSIS ==========")

print("Minimum distance:", df["distance"].min(), "m")
print("Maximum distance:", df["distance"].max(), "m")
print("Candidates above 3000 m:", (df["distance"] > 3000).sum())
print("Candidates above 4000 m:", (df["distance"] > 4000).sum())

print("\n========== TIME ANALYSIS ==========")

print("Minimum time:", df["time_diff"].min(), "seconds")
print("Maximum time:", df["time_diff"].max(), "seconds")
print("Candidates above 1800 sec:", (df["time_diff"] > 1800).sum())
print("Candidates above 2400 sec:", (df["time_diff"] > 2400).sum())

print("\n========== COMBINED THRESHOLD ANALYSIS ==========")

conditions = [
    ("4000 m / 30 min",
     (df["distance"] <= 4000) & (df["time_diff"] <= 1800)),

    ("5000 m / 30 min",
     (df["distance"] <= 5000) & (df["time_diff"] <= 1800)),

    ("3000 m / 40 min",
     (df["distance"] <= 3000) & (df["time_diff"] <= 2400)),

    ("3000 m / 50 min",
     (df["distance"] <= 3000) & (df["time_diff"] <= 3000)),

    ("4000 m / 40 min",
     (df["distance"] <= 4000) & (df["time_diff"] <= 2400)),

    ("5000 m / 50 min",
     (df["distance"] <= 5000) & (df["time_diff"] <= 3000)),
]

for name, condition in conditions:
    print(f"{name}: {condition.sum()} candidate pairs")

print("\n========== DATASET DATES ==========")
print(df["date"].value_counts().sort_index())