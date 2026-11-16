import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import joblib

from scipy.stats import spearmanr

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    DATA_PROCESSED_DIR,
    METRICS_DIR,
    MODEL_PATH,
    SCALER_PATH,
    MODEL_FEATURES,
    SCALED_FEATURES,
    VAL_DATES,
)

# --------------------------------------------------
# 1. Load model and scaler
# --------------------------------------------------

print("Loading model and scaler...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --------------------------------------------------
# 2. Load validation candidates
# --------------------------------------------------

candidates = pd.read_csv(
    DATA_PROCESSED_DIR / "baseline_candidates.csv"
)

candidates["date"] = candidates["date"].astype(str)

candidates = candidates[
    candidates["date"].isin(VAL_DATES)
].copy()

print("Validation candidate pairs:", len(candidates))

# --------------------------------------------------
# 3. Apply E6 filtering
# --------------------------------------------------

filtered = candidates[
    (candidates["distance"] <= 3000)
    & (candidates["time_diff"] <= 1200)
    & (candidates["battery_diff"] <= 10)
    & (candidates["speed"] <= 20)
].copy()

print("E6 candidate pairs:", len(filtered))

# --------------------------------------------------
# 4. Prepare model features
# --------------------------------------------------

filtered = filtered.rename(columns={
    "lat_rental": "lat_lend",
    "lng_rental": "lng_lend",
    "speed": "mean_speed_distance_based",
    "speed_range_based": "mean_speed_range_based",
})

X = filtered[MODEL_FEATURES].copy()

X[SCALED_FEATURES] = scaler.transform(
    X[SCALED_FEATURES].astype(float)
)

X = X.astype(float)

# --------------------------------------------------
# 5. Define model prediction function
# --------------------------------------------------

def predict_fn(data):

    data = pd.DataFrame(
        data,
        columns=MODEL_FEATURES
    )

    return model.predict(data)


# --------------------------------------------------
# 6. Define stability experiments
# --------------------------------------------------

SEEDS = [42, 123, 2026]

BACKGROUND_SIZE = 50
EXPLANATION_SIZE = 100
PERMUTATIONS = 5

all_importance = []

# --------------------------------------------------
# 7. Repeat SHAP analysis
# --------------------------------------------------

for run_number, seed in enumerate(SEEDS, start=1):

    print(f"\n========== SHAP RUN {run_number} ==========")
    print("Random seed:", seed)

    # Select reproducible background sample
    background = X.sample(
        n=BACKGROUND_SIZE,
        random_state=seed
    ).copy()

    # Select reproducible explanation sample
    remaining = X.drop(background.index)

    explanation = remaining.sample(
        n=EXPLANATION_SIZE,
        random_state=seed
    ).copy()

    # Create explainer
    explainer = shap.PermutationExplainer(
        predict_fn,
        background
    )

    # Calculate SHAP values
    max_evals = (
        2 * len(MODEL_FEATURES) + 1
    ) * PERMUTATIONS

    shap_result = explainer(
        explanation,
        max_evals=max_evals
    )

    values = shap_result.values

    if values.ndim != 3 or values.shape[2] != 2:
        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}"
        )

    # Separate latitude and longitude
    latitude_shap = values[:, :, 0]
    longitude_shap = values[:, :, 1]

    # Calculate mean absolute SHAP importance
    latitude_importance = np.abs(
        latitude_shap
    ).mean(axis=0)

    longitude_importance = np.abs(
        longitude_shap
    ).mean(axis=0)

    # Store results
    for i, feature in enumerate(MODEL_FEATURES):

        all_importance.append({
            "run": run_number,
            "seed": seed,
            "feature": feature,
            "latitude_importance": latitude_importance[i],
            "longitude_importance": longitude_importance[i],
        })

    print("SHAP run completed.")

# --------------------------------------------------
# 8. Combine all runs
# --------------------------------------------------

importance_df = pd.DataFrame(all_importance)

# --------------------------------------------------
# 9. Calculate feature rankings per run
# --------------------------------------------------

importance_df["latitude_rank"] = (
    importance_df.groupby("run")["latitude_importance"]
    .rank(ascending=False, method="average")
)

importance_df["longitude_rank"] = (
    importance_df.groupby("run")["longitude_importance"]
    .rank(ascending=False, method="average")
)

# --------------------------------------------------
# 10. Calculate mean and standard deviation
# --------------------------------------------------

summary = importance_df.groupby("feature").agg(
    latitude_mean=("latitude_importance", "mean"),
    latitude_std=("latitude_importance", "std"),
    longitude_mean=("longitude_importance", "mean"),
    longitude_std=("longitude_importance", "std"),
    latitude_average_rank=("latitude_rank", "mean"),
    longitude_average_rank=("longitude_rank", "mean"),
).reset_index()

summary["latitude_final_rank"] = (
    summary["latitude_mean"]
    .rank(ascending=False, method="min")
)

summary["longitude_final_rank"] = (
    summary["longitude_mean"]
    .rank(ascending=False, method="min")
)

# --------------------------------------------------
# 11. Calculate Spearman rank correlations
# --------------------------------------------------

latitude_rank_matrix = importance_df.pivot(
    index="feature",
    columns="run",
    values="latitude_rank"
)

longitude_rank_matrix = importance_df.pivot(
    index="feature",
    columns="run",
    values="longitude_rank"
)

correlation_results = []

for i in range(len(SEEDS)):

    for j in range(i + 1, len(SEEDS)):

        lat_corr, lat_p = spearmanr(
            latitude_rank_matrix[i + 1],
            latitude_rank_matrix[j + 1]
        )

        lon_corr, lon_p = spearmanr(
            longitude_rank_matrix[i + 1],
            longitude_rank_matrix[j + 1]
        )

        correlation_results.append({
            "run_1": i + 1,
            "run_2": j + 1,
            "latitude_spearman": lat_corr,
            "latitude_p_value": lat_p,
            "longitude_spearman": lon_corr,
            "longitude_p_value": lon_p,
        })

correlation_df = pd.DataFrame(correlation_results)

# --------------------------------------------------
# 12. Save results
# --------------------------------------------------

METRICS_DIR.mkdir(parents=True, exist_ok=True)

importance_df.to_csv(
    METRICS_DIR / "shap_stability_all_runs.csv",
    index=False
)

summary.to_csv(
    METRICS_DIR / "shap_stability_summary.csv",
    index=False
)

correlation_df.to_csv(
    METRICS_DIR / "shap_stability_correlations.csv",
    index=False
)

# --------------------------------------------------
# 13. Display results
# --------------------------------------------------

print("\n========== SHAP STABILITY SUMMARY ==========")

print(
    summary.round(6).to_string(index=False)
)

print("\n========== SPEARMAN RANK CORRELATIONS ==========")

print(
    correlation_df.round(4).to_string(index=False)
)

print("\n========== SHAP STABILITY ANALYSIS COMPLETED ==========")

print("Saved:")
print("shap_stability_all_runs.csv")
print("shap_stability_summary.csv")
print("shap_stability_correlations.csv")