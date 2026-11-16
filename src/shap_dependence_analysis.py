import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
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
# 4. Prepare original feature values
# --------------------------------------------------

filtered = filtered.rename(columns={
    "lat_rental": "lat_lend",
    "lng_rental": "lng_lend",
    "speed": "mean_speed_distance_based",
    "speed_range_based": "mean_speed_range_based",
})

# Preserve original feature values for plotting
X_raw = filtered[MODEL_FEATURES].copy().astype(float)

# Create model input using the original scaler
X_model = X_raw.copy()

X_model[SCALED_FEATURES] = scaler.transform(
    X_model[SCALED_FEATURES]
)

# --------------------------------------------------
# 5. Select background and explanation samples
# --------------------------------------------------

background_size = min(50, len(X_model))
sample_size = min(100, len(X_model) - background_size)

background_indices = X_model.sample(
    n=background_size,
    random_state=42
).index

remaining_indices = X_model.index.difference(
    background_indices
)

sample_indices = X_model.loc[
    remaining_indices
].sample(
    n=sample_size,
    random_state=42
).index

X_background = X_model.loc[background_indices]
X_sample = X_model.loc[sample_indices]

# Original values corresponding to the explanation sample
X_plot = X_raw.loc[sample_indices]

print("\nBackground sample:", len(X_background))
print("Explanation sample:", len(X_sample))

# --------------------------------------------------
# 6. Define prediction function
# --------------------------------------------------

def predict_fn(data):

    data = pd.DataFrame(
        data,
        columns=MODEL_FEATURES
    )

    return model.predict(data)

# --------------------------------------------------
# 7. Create PermutationExplainer
# --------------------------------------------------

print("\nCreating SHAP PermutationExplainer...")

explainer = shap.PermutationExplainer(
    predict_fn,
    X_background
)

# --------------------------------------------------
# 8. Calculate SHAP values
# --------------------------------------------------

print("\nCalculating SHAP values...")

max_evals = (
    2 * len(MODEL_FEATURES) + 1
) * 5

shap_result = explainer(
    X_sample,
    max_evals=max_evals
)

shap_values = shap_result.values

print("SHAP output shape:", shap_values.shape)

if shap_values.ndim != 3 or shap_values.shape[2] != 2:
    raise ValueError(
        f"Unexpected SHAP output shape: {shap_values.shape}"
    )

# Separate coordinate outputs
latitude_shap = shap_values[:, :, 0]
longitude_shap = shap_values[:, :, 1]

# --------------------------------------------------
# 9. Select important features
# --------------------------------------------------

FEATURES_TO_PLOT = [
    "lat_lend",
    "lng_lend",
    "distance",
    "mean_speed_distance_based",
]

# Create output directories
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 10. Generate dependence plots
# --------------------------------------------------

dependence_records = []

output_data = {
    "Latitude": latitude_shap,
    "Longitude": longitude_shap,
}

for output_name, output_shap in output_data.items():

    print(f"\nGenerating {output_name} dependence plots...")

    for feature in FEATURES_TO_PLOT:

        feature_index = MODEL_FEATURES.index(feature)

        original_values = X_plot[feature].values

        shap_feature_values = output_shap[:, feature_index]

        # Store data for later analysis
        for original_value, shap_value in zip(
            original_values,
            shap_feature_values
        ):

            dependence_records.append({
                "output": output_name,
                "feature": feature,
                "original_feature_value": original_value,
                "shap_value": shap_value,
            })

        # Generate scatter plot
        plt.figure(figsize=(8, 5))

        plt.scatter(
            original_values,
            shap_feature_values,
            alpha=0.65
        )

        plt.axhline(
            y=0,
            linestyle="--"
        )

        plt.xlabel(feature)
        plt.ylabel("SHAP value")

        plt.title(
            f"SHAP Dependence: {feature} - {output_name}"
        )

        plt.grid(
            alpha=0.25
        )

        plt.tight_layout()

        filename = (
            f"shap_dependence_{feature}_{output_name.lower()}.png"
        )

        plt.savefig(
            FIGURES_DIR / filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("Saved:", filename)

# --------------------------------------------------
# 11. Save dependence data
# --------------------------------------------------

dependence_df = pd.DataFrame(
    dependence_records
)

dependence_df.to_csv(
    METRICS_DIR / "shap_dependence_data.csv",
    index=False
)

# --------------------------------------------------
# 12. Completion
# --------------------------------------------------

print("\n========== SHAP DEPENDENCE ANALYSIS COMPLETED ==========")

print("Generated 8 dependence plots.")

print("Saved data:")
print("results/metrics/shap_dependence_data.csv")