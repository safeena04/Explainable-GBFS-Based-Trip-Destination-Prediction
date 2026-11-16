import sys
from pathlib import Path

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

print("Loading XGBoost model and scaler...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --------------------------------------------------
# 2. Load validation candidate data
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

# Apply the original scaler to the same six features
X[SCALED_FEATURES] = scaler.transform(
    X[SCALED_FEATURES].astype(float)
)

X = X.astype(float)

print("\nModel input shape:", X.shape)

# --------------------------------------------------
# 5. Select background and explanation samples
# --------------------------------------------------

background_size = min(50, len(X))
sample_size = min(100, len(X))

X_background = X.sample(
    n=background_size,
    random_state=42
).copy()

X_sample = X.drop(X_background.index).sample(
    n=min(sample_size, len(X) - len(X_background)),
    random_state=42
).copy()

print("Background sample size:", len(X_background))
print("Explanation sample size:", len(X_sample))

# --------------------------------------------------
# 6. Define prediction function
# --------------------------------------------------

def predict_fn(data):

    # SHAP may pass NumPy arrays to the prediction function.
    # Convert them back to DataFrames with the original
    # model feature names.

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

number_of_features = len(MODEL_FEATURES)

# Five permutations for a more stable estimate.
max_evals = (2 * number_of_features + 1) * 5

shap_result = explainer(
    X_sample,
    max_evals=max_evals
)

shap_values = shap_result.values

print("\nSHAP calculation completed.")

print("SHAP output shape:", shap_values.shape)

# --------------------------------------------------
# 9. Separate latitude and longitude SHAP values
# --------------------------------------------------

if len(shap_values.shape) != 3:
    raise ValueError(
        f"Unexpected SHAP output shape: {shap_values.shape}"
    )

if shap_values.shape[1] != len(MODEL_FEATURES):
    raise ValueError("SHAP feature dimension does not match.")

if shap_values.shape[2] != 2:
    raise ValueError("Expected two model outputs: latitude and longitude.")

latitude_shap = shap_values[:, :, 0]
longitude_shap = shap_values[:, :, 1]

# --------------------------------------------------
# 10. Create output directories
# --------------------------------------------------

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 11. Latitude SHAP summary plot
# --------------------------------------------------

print("\nGenerating latitude SHAP summary plot...")

shap.summary_plot(
    latitude_shap,
    X_sample,
    show=False
)

plt.title("SHAP Summary - Destination Latitude")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "shap_latitude_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 12. Longitude SHAP summary plot
# --------------------------------------------------

print("Generating longitude SHAP summary plot...")

shap.summary_plot(
    longitude_shap,
    X_sample,
    show=False
)

plt.title("SHAP Summary - Destination Longitude")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "shap_longitude_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 13. Calculate latitude feature importance
# --------------------------------------------------

latitude_importance = pd.DataFrame({
    "feature": MODEL_FEATURES,
    "mean_absolute_shap": abs(latitude_shap).mean(axis=0)
}).sort_values(
    "mean_absolute_shap",
    ascending=False
)

latitude_importance.to_csv(
    METRICS_DIR / "shap_latitude_importance.csv",
    index=False
)

# --------------------------------------------------
# 14. Calculate longitude feature importance
# --------------------------------------------------

longitude_importance = pd.DataFrame({
    "feature": MODEL_FEATURES,
    "mean_absolute_shap": abs(longitude_shap).mean(axis=0)
}).sort_values(
    "mean_absolute_shap",
    ascending=False
)

longitude_importance.to_csv(
    METRICS_DIR / "shap_longitude_importance.csv",
    index=False
)

# --------------------------------------------------
# 15. Display feature importance
# --------------------------------------------------

print("\n========== LATITUDE FEATURE IMPORTANCE ==========")

print(latitude_importance.to_string(index=False))

print("\n========== LONGITUDE FEATURE IMPORTANCE ==========")

print(longitude_importance.to_string(index=False))

# --------------------------------------------------
# 16. Completion
# --------------------------------------------------

print("\n========== SHAP ANALYSIS COMPLETED ==========")

print("Saved figures:")
print("shap_latitude_summary.png")
print("shap_longitude_summary.png")

print("\nSaved metrics:")
print("shap_latitude_importance.csv")
print("shap_longitude_importance.csv")