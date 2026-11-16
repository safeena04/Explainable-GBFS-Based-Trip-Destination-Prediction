import sys
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_PATH, SCALER_PATH, MODEL_FEATURES, SCALED_FEATURES

# Load saved model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("\n========== MODEL INFORMATION ==========")

print("Model type:", type(model))
print("Model features:", getattr(model, "n_features_in_", "Not available"))

print("\n========== SCALER INFORMATION ==========")

print("Scaler type:", type(scaler))
print("Scaler features:", getattr(scaler, "n_features_in_", "Not available"))

print("\n========== FEATURE CONFIGURATION ==========")

print("Total model features:", len(MODEL_FEATURES))

for feature in MODEL_FEATURES:
    print("-", feature)

print("\nFeatures to be scaled:", len(SCALED_FEATURES))

for feature in SCALED_FEATURES:
    print("-", feature)

print("\n========== MODEL OUTPUT INFORMATION ==========")

if hasattr(model, "estimators_"):
    print("Number of estimators:", len(model.estimators_))

    for i, estimator in enumerate(model.estimators_):
        print(f"Estimator {i + 1}:", type(estimator))
print("\n========== PREDICTION TEST ==========")

import numpy as np

# Create one sample with 8 features
sample = np.zeros((1, 8))

prediction = model.predict(sample)

print("Input shape:", sample.shape)
print("Prediction:", prediction)
print("Prediction shape:", np.asarray(prediction).shape)
print("\nModel inspection completed!")