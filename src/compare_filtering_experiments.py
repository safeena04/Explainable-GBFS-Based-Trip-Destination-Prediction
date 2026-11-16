import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import METRICS_DIR

# Load validation experiment results
data = pd.read_csv(
    METRICS_DIR / "validation_filtered_prediction_results.csv"
)

# Select 1000m as the primary tolerance for this comparison
comparison = data[data["tolerance_m"] == 1000].copy()

# Keep relevant metrics
comparison = comparison[
    [
        "experiment",
        "candidate_availability_percent",
        "candidate_coverage_percent",
        "average_candidates",
        "conditional_accuracy_percent",
        "end_to_end_accuracy_percent"
    ]
]

# Calculate reduction relative to baseline
baseline = comparison[
    comparison["experiment"] == "Baseline"
].iloc[0]

comparison["candidate_reduction_percent"] = (
    (
        baseline["average_candidates"]
        - comparison["average_candidates"]
    )
    / baseline["average_candidates"]
    * 100
)

comparison["coverage_change_pp"] = (
    comparison["candidate_coverage_percent"]
    - baseline["candidate_coverage_percent"]
)

comparison["accuracy_change_pp"] = (
    comparison["end_to_end_accuracy_percent"]
    - baseline["end_to_end_accuracy_percent"]
)

# Save comparison
output_path = (
    METRICS_DIR / "filtering_experiment_comparison.csv"
)

comparison.to_csv(output_path, index=False)

print("\n========== FILTERING EXPERIMENT COMPARISON ==========")

print(
    comparison.round(2).to_string(index=False)
)

print("\nSaved to:", output_path)