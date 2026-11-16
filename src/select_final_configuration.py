import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import METRICS_DIR

# ==================================================
# 1. CONFIGURATION
# ==================================================

PRIMARY_TOLERANCE = 1000

# ==================================================
# 2. LOAD BASELINE AND E1-E6 ACCURACY
# ==================================================

accuracy = pd.read_csv(
    METRICS_DIR / "validation_geographic_accuracy.csv"
)

accuracy = accuracy[
    accuracy["tolerance_m"] == PRIMARY_TOLERANCE
].copy()

accuracy = accuracy[
    [
        "configuration",
        "tolerance_m",
        "selected_candidate_end_to_end_accuracy_percent",
        "selected_candidate_conditional_accuracy_percent"
    ]
].copy()

# ==================================================
# 3. LOAD BASELINE AND E1-E6 COVERAGE
# ==================================================

base_coverage = pd.read_csv(
    METRICS_DIR / "validation_geographic_filtering_results.csv"
)

base_coverage = base_coverage[
    [
        "experiment",
        "candidate_coverage_1000m_percent",
        "candidate_availability_percent",
        "average_candidates_per_rental"
    ]
].copy()

base_coverage = base_coverage.rename(
    columns={
        "experiment": "configuration",
        "candidate_coverage_1000m_percent":
            "candidate_coverage_percent",
        "average_candidates_per_rental":
            "average_candidates"
    }
)

base_results = accuracy.merge(
    base_coverage,
    on="configuration",
    how="inner",
    validate="one_to_one"
)

base_results["candidate_pairs"] = float("nan")

# ==================================================
# 4. LOAD E7-E9 ACCURACY
# ==================================================

extended_accuracy = pd.read_csv(
    METRICS_DIR / "extended_validation_results.csv"
)

extended_accuracy = extended_accuracy[
    extended_accuracy["tolerance_m"] == PRIMARY_TOLERANCE
].copy()

extended_accuracy = extended_accuracy.rename(
    columns={
        "end_to_end_accuracy_percent":
            "selected_candidate_end_to_end_accuracy_percent",
        "conditional_accuracy_percent":
            "selected_candidate_conditional_accuracy_percent"
    }
)

extended_accuracy = extended_accuracy[
    [
        "configuration",
        "tolerance_m",
        "selected_candidate_end_to_end_accuracy_percent",
        "selected_candidate_conditional_accuracy_percent"
    ]
].copy()

# ==================================================
# 5. LOAD E7-E9 GEOGRAPHICAL COVERAGE
# ==================================================

extended_coverage = pd.read_csv(
    METRICS_DIR / "extended_validation_geographic_coverage.csv"
)

extended_coverage = extended_coverage.rename(
    columns={
        "experiment": "configuration",
        "coverage_1000m_percent":
            "candidate_coverage_percent",
        "availability_percent":
            "candidate_availability_percent"
    }
)

extended_coverage = extended_coverage[
    [
        "configuration",
        "candidate_coverage_percent",
        "candidate_availability_percent",
        "average_candidates",
        "candidate_pairs"
    ]
].copy()

extended_results = extended_accuracy.merge(
    extended_coverage,
    on="configuration",
    how="inner",
    validate="one_to_one"
)

# ==================================================
# 6. LOAD E10-E15 EXPANDED EXPERIMENTS
# ==================================================

expanded = pd.read_csv(
    METRICS_DIR / "expanded_validation_prediction_results.csv"
)

expanded = expanded[
    expanded["tolerance_m"] == PRIMARY_TOLERANCE
].copy()

expanded = expanded.rename(
    columns={
        "candidate_coverage_1000m_percent":
            "candidate_coverage_percent",
        "end_to_end_accuracy_percent":
            "selected_candidate_end_to_end_accuracy_percent",
        "conditional_accuracy_percent":
            "selected_candidate_conditional_accuracy_percent"
    }
)

expanded = expanded[
    [
        "configuration",
        "tolerance_m",
        "candidate_pairs",
        "candidate_availability_percent",
        "candidate_coverage_percent",
        "average_candidates",
        "selected_candidate_end_to_end_accuracy_percent",
        "selected_candidate_conditional_accuracy_percent"
    ]
].copy()

# ==================================================
# 7. COMBINE ALL CONFIGURATIONS
# ==================================================

results = pd.concat(
    [
        base_results,
        extended_results,
        expanded
    ],
    ignore_index=True
)

expected = {
    "Baseline",
    "E1", "E2", "E3", "E4", "E5", "E6",
    "E7", "E8", "E9",
    "E10", "E11", "E12", "E13", "E14", "E15"
}

found = set(results["configuration"])

if found != expected:
    raise ValueError(
        f"Configuration mismatch.\n"
        f"Missing: {expected - found}\n"
        f"Unexpected: {found - expected}"
    )

if results["configuration"].duplicated().any():
    raise ValueError("Duplicate configuration records found.")

# ==================================================
# 8. DEFINE BASELINE ACCURACY THRESHOLD
# ==================================================

baseline = results[
    results["configuration"] == "Baseline"
].iloc[0]

A_MIN = baseline[
    "selected_candidate_end_to_end_accuracy_percent"
]

print("\n========== SELECTION CRITERION ==========")

print(f"Primary tolerance: {PRIMARY_TOLERANCE} metres")
print(f"Baseline accuracy: {A_MIN:.2f}%")
print(f"Minimum acceptable accuracy: {A_MIN:.2f}%")

# ==================================================
# 9. CHECK ELIGIBILITY
# ==================================================

proposed = results[
    results["configuration"] != "Baseline"
].copy()

proposed["eligible"] = (
    proposed[
        "selected_candidate_end_to_end_accuracy_percent"
    ] >= A_MIN
)

display_columns = [
    "configuration",
    "candidate_pairs",
    "candidate_availability_percent",
    "candidate_coverage_percent",
    "average_candidates",
    "selected_candidate_end_to_end_accuracy_percent",
    "eligible"
]

print("\n========== ALL CONFIGURATION RESULTS ==========")

print(
    proposed[display_columns]
    .sort_values("configuration")
    .to_string(index=False)
)

eligible = proposed[
    proposed["eligible"]
].copy()

# ==================================================
# 10. SELECT FINAL CONFIGURATION
# ==================================================

if eligible.empty:
    print(
        "\nNo proposed configuration satisfies "
        "the minimum accuracy constraint."
    )
    raise SystemExit(1)

selected = eligible.sort_values(
    by=[
        "candidate_coverage_percent",
        "average_candidates"
    ],
    ascending=[False, True]
).iloc[0]

print("\n========== FINAL SELECTED CONFIGURATION ==========")

print(f"Selected configuration: {selected['configuration']}")

print(
    f"Candidate coverage: "
    f"{selected['candidate_coverage_percent']:.2f}%"
)

print(
    f"Candidate availability: "
    f"{selected['candidate_availability_percent']:.2f}%"
)

print(
    f"Average candidates: "
    f"{selected['average_candidates']:.2f}"
)

print(
    f"Validation end-to-end accuracy: "
    f"{selected['selected_candidate_end_to_end_accuracy_percent']:.2f}%"
)

# ==================================================
# 11. SAVE FINAL SELECTION
# ==================================================

output_columns = [
    "configuration",
    "tolerance_m",
    "candidate_pairs",
    "candidate_coverage_percent",
    "candidate_availability_percent",
    "average_candidates",
    "selected_candidate_end_to_end_accuracy_percent",
    "selected_candidate_conditional_accuracy_percent"
]

output = selected[output_columns].to_frame().T

output.to_csv(
    METRICS_DIR / "final_selected_configuration_corrected.csv",
    index=False
)

results.to_csv(
    METRICS_DIR / "all_configurations_corrected.csv",
    index=False
)

print("\nFinal configuration saved to:")
print(METRICS_DIR / "final_selected_configuration_corrected.csv")

print("\nComplete comparison saved to:")
print(METRICS_DIR / "all_configurations_corrected.csv")