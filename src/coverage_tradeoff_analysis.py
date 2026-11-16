"""
Coverage and Candidate Count Trade-off Analysis.

Purpose:
Identify Pareto-optimal filtering configurations
using validation results.
"""

import pandas as pd
from pathlib import Path

import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import METRICS_DIR, C_MIN


# ---------------------------------------------------------
# LOAD VALIDATION RESULTS
# ---------------------------------------------------------

def load_validation_results():

    file_path = METRICS_DIR / "validation_filtering_results.csv"

    results = pd.read_csv(file_path)

    print("\nValidation results loaded successfully!")
    print("Total configurations:", len(results))

    return results
# ---------------------------------------------------------
# IDENTIFY PARETO-OPTIMAL CONFIGURATIONS
# ---------------------------------------------------------

def find_pareto_optimal(results):

    pareto_indices = []

    for i, current in results.iterrows():

        dominated = False

        for j, other in results.iterrows():

            if i == j:
                continue

            # Another configuration is better if:
            # 1. It has equal or higher coverage.
            # 2. It has equal or fewer average candidates.
            # 3. It is strictly better in at least one metric.

            better_coverage = (
                other["candidate_coverage_percent"]
                >= current["candidate_coverage_percent"]
            )

            fewer_candidates = (
                other["average_candidates_per_rental"]
                <= current["average_candidates_per_rental"]
            )

            strictly_better = (
                other["candidate_coverage_percent"]
                > current["candidate_coverage_percent"]
                or
                other["average_candidates_per_rental"]
                < current["average_candidates_per_rental"]
            )

            if (
                better_coverage
                and fewer_candidates
                and strictly_better
            ):
                dominated = True
                break

        if not dominated:
            pareto_indices.append(i)

    pareto_results = results.loc[pareto_indices].copy()

    return pareto_results

def select_configuration(results, min_coverage):
    eligible = results[
        results["candidate_coverage_percent"] >= min_coverage
    ].copy()

    if eligible.empty:
        raise ValueError(
            f"No configuration satisfies minimum coverage of {min_coverage}%"
        )

    selected = eligible.sort_values(
        by=[
            "average_candidates_per_rental",
            "candidate_coverage_percent"
        ],
        ascending=[True, False]
    ).iloc[0]

    return selected

# ---------------------------------------------------------
# DISPLAY AND SAVE PARETO RESULTS
# ---------------------------------------------------------

def run_tradeoff_analysis():

    # Load validation results
    results = load_validation_results()

    # Find Pareto-optimal configurations
    pareto_results = find_pareto_optimal(results)

    # Sort by candidate coverage
    pareto_results = pareto_results.sort_values(
        by="candidate_coverage_percent"
    )
    selected = select_configuration(results, C_MIN)

    # Display results
    print("\n========== PARETO-OPTIMAL CONFIGURATIONS ==========")

    print(
        pareto_results[
            [
                "experiment",
                "max_distance",
                "max_time_minutes",
                "candidate_coverage_percent",
                "average_candidates_per_rental"
            ]
        ].to_string(index=False)
    )

    # Save results
    output_path = METRICS_DIR / "pareto_optimal_configurations.csv"

    pareto_results.to_csv(output_path, index=False)

    print("\nPareto results saved to:")
    print(output_path)
    print("\n========== SELECTED CONFIGURATION ==========")
    print("Minimum required coverage:", C_MIN, "%")
    print("Selected experiment:", selected["experiment"])
    print("Maximum distance:", selected["max_distance"], "meters")
    print("Maximum time:", selected["max_time_minutes"], "minutes")
    print("Validation coverage:", selected["candidate_coverage_percent"], "%")
    print("Average candidates per rental:", selected["average_candidates_per_rental"])

    selected_path = METRICS_DIR / "selected_filtering_configuration.csv"
    selected.to_frame().T.to_csv(selected_path, index=False)

    print("\nSelected configuration saved to:", selected_path)


# ---------------------------------------------------------
# PROGRAM ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    run_tradeoff_analysis()