from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

FILES_TO_CHECK = {
    "Baseline evaluation": (
        "results/metrics/complete_baseline_evaluation.csv"
    ),
    "Filtering experiment comparison": (
        "results/metrics/filtering_experiment_comparison.csv"
    ),
    "Validation filtering results": (
        "results/metrics/validation_filtered_prediction_results.csv"
    ),
    "Final selected configuration": (
        "results/metrics/final_selected_configuration.csv"
    ),
    "Final test comparison": (
        "results/metrics/final_test_comparison.csv"
    ),
    "McNemar test": (
        "results/metrics/mcnemar_test_results.csv"
    ),
    "Candidate ranking": (
        "results/metrics/candidate_ranking_summary.csv"
    ),
    "SHAP latitude importance": (
        "results/metrics/shap_latitude_importance.csv"
    ),
    "SHAP longitude importance": (
        "results/metrics/shap_longitude_importance.csv"
    ),
    "SHAP stability summary": (
        "results/metrics/shap_stability_summary.csv"
    ),
    "SHAP stability correlations": (
        "results/metrics/shap_stability_correlations.csv"
    ),
}

print("\n" + "=" * 65)
print("FINAL RESEARCH RESULTS AUDIT")
print("=" * 65)

for title, relative_path in FILES_TO_CHECK.items():

    file_path = ROOT / relative_path

    print("\n" + "-" * 65)
    print(title)
    print("-" * 65)

    if not file_path.exists():
        print("FILE NOT FOUND:", relative_path)
        continue

    try:
        df = pd.read_csv(file_path)

        print("File:", relative_path)
        print("Rows:", len(df))
        print("Columns:", list(df.columns))

        print("\nData:")
        print(df.to_string(index=False))

    except Exception as error:
        print("ERROR READING FILE:", error)

print("\n" + "=" * 65)
print("AUDIT DATA COLLECTION COMPLETED")
print("=" * 65)
