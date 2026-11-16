from pathlib import Path
from scipy.stats import binomtest
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

METRICS = ROOT / "results" / "metrics"

ID = "spoofed_id_rental"
TOLERANCE = 1000

# Load selected predictions
details = pd.read_csv(
    METRICS / "final_test_selected_candidates.csv"
)

# Load all test rental events
rentals = pd.read_csv(
    METRICS / "baseline_geographic_coverage.csv"
)

from config import TEST_DATES

rentals["date"] = rentals["date"].astype(str)

rentals = rentals[
    rentals["date"].isin(TEST_DATES)
].copy()

rentals = rentals.rename(
    columns={"rental_id": ID}
)

# Start with all test rental IDs.
# Rentals without a candidate remain incorrect.
results = pd.DataFrame({
    ID: rentals[ID].unique()
})

# Convert selected prediction errors into correctness indicators.
for strategy in [
    "Baseline + Nearest",
    "E11 + Nearest",
    "E11 + W3"
]:

    subset = details[
        details["strategy"] == strategy
    ][[ID, "true_destination_error_m"]].copy()

    subset[f"{strategy}_correct"] = (
        subset["true_destination_error_m"] <= TOLERANCE
    )

    results = results.merge(
        subset[[ID, f"{strategy}_correct"]],
        on=ID,
        how="left"
    )

    results[f"{strategy}_correct"] = (
        results[f"{strategy}_correct"]
        .fillna(False)
        .astype(bool)
    )


def mcnemar_test(strategy_a, strategy_b):

    a = results[f"{strategy_a}_correct"]
    b = results[f"{strategy_b}_correct"]

    # A correct, B incorrect
    a_only = int((a & ~b).sum())

    # A incorrect, B correct
    b_only = int((~a & b).sum())

    discordant = a_only + b_only

    if discordant == 0:
        p_value = 1.0
    else:
        p_value = binomtest(
            min(a_only, b_only),
            discordant,
            p=0.5,
            alternative="two-sided"
        ).pvalue

    print("\nComparison:")
    print(strategy_a, "vs", strategy_b)
    print("A only correct:", a_only)
    print("B only correct:", b_only)
    print("Discordant pairs:", discordant)
    print("Exact McNemar p-value:", round(p_value, 6))

    if p_value < 0.05:
        print("Result: Statistically significant")
    else:
        print("Result: Not statistically significant")


print("Total paired test rentals:", len(results))

mcnemar_test(
    "E11 + Nearest",
    "E11 + W3"
)

mcnemar_test(
    "Baseline + Nearest",
    "E11 + W3"
)