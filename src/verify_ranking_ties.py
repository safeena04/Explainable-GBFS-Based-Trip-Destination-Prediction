import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ranking_path = (
    ROOT / "results/metrics/candidate_level_ranking_details.csv"
)

prediction_dir = ROOT / "results/predictions"

ranking = pd.read_csv(ranking_path)

files = {
    "Baseline": "validation_baseline_predictions.csv",
    "E6": "validation_e6_predictions.csv"
}

for name, filename in files.items():

    print(f"\n========== {name} ==========")

    data = ranking[
        ranking["configuration"] == name
    ].copy()

    # Find minimum prediction distance for each rental
    minimum = data.groupby(
        "spoofed_id_rental"
    )["prediction_distance_m"].transform("min")

    # Identify candidates tied at minimum distance
    tied = data[
        (data["prediction_distance_m"] - minimum).abs() < 1e-9
    ]

    tie_counts = tied.groupby(
        "spoofed_id_rental"
    ).size()

    tied_rentals = tie_counts[
        tie_counts > 1
    ].index

    predictions = pd.read_csv(
        prediction_dir / filename
    ).set_index("spoofed_id_rental")

    print("Rentals with tied minimum distances:", len(tied_rentals))

    for rental_id in tied_rentals:

        candidates = tied[
            tied["spoofed_id_rental"] == rental_id
        ]

        correct_among_ties = candidates[
            "is_correct"
        ].any()

        actual_selected_correct = predictions.loc[
            rental_id,
            "is_correct_destination"
        ]

        print("\nRental:", rental_id)
        print("Correct candidate among ties:", correct_among_ties)
        print("Actual selected candidate correct:", actual_selected_correct)