import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

METRICS = ROOT / "results" / "metrics"
OUTPUT = ROOT / "results" / "figures"

OUTPUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# PAPER FIGURE SETTINGS
# --------------------------------------------------

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# --------------------------------------------------
# LOAD RESULTS
# --------------------------------------------------

df = pd.read_csv(
    METRICS / "final_test_ranking_results.csv"
)

strategies = [
    "Baseline + Nearest",
    "E11 + Nearest",
    "E11 + W3"
]

df = (
    df.set_index("strategy")
    .loc[strategies]
    .reset_index()
)

thresholds = [100, 250, 500, 1000]

labels = ["100 m", "250 m", "500 m", "1 km"]

print("\nFinal test results loaded successfully.")

# --------------------------------------------------
# FIGURE 1: ACCURACY COMPARISON
# --------------------------------------------------

accuracy_columns = [
    f"end_to_end_accuracy_{t}m_percent"
    for t in thresholds
]

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(thresholds))
width = 0.25

for i, strategy in enumerate(strategies):

    values = df.loc[
        df["strategy"] == strategy,
        accuracy_columns
    ].values.flatten()

    bars = ax.bar(
        x + (i - 1) * width,
        values,
        width,
        label=strategy
    )

    ax.bar_label(
        bars,
        fmt="%.2f",
        padding=3,
        fontsize=8
    )

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_xlabel("Geographic Distance Tolerance")
ax.set_ylabel("End-to-End Accuracy (%)")

ax.set_title(
    "Destination Prediction Accuracy at Different Tolerances"
)

ax.set_ylim(0, 85)

ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.35)

fig.tight_layout()

fig.savefig(
    OUTPUT / "accuracy_comparison.png"
)

plt.show()
plt.close(fig)

# --------------------------------------------------
# FIGURE 2: CANDIDATE AVAILABILITY
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(
    df["strategy"],
    df["availability_percent"],
    width=0.55
)

ax.bar_label(
    bars,
    labels=[
        f"{v:.2f}%"
        for v in df["availability_percent"]
    ],
    padding=4
)

ax.set_xlabel("Candidate Selection Strategy")
ax.set_ylabel("Candidate Availability (%)")

ax.set_title("Candidate Availability Comparison")

ax.set_ylim(0, 105)

ax.grid(axis="y", linestyle="--", alpha=0.35)

plt.xticks(rotation=10)

fig.tight_layout()

fig.savefig(
    OUTPUT / "candidate_availability.png"
)

plt.show()
plt.close(fig)

# --------------------------------------------------
# FIGURE 3: CORRECT PREDICTIONS
# --------------------------------------------------

correct_columns = [
    f"correct_{t}m"
    for t in thresholds
]

fig, ax = plt.subplots(figsize=(10, 6))

for strategy in strategies:

    values = df.loc[
        df["strategy"] == strategy,
        correct_columns
    ].values.flatten()

    ax.plot(
        thresholds,
        values,
        marker="o",
        linewidth=2,
        markersize=7,
        label=strategy
    )

ax.set_xlabel("Geographic Distance Tolerance")
ax.set_ylabel("Number of Correct Predictions")

ax.set_title(
    "Correct Predictions Across Geographic Tolerances"
)

ax.set_xticks(
    thresholds,
    labels
)

ax.set_ylim(0, 1800)

ax.legend()
ax.grid(True, linestyle="--", alpha=0.35)

fig.tight_layout()

fig.savefig(
    OUTPUT / "correct_predictions.png"
)

plt.show()
plt.close(fig)

# --------------------------------------------------
# FIGURE 4: W3 ACCURACY IMPROVEMENT
# --------------------------------------------------

baseline = df[
    df["strategy"] == "Baseline + Nearest"
].iloc[0]

e11 = df[
    df["strategy"] == "E11 + Nearest"
].iloc[0]

w3 = df[
    df["strategy"] == "E11 + W3"
].iloc[0]

baseline_accuracy = np.array([
    baseline[f"end_to_end_accuracy_{t}m_percent"]
    for t in thresholds
])

e11_accuracy = np.array([
    e11[f"end_to_end_accuracy_{t}m_percent"]
    for t in thresholds
])

w3_accuracy = np.array([
    w3[f"end_to_end_accuracy_{t}m_percent"]
    for t in thresholds
])

baseline_improvement = w3_accuracy - baseline_accuracy

e11_improvement = w3_accuracy - e11_accuracy

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(thresholds))
width = 0.35

bars1 = ax.bar(
    x - width / 2,
    baseline_improvement,
    width,
    label="W3 vs Baseline"
)

bars2 = ax.bar(
    x + width / 2,
    e11_improvement,
    width,
    label="W3 vs E11 Nearest"
)

ax.bar_label(
    bars1,
    fmt="%.2f",
    padding=3,
    fontsize=9
)

ax.bar_label(
    bars2,
    fmt="%.2f",
    padding=3,
    fontsize=9
)

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_xlabel("Geographic Distance Tolerance")

ax.set_ylabel(
    "Accuracy Improvement (Percentage Points)"
)

ax.set_title("Accuracy Improvement Achieved by W3")

ax.axhline(0, linewidth=1)

ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.35)

fig.tight_layout()

fig.savefig(
    OUTPUT / "w3_accuracy_improvement.png"
)

plt.show()
plt.close(fig)

# --------------------------------------------------
# COMPLETION
# --------------------------------------------------

print("\nAll research figures generated successfully!")

print("\nSaved figures:")

for file in OUTPUT.glob("*.png"):
    print(file.name)

print("\nOutput directory:", OUTPUT)