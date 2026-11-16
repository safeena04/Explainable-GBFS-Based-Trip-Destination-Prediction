"""
setup_project.py -- builds the project folder and copies the original assets into it.

Usage (from the folder that contains this file):
    python setup_project.py "C:\\path\\to\\trip-destination-prediction-main"
Optional second argument = where to create the project (default: a folder named
Trip_Destination_Prediction next to the original repo folder).

Safe to re-run: existing files are never overwritten.
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

FOLDERS = [
    "data/raw", "data/processed", "ml", "src", "experiments", "dashboard",
    "notebooks", "tests", "results/figures", "results/metrics",
    "results/predictions", "results/reports",
]

DATA_README = """# Data

`raw/` holds the 7 daily CSV files (2024-03-01 .. 2024-03-07, Heidelberg) copied unchanged
from the original repository. One row = one COMPLETE trip (rental + return already paired
by vehicle_id). The pipeline hides vehicle_id from the candidate generator and uses it only
to know the true destination for evaluation.

Columns: vehicle_id, timestamp_lend, lng_lend, lat_lend, timestamp_returned, lng_returned,
lat_returned, current_range_meters_lend, pedelec_battery_lend, pedelec_battery_returned,
current_range_meters_returned.

Usage restriction (from the original README): the dataset is provided for reproducing the
experiments of Kerger and Stuckenschmidt. Any other use needs the authors' permission.

`processed/` is for files our own code generates. Never edit `raw/` by hand.
"""

GITIGNORE = """.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
results/predictions/
.DS_Store
"""


def copy_if_missing(src: Path, dst: Path) -> None:
    if dst.exists():
        print(f"  skip (already exists): {dst}")
    elif not src.exists():
        print(f"  WARNING source missing: {src}")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  copied: {dst}")


def write_if_missing(dst: Path, text: str) -> None:
    if dst.exists():
        print(f"  skip (already exists): {dst}")
    else:
        dst.write_text(text, encoding="utf-8")
        print(f"  wrote : {dst}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    repo = Path(sys.argv[1]).expanduser().resolve()
    if not (repo / "main.ipynb").exists() or not (repo / "ml").exists():
        print(f"ERROR: {repo} does not look like the original repository (main.ipynb / ml missing).")
        return 1
    project = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 else repo.parent / "Trip_Destination_Prediction"

    print(f"Original repo : {repo}\nNew project   : {project}\n")
    for folder in FOLDERS:
        (project / folder).mkdir(parents=True, exist_ok=True)

    print("Copying original data and model files:")
    for csv_file in sorted((repo / "data").glob("*.csv")):
        copy_if_missing(csv_file, project / "data" / "raw" / csv_file.name)
    copy_if_missing(repo / "ml" / "model.pkl", project / "ml" / "model.pkl")
    copy_if_missing(repo / "ml" / "scaler.pkl", project / "ml" / "scaler.pkl")
    copy_if_missing(repo / "main.ipynb", project / "notebooks" / "original_main.ipynb")
    copy_if_missing(repo / "LICENSE", project / "LICENSE_original_repo")

    print("\nCopying our Phase 1 files:")
    for name in ("config.py", "requirements.txt", "check_environment.py"):
        copy_if_missing(HERE / name, project / name)

    print("\nWriting helper files:")
    write_if_missing(project / "data" / "README.md", DATA_README)
    write_if_missing(project / ".gitignore", GITIGNORE)

    print(f"\nDone. Next:\n  cd \"{project}\"\n  python -m venv .venv\n  .venv\\Scripts\\activate\n"
          "  pip install -r requirements.txt\n  python check_environment.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
