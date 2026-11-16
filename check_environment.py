"""
check_environment.py -- Phase 1 sanity check.

Run from the project root:    python check_environment.py

It checks: Python version, packages, project files, the CSV layout, that the
pretrained scaler/model load, that one test prediction works, and whether SHAP
can explain the pickled model. It changes nothing on disk.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

FAILED = []


def report(status: str, label: str, detail: str = "") -> None:
    """status is OK, WARN or FAIL (ASCII only, so Windows consoles print it fine)."""
    print(f"[{status:4s}] {label}" + (f"  --  {detail}" if detail else ""))
    if status == "FAIL":
        FAILED.append(label)


def check_python() -> None:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    report("OK" if ok else "FAIL", "Python >= 3.10", f"found {v.major}.{v.minor}.{v.micro}")


def check_packages() -> None:
    required_now = {"numpy": "numpy", "pandas": "pandas", "joblib": "joblib",
                    "sklearn": "scikit-learn", "xgboost": "xgboost",
                    "geopy": "geopy", "tqdm": "tqdm"}
    needed_later = {"shap": "shap", "matplotlib": "matplotlib", "seaborn": "seaborn",
                    "folium": "folium", "streamlit": "streamlit"}
    for table, missing_status in ((required_now, "FAIL"), (needed_later, "WARN")):
        for module, pip_name in table.items():
            try:
                mod = __import__(module)
                report("OK", f"import {module}", getattr(mod, "__version__", "version unknown"))
            except Exception as exc:  # ImportError or a broken install
                report(missing_status, f"import {module}", f"{type(exc).__name__}: {exc}  (pip install {pip_name})")


def check_files() -> None:
    for label, path in (("model.pkl", config.MODEL_PATH), ("scaler.pkl", config.SCALER_PATH)):
        if path.exists():
            report("OK", f"found {label}", f"{path.stat().st_size / 1e6:.1f} MB")
        else:
            report("FAIL", f"found {label}", f"missing: {path}")
    csvs = sorted(config.DATA_RAW_DIR.glob("*.csv"))
    report("OK" if csvs else "FAIL", "CSV files in data/raw", f"{len(csvs)} file(s)")


def check_data():
    import pandas as pd
    csvs = sorted(config.DATA_RAW_DIR.glob("*.csv"))
    if not csvs:
        return None
    frames = []
    for f in csvs:
        df = pd.read_csv(f)
        if list(df.columns) != config.RAW_COLUMNS:
            report("FAIL", f"columns of {f.name}", f"unexpected: {list(df.columns)}")
            return None
        df["file_date"] = f.stem
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    report("OK", "CSV columns match config.RAW_COLUMNS", f"{len(data)} trips in {len(csvs)} days")
    missing_dates = [d for d in config.VAL_DATES + config.TEST_DATES if d not in set(data.file_date)]
    report("OK" if not missing_dates else "FAIL", "VAL/TEST dates exist as files",
           "all present" if not missing_dates else f"missing: {missing_dates}")
    n_null = int(data.isna().sum().sum())
    report("OK" if n_null == 0 else "WARN", "missing values in CSVs", str(n_null))
    return data


def check_model(data) -> None:
    import joblib
    import numpy as np
    import pandas as pd
    from geopy.distance import geodesic
    try:
        scaler = joblib.load(config.SCALER_PATH)
        model = joblib.load(config.MODEL_PATH)
    except Exception as exc:
        report("FAIL", "load scaler/model with joblib", f"{type(exc).__name__}: {exc}")
        return
    report("OK", "load scaler/model with joblib", f"{type(model).__name__}, {type(scaler).__name__}")

    names = list(getattr(scaler, "feature_names_in_", []))
    report("OK" if names == config.SCALED_FEATURES else "FAIL",
           "scaler feature names == config.SCALED_FEATURES", str(names))
    booster_names = model.get_booster().feature_names
    report("OK" if booster_names == config.MODEL_FEATURES else "FAIL",
           "model feature names == config.MODEL_FEATURES", str(booster_names))
    try:
        model.get_params()
    except Exception as exc:  # known when the pickle was written by a different xgboost version
        report("WARN", "model.get_params()", f"{type(exc).__name__}: {exc}  (pickle/xgboost version mismatch; predict may still work)")

    if data is None:
        return
    # Plumbing test only: one real trip, features built the way the ORIGINAL notebook builds them.
    # Which unit/sign convention is actually correct is investigated in Phase 3.
    row = data.iloc[0]
    t0, t1 = pd.Timestamp(row.timestamp_lend), pd.Timestamp(row.timestamp_returned)
    time_diff = (t1 - t0).total_seconds()
    dist = int(geodesic((row.lat_lend, row.lng_lend), (row.lat_returned, row.lng_returned)).meters)
    range_diff = row.current_range_meters_lend - row.current_range_meters_returned
    speed = (dist / 1000) / (time_diff / 3600) if time_diff > 0 else 0
    speed_rng = (range_diff / 1000) / (time_diff / 3600) if time_diff > 0 else 0
    X = pd.DataFrame([{
        "lat_lend": row.lat_lend, "lng_lend": row.lng_lend, "time_diff": time_diff,
        "battery_diff": row.pedelec_battery_lend - row.pedelec_battery_returned,
        "range_diff": range_diff, "distance": dist,
        "mean_speed_distance_based": speed, "mean_speed_range_based": speed_rng,
    }])[config.MODEL_FEATURES]
    X[config.SCALED_FEATURES] = scaler.transform(X[config.SCALED_FEATURES])
    try:
        pred = np.asarray(model.predict(X))
        ok = pred.shape == (1, 2)
        away = geodesic((pred[0][0], pred[0][1]), (row.lat_lend, row.lng_lend)).meters
        report("OK" if ok else "FAIL", "test prediction returns [[lat, lng]]",
               f"shape {pred.shape}, value {pred[0].round(5).tolist()}, {away:.0f} m from the rental point")
    except Exception as exc:
        report("FAIL", "test prediction", f"{type(exc).__name__}: {exc}")
        return

    try:
        import shap
        values = np.asarray(shap.TreeExplainer(model).shap_values(X))
        report("OK", "SHAP TreeExplainer on the pickled model", f"output shape {values.shape}")
    except Exception as exc:
        report("WARN", "SHAP TreeExplainer on the pickled model",
               f"{type(exc).__name__}: {str(exc)[:150]}  (we have a fallback, tell me this line)")


def main() -> None:
    print(f"Project root: {ROOT}\n")
    check_python()
    check_packages()
    check_files()
    data = check_data()
    check_model(data)
    print()
    if FAILED:
        print(f"RESULT: {len(FAILED)} check(s) FAILED -> {FAILED}")
    else:
        print("RESULT: all required checks passed.")


if __name__ == "__main__":
    main()
