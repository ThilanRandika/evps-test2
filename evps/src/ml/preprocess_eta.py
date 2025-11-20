import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .common import (
    BASE_DIR,
    PROCESSED_DIR,
    RAW_DATA_DIR,
    SCALERS_DIR,
    set_seed,
    save_json,
)

REQUIRED_COLS = [
    "run_id",
    "scenario_id",
    "veh_id",
    "simulation_time",
    "ev_speed",
    "ev_acceleration",
    "ev_distance_to_intersection",
    "next_tls_id",
]
FEATURE_COLS = ["ev_speed", "ev_acceleration", "ev_distance_to_intersection"]


def load_raw_eta() -> pd.DataFrame:
    eta_dir = RAW_DATA_DIR / "eta"
    csvs = sorted(eta_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No ETA CSVs found in {eta_dir}")
    frames = []
    for csv in csvs:
        df = pd.read_csv(csv)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    missing_cols = [c for c in REQUIRED_COLS if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    data = data.dropna(subset=REQUIRED_COLS)
    data["ev_distance_to_intersection"] = data["ev_distance_to_intersection"].clip(lower=0)
    data = data.sort_values(["run_id", "veh_id", "simulation_time"]).reset_index(drop=True)
    return data


def segment_vehicle(df: pd.DataFrame) -> List[Tuple[pd.DataFrame, float]]:
    segments: List[Tuple[pd.DataFrame, float]] = []
    rows = df.to_dict("records")
    if not rows:
        return segments
    start_idx = 0
    for idx in range(1, len(rows)):
        if rows[idx]["next_tls_id"] != rows[idx - 1]["next_tls_id"]:
            # segment ends at idx-1, arrival is current sim time
            segment_df = df.iloc[start_idx:idx]
            arrival_time = rows[idx]["simulation_time"]
            segments.append((segment_df.copy(), arrival_time))
            start_idx = idx
    # handle final segment
    final_segment = df.iloc[start_idx:]
    arrival_time = None
    dist_sub = final_segment["ev_distance_to_intersection"]
    time_sub = final_segment["simulation_time"]
    mask = dist_sub <= 0
    if mask.any():
        arrival_time = float(time_sub[mask].iloc[0])
    if arrival_time is not None:
        segments.append((final_segment.copy(), arrival_time))
    return segments


def compute_eta_labels(df: pd.DataFrame) -> pd.DataFrame:
    labeled_segments = []
    for (run_id, veh_id), vehicle_df in df.groupby(["run_id", "veh_id"]):
        vehicle_df = vehicle_df.sort_values("simulation_time")
        segments = segment_vehicle(vehicle_df)
        for seg_df, arrival_time in segments:
            if arrival_time is None:
                continue
            seg_df = seg_df.copy()
            seg_df["eta_true"] = (arrival_time - seg_df["simulation_time"]).clip(lower=0)
            seg_df = seg_df[seg_df["eta_true"] >= 0]
            labeled_segments.append(seg_df)
    if not labeled_segments:
        raise ValueError("No segments with defined arrival times for ETA labeling")
    return pd.concat(labeled_segments, ignore_index=True)


def build_sequences(df: pd.DataFrame, seq_len: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    sequences = []
    targets = []
    for (run_id, veh_id), vehicle_df in df.groupby(["run_id", "veh_id"], sort=False):
        vehicle_df = vehicle_df.sort_values("simulation_time")
        arr = vehicle_df[FEATURE_COLS + ["eta_true"]].to_numpy()
        for start in range(0, len(arr) - seq_len - horizon + 1):
            end = start + seq_len
            target_idx = end + horizon - 1
            window = arr[start:end, : len(FEATURE_COLS)]
            target = arr[target_idx, -1]
            sequences.append(window)
            targets.append(target)
    if not sequences:
        raise ValueError("No sequences built; check data volume and sequence length")
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)


def split_by_vehicle(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = df[["run_id", "veh_id"]].drop_duplicates().sample(frac=1, random_state=seed)
    n = len(keys)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    train_keys = keys.iloc[:train_end]
    val_keys = keys.iloc[train_end:val_end]
    test_keys = keys.iloc[val_end:]

    def filter_keys(sub_keys: pd.DataFrame) -> pd.DataFrame:
        merged = df.merge(sub_keys, on=["run_id", "veh_id"], how="inner")
        return merged

    return filter_keys(train_keys), filter_keys(val_keys), filter_keys(test_keys)


def main(seq_len: int, horizon: int, seed: int = 42) -> None:
    set_seed(seed)
    raw_df = load_raw_eta()
    labeled_df = compute_eta_labels(raw_df)
    train_df, val_df, test_df = split_by_vehicle(labeled_df, seed=seed)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[FEATURE_COLS])
    for split_df in [train_df, val_df, test_df]:
        split_df[FEATURE_COLS] = scaler.transform(split_df[FEATURE_COLS])

    X_train, y_train = build_sequences(train_df, seq_len, horizon)
    X_val, y_val = build_sequences(val_df, seq_len, horizon)
    X_test, y_test = build_sequences(test_df, seq_len, horizon)

    save_path = PROCESSED_DIR / "eta_sequences.npz"
    np.savez(
        save_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    scaler_path = SCALERS_DIR / "scaler_lstm.pkl"
    import joblib

    joblib.dump(scaler, scaler_path)

    manifest = {
        "seq_len": seq_len,
        "horizon": horizon,
        "feature_order": FEATURE_COLS,
        "n_train_samples": int(len(y_train)),
        "n_val_samples": int(len(y_val)),
        "n_test_samples": int(len(y_test)),
        "scaler_path": str(scaler_path.resolve()),
    }
    info_path = SCALERS_DIR / "scaler_lstm_features.txt"
    info_path.write_text(json.dumps(manifest, indent=2))

    meta_path = PROCESSED_DIR / "eta_sequences_manifest.json"
    save_json(meta_path, manifest)

    print(f"Saved sequences to {save_path}")
    print(f"Scaler saved to {scaler_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ETA data and build sequences")
    parser.add_argument("--seq-len", type=int, default=10, help="Sequence length")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(seq_len=args.seq_len, horizon=args.horizon, seed=args.seed)
