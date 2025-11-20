import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .common import PROCESSED_DIR, RAW_DATA_DIR, SCALERS_DIR, save_json, set_seed

REQUIRED_COLS = [
    "run_id",
    "scenario_id",
    "veh_id",
    "simulation_time",
    "intersection_id",
    "conflicting_lane_ids",
    "queue_lengths",
    "mean_speeds",
    "current_signal_phase",
]
FEATURE_COLS = [
    "num_conflicting_lanes",
    "max_queue",
    "mean_queue",
    "sum_queue",
    "std_queue",
    "max_mean_speed",
    "mean_mean_speed",
    "moving_lane_fraction",
    "current_signal_phase",
]


def parse_json_list(value: str) -> List:
    try:
        return json.loads(value)
    except Exception as exc:
        raise ValueError(f"Failed to parse list from {value}") from exc


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        lane_ids = parse_json_list(row["conflicting_lane_ids"])
        queue_lengths = parse_json_list(row["queue_lengths"])
        mean_speeds = parse_json_list(row["mean_speeds"])
        min_len = min(len(lane_ids), len(queue_lengths), len(mean_speeds))
        if min_len == 0:
            # If empty, pad with zeros
            lane_ids = []
            queue_lengths = []
            mean_speeds = []
        else:
            lane_ids = lane_ids[:min_len]
            queue_lengths = queue_lengths[:min_len]
            mean_speeds = mean_speeds[:min_len]
        num_conflicting = len(lane_ids)
        if num_conflicting == 0:
            max_queue = mean_queue = sum_queue = std_queue = 0.0
            max_mean_speed = mean_mean_speed = moving_lane_fraction = 0.0
        else:
            queue_arr = np.array(queue_lengths, dtype=float)
            speed_arr = np.array(mean_speeds, dtype=float)
            max_queue = float(queue_arr.max())
            mean_queue = float(queue_arr.mean())
            sum_queue = float(queue_arr.sum())
            std_queue = float(queue_arr.std()) if len(queue_arr) >= 2 else 0.0
            max_mean_speed = float(speed_arr.max())
            mean_mean_speed = float(speed_arr.mean())
            moving_lane_fraction = float((speed_arr > 0.1).sum() / num_conflicting)
        records.append(
            {
                "run_id": row["run_id"],
                "scenario_id": row["scenario_id"],
                "veh_id": row["veh_id"],
                "simulation_time": row["simulation_time"],
                "intersection_id": row["intersection_id"],
                "num_conflicting_lanes": num_conflicting,
                "max_queue": max_queue,
                "mean_queue": mean_queue,
                "sum_queue": sum_queue,
                "std_queue": std_queue,
                "max_mean_speed": max_mean_speed,
                "mean_mean_speed": mean_mean_speed,
                "moving_lane_fraction": moving_lane_fraction,
                "current_signal_phase": row["current_signal_phase"],
            }
        )
    return pd.DataFrame(records)


def label_rows(df: pd.DataFrame, q_thresh: float, moving_frac: float) -> pd.DataFrame:
    df = df.copy()
    unsafe_mask = (df["max_queue"] > q_thresh) | (df["moving_lane_fraction"] > moving_frac)
    df["label"] = np.where(unsafe_mask, "Unsafe", "Safe")
    return df


def stratified_split(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=seed, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df["label"])
    return train_df, val_df, test_df


def main(q_thresh: float, moving_frac: float, seed: int = 42) -> None:
    set_seed(seed)
    conflict_dir = RAW_DATA_DIR / "conflicts"
    csvs = sorted(conflict_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No conflict CSVs found in {conflict_dir}")
    frames = [pd.read_csv(p) for p in csvs]
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.dropna(subset=REQUIRED_COLS)
    features_df = build_features(df)
    labeled_df = label_rows(features_df, q_thresh=q_thresh, moving_frac=moving_frac)
    train_df, val_df, test_df = stratified_split(labeled_df, seed=seed)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[FEATURE_COLS])

    def transform_split(split_df: pd.DataFrame) -> pd.DataFrame:
        transformed = split_df.copy()
        transformed[FEATURE_COLS] = scaler.transform(transformed[FEATURE_COLS])
        return transformed

    train_scaled = transform_split(train_df)
    val_scaled = transform_split(val_df)
    test_scaled = transform_split(test_df)

    for name, split in [("train", train_scaled), ("val", val_scaled), ("test", test_scaled)]:
        out_path = PROCESSED_DIR / f"conflict_{name}.csv"
        split.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")

    scaler_path = SCALERS_DIR / "scaler_conflict.pkl"
    import joblib

    joblib.dump(scaler, scaler_path)
    manifest = {
        "feature_order": FEATURE_COLS,
        "q_thresh": q_thresh,
        "moving_frac": moving_frac,
        "scaler_path": str(scaler_path.resolve()),
    }
    info_path = SCALERS_DIR / "scaler_conflict_features.txt"
    info_path.write_text(json.dumps(manifest, indent=2))
    meta_path = PROCESSED_DIR / "conflict_manifest.json"
    save_json(meta_path, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess conflict data")
    parser.add_argument("--q-thresh", type=float, default=15, help="Queue length threshold for unsafe")
    parser.add_argument("--moving-frac", type=float, default=0.25, help="Moving lane fraction threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(q_thresh=args.q_thresh, moving_frac=args.moving_frac, seed=args.seed)
