import pandas as pd
from pathlib import Path

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


def test_conflict_processed_columns_and_split():
    paths = {name: Path(f"data/processed/conflict_{name}.csv") for name in ["train", "val", "test"]}
    for p in paths.values():
        assert p.exists(), "Processed conflict split missing; run preprocessing"
    dfs = {name: pd.read_csv(path) for name, path in paths.items()}

    # Column presence
    for df in dfs.values():
        for col in FEATURE_COLS + ["label"]:
            assert col in df.columns, f"Missing column {col}"

    # Stratified split approx 70/15/15
    total = sum(len(df) for df in dfs.values())
    ratios = {name: len(df) / total for name, df in dfs.items()}
    assert 0.65 <= ratios["train"] <= 0.75
    assert 0.1 <= ratios["val"] <= 0.2
    assert 0.1 <= ratios["test"] <= 0.2

    # Check label distribution similarity
    train_label_ratio = dfs["train"]["label"].value_counts(normalize=True)
    for name, df in dfs.items():
        dist = df["label"].value_counts(normalize=True)
        for label in train_label_ratio.index:
            assert abs(train_label_ratio.get(label, 0) - dist.get(label, 0)) < 0.2
