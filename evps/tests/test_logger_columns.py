"""Schema validation tests for logger outputs."""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
import pytest

ETA_COLUMNS = [
    "run_id",
    "scenario_id",
    "veh_id",
    "simulation_time",
    "ev_speed",
    "ev_acceleration",
    "ev_distance_to_intersection",
    "next_tls_id",
]

CONFLICT_COLUMNS = [
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


def _latest_csv(pattern: str) -> Path:
    files = sorted(glob.glob(pattern))
    if not files:
        pytest.skip(f"No files matched pattern: {pattern}. Run the data logger first.")
    return Path(files[-1])


def test_eta_columns_present() -> None:
    csv_path = _latest_csv("data/raw/eta/eta_*.csv")
    df = pd.read_csv(csv_path)
    assert all(column in df.columns for column in ETA_COLUMNS)


def test_conflict_columns_present() -> None:
    csv_path = _latest_csv("data/raw/conflicts/conflicts_*.csv")
    df = pd.read_csv(csv_path)
    assert all(column in df.columns for column in CONFLICT_COLUMNS)
    for column in ["conflicting_lane_ids", "queue_lengths", "mean_speeds"]:
        assert df[column].dtype == object
