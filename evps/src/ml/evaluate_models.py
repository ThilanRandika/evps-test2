import argparse
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .common import METRICS_DIR, MODELS_DIR, PROCESSED_DIR, mae, rmse, save_json

FEATURE_COLS_CONFLICT = [
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


def evaluate_eta() -> dict:
    data = np.load(PROCESSED_DIR / "eta_sequences.npz")
    X_test, y_test = data["X_test"], data["y_test"]
    model_path = MODELS_DIR / "eta_predictor.h5"
    model = tf.keras.models.load_model(model_path)
    preds = model.predict(X_test, verbose=0).flatten()
    metrics = {
        "mae": mae(y_test, preds),
        "rmse": rmse(y_test, preds),
        "n_test_samples": int(len(y_test)),
    }
    save_json(METRICS_DIR / "eta_metrics.json", metrics)
    return metrics


def evaluate_conflict() -> dict:
    test_df = pd.read_csv(PROCESSED_DIR / "conflict_test.csv")
    X_test = test_df[FEATURE_COLS_CONFLICT]
    y_test = test_df["label"]
    model = joblib.load(MODELS_DIR / "conflict_classifier.pkl")
    preds = model.predict(X_test)
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, pos_label="Unsafe"),
        "recall": recall_score(y_test, preds, pos_label="Unsafe"),
        "n_test_samples": int(len(y_test)),
    }
    save_json(METRICS_DIR / "conflict_metrics.json", metrics)
    return metrics


def main():
    eta_metrics = evaluate_eta()
    conflict_metrics = evaluate_conflict()
    print("ETA metrics:", json.dumps(eta_metrics, indent=2))
    print("Conflict metrics:", json.dumps(conflict_metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    args = parser.parse_args()
    main()
