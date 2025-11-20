import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers

from .common import METRICS_DIR, MODELS_DIR, PROCESSED_DIR, SCALERS_DIR, mae, rmse, save_json, set_seed


DEFAULT_SEQ_LEN = 10
DEFAULT_HORIZON = 1


def build_model(seq_len: int, n_features: int = 3) -> tf.keras.Model:
    inputs = layers.Input(shape=(seq_len, n_features))
    x = layers.LSTM(128, return_sequences=False)(inputs)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="linear")(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model


def main(seq_len: int, horizon: int, seed: int = 42) -> None:
    set_seed(seed)
    data_path = PROCESSED_DIR / "eta_sequences.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing processed sequences at {data_path}")
    data = np.load(data_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    model = build_model(seq_len=seq_len, n_features=X_train.shape[-1])
    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5),
        callbacks.ModelCheckpoint(filepath=MODELS_DIR / "eta_predictor.h5", monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=cb,
        verbose=0,
    )

    preds = model.predict(X_test, verbose=0).flatten()
    metrics = {
        "mae": mae(y_test, preds),
        "rmse": rmse(y_test, preds),
        "n_test_samples": int(len(y_test)),
        "seq_len": seq_len,
    }
    metrics_path = METRICS_DIR / "eta_metrics.json"
    save_json(metrics_path, metrics)

    # Save training config
    config = {
        "seq_len": seq_len,
        "horizon": horizon,
        "seed": seed,
        "feature_order": ["ev_speed", "ev_acceleration", "ev_distance_to_intersection"],
        "scaler_path": str((SCALERS_DIR / "scaler_lstm.pkl").resolve()),
    }
    config_path = MODELS_DIR / "eta_predictor_config.json"
    save_json(config_path, config)

    print(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {metrics_path}")
    print(f"Model saved to {MODELS_DIR / 'eta_predictor.h5'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ETA LSTM model")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seq_len=args.seq_len, horizon=args.horizon, seed=args.seed)
