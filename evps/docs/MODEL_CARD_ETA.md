# ETA Predictor Model Card

## Intended Use
Predict remaining time-to-arrival for an emergency vehicle approaching a traffic light. Inputs are sliding windows of scaled telemetry features.

## Data
- Source: Phase 1 ETA logs (`data/raw/eta/*.csv`).
- Features: `ev_speed`, `ev_acceleration`, `ev_distance_to_intersection` (scaled via MinMaxScaler [0,1]).
- Labels: Ground-truth ETA derived per vehicle/TLS segment using arrival-time detection.
- Sequence length: 10 timesteps; horizon: 1.

## Model
- Architecture: LSTM(128) → Dropout(0.25) → Dense(1, linear).
- Loss/Optimizer: MSE with Adam (lr=1e-3).
- Callbacks: EarlyStopping(patience=8), ReduceLROnPlateau, ModelCheckpoint (best to `models/eta_predictor.h5`).

## Training
- Train/val/test split by vehicle (70/15/15) to avoid leakage.
- Features scaled with `artifacts/scalers/scaler_lstm.pkl`.
- Metrics saved to `artifacts/metrics/eta_metrics.json`.

## Evaluation
- Reported metrics: MAE, RMSE on the held-out test set.

## Limitations
- Depends on accurate arrival-time detection; noisy or missing TLS transitions reduce label quality.
- Trained on simulated SUMO data only; real-world generalization is unverified.
- Sequence length fixed at 10; longer temporal dependencies are not captured.
