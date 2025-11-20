#!/usr/bin/env bash
set -euo pipefail

VENV=.venv
if [ ! -d "$VENV" ]; then
  python -m venv $VENV
fi
source $VENV/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python src/ml/preprocess_eta.py --seq-len 10 --horizon 1
python src/ml/preprocess_conflict.py --q-thresh 15 --moving-frac 0.25
python src/ml/train_eta_lstm.py
python src/ml/train_conflict_dt.py
python src/ml/evaluate_models.py

echo "ETA model: $(realpath models/eta_predictor.h5)"
echo "Conflict model: $(realpath models/conflict_classifier.pkl)"
echo "Metrics dir: $(realpath artifacts/metrics)"
