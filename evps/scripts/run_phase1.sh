#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
  python -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

pip install --upgrade pip >/dev/null
pip install -r "$ROOT_DIR/requirements.txt"

LEVELS=(low medium high)
SEEDS=(11 12 13)

for idx in "${!LEVELS[@]}"; do
  LEVEL="${LEVELS[$idx]}"
  SEED="${SEEDS[$idx]}"
  echo "[EVPS] Generating $LEVEL scenario (seed=$SEED)"
  python "$ROOT_DIR/src/data/scenario_generator.py" --level "$LEVEL" --seed "$SEED"
  SUMOCFG="$ROOT_DIR/data/scenarios/$LEVEL/scenario.sumocfg"
  echo "[EVPS] Logging data for $LEVEL scenario"
  python "$ROOT_DIR/src/data/data_logger.py" --sumocfg "$SUMOCFG" --scenario-id "$LEVEL" --nogui
  ETA_FILE=$(ls -t "$ROOT_DIR/data/raw/eta/eta_${LEVEL}_"*.csv | head -n 1)
  CONFLICT_FILE=$(ls -t "$ROOT_DIR/data/raw/conflicts/conflicts_${LEVEL}_"*.csv | head -n 1)
  echo "[EVPS] Latest ETA CSV for $LEVEL: $(realpath "$ETA_FILE")"
  echo "[EVPS] Latest conflict CSV for $LEVEL: $(realpath "$CONFLICT_FILE")"
  echo "------------------------------------------------------------"
fi
