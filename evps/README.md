# AI-Driven Emergency Vehicle Preemption System (EVPS) – Phase 1

This repository implements Phase 1 of the AI-Driven Emergency Vehicle Preemption System
(EVPS). It scaffolds a reproducible SUMO environment, generates congestion-specific
scenarios, and logs master datasets that downstream ETA predictors and conflict
minimizers consume.

## Prerequisites

- Python 3.8+
- [SUMO](https://www.eclipse.org/sumo/) installed locally with the `SUMO_HOME`
  environment variable set (TraCI and `sumolib` are loaded from `SUMO_HOME/tools`).
- Bash shell (for `scripts/run_phase1.sh`).

## Quickstart

```bash
# create venv and install deps
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# generate scenarios
python src/data/scenario_generator.py --level low --seed 11
python src/data/scenario_generator.py --level medium --seed 12
python src/data/scenario_generator.py --level high --seed 13

# run data logger (headless) for each scenario
python src/data/data_logger.py --sumocfg data/scenarios/low/scenario.sumocfg --scenario-id low --nogui
python src/data/data_logger.py --sumocfg data/scenarios/medium/scenario.sumocfg --scenario-id medium --nogui
python src/data/data_logger.py --sumocfg data/scenarios/high/scenario.sumocfg --scenario-id high --nogui
```

Run everything end-to-end with:

```bash
bash scripts/run_phase1.sh
```

The script provisions a virtual environment, installs dependencies, generates all three
scenarios, runs the TraCI data logger headless, and prints the absolute paths to the
resulting ETA and conflict CSV files.

## Environment Variables & SUMO Tools

Both the scenario generator and data logger import SUMO's Python utilities dynamically via
`SUMO_HOME/tools`. If `SUMO_HOME` is missing or incorrect, the utilities fail fast with an
actionable error message. No SUMO wheels are installed via pip.

## Phase 1 Data Contract (Authoritative)

### ETA Predictor (time-series, logged every simulation step)

| Column | Type | Description |
|--------|------|-------------|
| run_id | str | Unique UUID per logger run |
| scenario_id | str | Identifier passed via CLI (e.g., `low`) |
| veh_id | str | SUMO vehicle id for the emergency vehicle |
| simulation_time | float | Simulation time in seconds |
| ev_speed | float | `traci.vehicle.getSpeed()` in m/s |
| ev_acceleration | float | `traci.vehicle.getAcceleration()` (fallback: finite difference) in m/s² |
| ev_distance_to_intersection | float | Distance (meters) to the next signalized junction based on `traci.vehicle.getNextTLS()` (fallback: lane summation) |
| next_tls_id | str | TLS id of the next intersection or empty if none |

### Conflict Minimizer Snapshot (one row per EV–TLS approach)

| Column | Type | Description |
|--------|------|-------------|
| run_id | str | Same UUID as the ETA log |
| scenario_id | str | Scenario identifier |
| veh_id | str | Emergency vehicle id |
| simulation_time | float | Time of the snapshot |
| intersection_id | str | Upcoming TLS id |
| conflicting_lane_ids | JSON list[str] | All lanes controlled by the TLS minus the EV's approach lanes |
| queue_lengths | JSON list[int] | `traci.lane.getLastStepHaltingNumber()` for each conflicting lane |
| mean_speeds | JSON list[float] | `traci.lane.getLastStepMeanSpeed()` for each conflicting lane |
| current_signal_phase | int | Current index from `traci.trafficlight.getPhase()` |

These contracts are enforced programmatically so that future ML phases can consume the
CSV outputs without modification.

## Repository Layout

```
evps/
├── configs/              # Scenario documentation
├── data/                 # Generated artifacts (raw and processed data are gitignored)
├── scripts/              # Automation entrypoints
├── src/                  # Python packages (scenario generator, data logger, SUMO utils)
└── tests/                # Pytest-based regression suite
```

## Testing

After generating data, run:

```bash
pytest -q
```

The tests validate the schema of the most recent ETA and conflict CSV files.

## Data Handling

All raw and processed datasets are written under `data/` but excluded from Git via
`.gitignore`. Keep personal runs out of version control to preserve repository hygiene.
