# Scenario Configuration Guide

The SUMO scenarios for the AI-Driven Emergency Vehicle Preemption System (EVPS) are
generated dynamically using `src/data/scenario_generator.py`. The resulting assets are
stored under `data/scenarios/<level>/`:

- `network.net.xml` – network description generated via `netgenerate` (grid network).
- `routes.rou.xml` – route file created by `randomTrips.py` plus the injected emergency vehicle (EV).
- `scenario.sumocfg` – SUMO configuration referencing the network and routes for the level.

## Congestion Levels

| Level  | Default VPH | Notes |
|--------|-------------|-------|
| Low    | 300         | Light traffic, useful for baseline ETA measurements |
| Medium | 900         | Typical urban load |
| High   | 1800        | Stress-test conditions with dense traffic |

You can override defaults via CLI flags when running the scenario generator. Each level
produces deterministic assets when the same seed is reused, enabling reproducible
experiments.
