# Phase 2 Conflict Labeling Rules

The conflict minimizer uses deterministic heuristics to derive binary labels from raw SUMO snapshots:

| Feature | Description | Threshold | Unsafe Condition |
| --- | --- | --- | --- |
| `max_queue` | Maximum queue length across conflicting lanes | `--q-thresh` (default 15) | `max_queue > q_thresh` |
| `moving_lane_fraction` | Fraction of conflicting lanes with mean speed > 0.1 m/s | `--moving-frac` (default 0.25) | `moving_lane_fraction > moving_frac` |

If either condition is true, the sample is labeled **Unsafe**; otherwise it is labeled **Safe**.
