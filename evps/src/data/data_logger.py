"""Run SUMO via TraCI and log ETA/conflict datasets."""
from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import pandas as pd

from src.utils.sumo_paths import check_binary, ensure_sumo_imports

ensure_sumo_imports()
import traci  # type: ignore


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log EVPS SUMO data via TraCI")
    parser.add_argument("--sumocfg", type=Path, required=True, help="Path to scenario.sumocfg")
    parser.add_argument("--scenario-id", required=True, help="Identifier for this scenario run")
    parser.add_argument("--outdir", type=Path, default=Path("data/raw"), help="Output directory root")
    parser.add_argument("--ev-id-prefix", default="ev", help="Vehicle id prefix for EVs")
    parser.add_argument("--ev-type-hint", default="emergency", help="Vehicle type id for EVs")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gui", dest="gui", action="store_true", help="Use sumo-gui")
    group.add_argument("--nogui", dest="gui", action="store_false", help="Use headless sumo")
    parser.set_defaults(gui=False)
    return parser.parse_args()



def start_traci(sumocfg: Path, gui: bool) -> None:
    binary = "sumo-gui" if gui else "sumo"
    cmd = [check_binary(binary), "-c", str(sumocfg), "--start", "--quit-on-end"]
    traci.start(cmd)


def is_emergency_vehicle(veh_id: str, prefix: str, type_hint: str) -> bool:
    if veh_id.startswith(prefix):
        return True
    try:
        return traci.vehicle.getTypeID(veh_id) == type_hint
    except traci.TraCIException:
        return False


def build_lane_tls_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for tls_id in traci.trafficlight.getIDList():
        for lane in traci.trafficlight.getControlledLanes(tls_id):
            if lane not in mapping:
                mapping[lane] = tls_id
    return mapping


def estimate_distance_to_tls(veh_id: str, lane_to_tls: Dict[str, str]) -> Tuple[float, Optional[str]]:
    next_tls = traci.vehicle.getNextTLS(veh_id)
    if next_tls:
        tls_id, _, distance, _ = next_tls[0]
        return distance, tls_id

    lane_id = traci.vehicle.getLaneID(veh_id)
    lane_pos = traci.vehicle.getLanePosition(veh_id)
    try:
        remaining = traci.lane.getLength(lane_id) - lane_pos
    except traci.TraCIException:
        remaining = float("nan")
    if lane_id in lane_to_tls:
        return remaining, lane_to_tls[lane_id]

    route = list(traci.vehicle.getRoute(veh_id))
    current_edge = lane_id.rsplit("_", 1)[0] if "_" in lane_id else traci.vehicle.getRoadID(veh_id)
    distance = remaining
    tls_id: Optional[str] = None
    try:
        start_idx = route.index(current_edge)
    except ValueError:
        start_idx = -1
    if start_idx >= 0:
        for edge_id in route[start_idx + 1 :]:
            lane_candidate = f"{edge_id}_0"
            try:
                distance += traci.lane.getLength(lane_candidate)
            except traci.TraCIException:
                continue
            if lane_candidate in lane_to_tls:
                tls_id = lane_to_tls[lane_candidate]
                break
    return distance, tls_id


def ev_lane_ids(veh_id: str) -> List[str]:
    lane_id = traci.vehicle.getLaneID(veh_id)
    if "_" not in lane_id:
        return [lane_id]
    edge_id = lane_id.rsplit("_", 1)[0]
    try:
        lane_count = traci.edge.getLaneNumber(edge_id)
    except traci.TraCIException:
        lane_count = 1
    return [f"{edge_id}_{idx}" for idx in range(lane_count)]


def conflicting_lanes(tls_id: str, ev_lanes: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lane in traci.trafficlight.getControlledLanes(tls_id):
        if lane in seen or lane in ev_lanes:
            continue
        seen.add(lane)
        result.append(lane)
    return result


def snapshot_conflict(
    records: List[Dict[str, object]],
    run_id: str,
    scenario_id: str,
    veh_id: str,
    simulation_time: float,
    tls_id: str,
    ev_lanes: List[str],
) -> None:
    lanes = conflicting_lanes(tls_id, ev_lanes)
    queue_lengths = [int(traci.lane.getLastStepHaltingNumber(lane)) for lane in lanes]
    mean_speeds = [float(traci.lane.getLastStepMeanSpeed(lane)) for lane in lanes]
    phase = int(traci.trafficlight.getPhase(tls_id))
    records.append(
        {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "veh_id": veh_id,
            "simulation_time": simulation_time,
            "intersection_id": tls_id,
            "conflicting_lane_ids": json.dumps(lanes),
            "queue_lengths": json.dumps(queue_lengths),
            "mean_speeds": json.dumps(mean_speeds),
            "current_signal_phase": phase,
        }
    )


def write_csv(records: List[Dict[str, object]], columns: Sequence[str], path: Path) -> None:
    df = pd.DataFrame(records, columns=columns)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    sumocfg = args.sumocfg
    if not sumocfg.exists():
        raise FileNotFoundError(f"SUMO configuration not found: {sumocfg}")

    run_id = uuid.uuid4().hex
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    eta_dir = (args.outdir / "eta").resolve()
    conflict_dir = (args.outdir / "conflicts").resolve()
    eta_dir.mkdir(parents=True, exist_ok=True)
    conflict_dir.mkdir(parents=True, exist_ok=True)

    eta_records: List[Dict[str, object]] = []
    conflict_records: List[Dict[str, object]] = []
    snapshots_seen: set[Tuple[str, str]] = set()
    prev_speeds: Dict[str, float] = {}

    try:
        start_traci(sumocfg, args.gui)
        lane_to_tls = build_lane_tls_map()
        step_length = max(traci.simulation.getDeltaT() / 1000.0, 1e-3)
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            sim_time = float(traci.simulation.getTime())
            for veh_id in traci.vehicle.getIDList():
                if not is_emergency_vehicle(veh_id, args.ev_id_prefix, args.ev_type_hint):
                    continue
                speed = float(traci.vehicle.getSpeed(veh_id))
                try:
                    acceleration = float(traci.vehicle.getAcceleration(veh_id))
                except traci.TraCIException:
                    prev_speed = prev_speeds.get(veh_id, speed)
                    acceleration = (speed - prev_speed) / step_length
                prev_speeds[veh_id] = speed

                distance, next_tls_id = estimate_distance_to_tls(veh_id, lane_to_tls)
                eta_records.append(
                    {
                        "run_id": run_id,
                        "scenario_id": args.scenario_id,
                        "veh_id": veh_id,
                        "simulation_time": sim_time,
                        "ev_speed": speed,
                        "ev_acceleration": acceleration,
                        "ev_distance_to_intersection": distance,
                        "next_tls_id": next_tls_id or "",
                    }
                )

                if next_tls_id and (veh_id, next_tls_id) not in snapshots_seen and distance <= 300:
                    ev_lanes = ev_lane_ids(veh_id)
                    snapshot_conflict(
                        conflict_records,
                        run_id,
                        args.scenario_id,
                        veh_id,
                        sim_time,
                        next_tls_id,
                        ev_lanes,
                    )
                    snapshots_seen.add((veh_id, next_tls_id))
    finally:
        try:
            is_libsumo = getattr(traci, 'isLibSumo', lambda: False)()
        except Exception:
            is_libsumo = False
        if not is_libsumo:
            try:
                traci.close(False)
            except Exception:
                pass

    eta_file = eta_dir / f"eta_{args.scenario_id}_{timestamp}.csv"
    conflict_file = conflict_dir / f"conflicts_{args.scenario_id}_{timestamp}.csv"
    write_csv(eta_records, ETA_COLUMNS, eta_file)
    write_csv(conflict_records, CONFLICT_COLUMNS, conflict_file)
    print(f"[EVPS] ETA log written to {eta_file}")
    print(f"[EVPS] Conflict log written to {conflict_file}")


if __name__ == "__main__":
    main()
