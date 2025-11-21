"""CLI to generate SUMO scenarios for different congestion tiers."""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from src.utils.sumo_paths import check_binary, ensure_sumo_imports

ensure_sumo_imports()
from sumolib import net  # type: ignore

DEFAULT_VPH: Dict[str, int] = {"low": 300, "medium": 900, "high": 1800}


def parse_net_size(value: str) -> Tuple[int, int]:
    try:
        x_str, y_str = value.lower().split("x")
        return int(x_str), int(y_str)
    except Exception as exc:  # pragma: no cover - argument parsing safeguard
        raise argparse.ArgumentTypeError("--net-size must look like '3x3'") from exc


def run_command(cmd: Sequence[str]) -> None:
    subprocess.run(list(cmd), check=True)


def generate_network(net_file: Path, size: Tuple[int, int], edge_length: float) -> None:
    net_file.parent.mkdir(parents=True, exist_ok=True)
    nx, ny = size
    cmd = [
        check_binary("netgenerate"),
        "--grid",
        "--grid.x-number",
        str(nx),
        "--grid.y-number",
        str(ny),
        "--grid.length",
        str(edge_length),
        "--default.lanenumber",
        "2",
        "--default.speed",
        "13.9",
        "--tls.guess",
        "true",
        "--output",
        str(net_file),
    ]
    run_command(cmd)


def generate_routes(net_file: Path, route_file: Path, seed: int, vph: int) -> None:
    period = 3600.0 / float(vph)
    random_trips = Path(os.environ["SUMO_HOME"]) / "tools" / "randomTrips.py"
    cmd = [
        sys.executable,
        str(random_trips),
        "-n",
        str(net_file),
        "-r",
        str(route_file),
        "--seed",
        str(seed),
        "--period",
        f"{period:.4f}",
        "--binomial",
        "1",
        "--prefix",
        "veh",
        "--trip-attributes",
        "departLane=\"best\" departSpeed=\"max\" departPos=\"base\"",
    ]
    run_command(cmd)


def build_ev_route(net_file: Path, seed: int) -> List[str]:
    """
    Build a deterministic EV route using SUMO's shortest path.
    Returns a list of edge IDs.
    """
    network = net.readNet(str(net_file))
    rng = random.Random(seed)

    # Choose valid candidate edges (exclude internal)
    candidate_edges = [
        edge for edge in network.getEdges()
        if edge.getFunction() not in {"internal"}
    ]

    if len(candidate_edges) < 2:
        raise RuntimeError("Network is too small to build an emergency route")

    # Random but deterministic start + end edges
    start_edge = rng.choice(candidate_edges)
    end_edge = rng.choice([e for e in candidate_edges if e.getID() != start_edge.getID()])

    # --- FIXED: Correct unpacking ---
    path_edges, path_length = network.getShortestPath(start_edge, end_edge)

    # Ensure we have a valid path
    if not path_edges:
        raise RuntimeError(
            f"Shortest path failed between {start_edge.getID()} and {end_edge.getID()}"
        )

    # Convert edge objects → edge IDs
    route_edge_ids = [edge.getID() for edge in path_edges]

    # Safety corrections (should rarely trigger)
    if route_edge_ids[0] != start_edge.getID():
        route_edge_ids.insert(0, start_edge.getID())

    if route_edge_ids[-1] != end_edge.getID():
        route_edge_ids.append(end_edge.getID())

    return route_edge_ids



def inject_emergency_vehicle(route_file: Path, net_file: Path, seed: int, ev_prefix: str) -> None:
    edges = build_ev_route(net_file, seed)
    ev_route_id = f"{ev_prefix}_route"
    ev_vehicle_id = f"{ev_prefix}_{seed}"
    route_definition = f"    <route id=\"{ev_route_id}\" edges=\"{' '.join(edges)}\"/>\n"
    vehicle_definition = (
        f"    <vehicle id=\"{ev_vehicle_id}\" type=\"emergency\" "
        f"route=\"{ev_route_id}\" depart=\"0\" departLane=\"best\" departSpeed=\"max\"/>\n"
    )
    emergency_type = (
        "    <vType id=\"emergency\" accel=\"2.6\" decel=\"4.5\" sigma=\"0.5\" "
        "length=\"6.5\" maxSpeed=\"33.33\" color=\"1,0,0\"/>\n"
    )

    contents = route_file.read_text()
    if "</routes>" not in contents:
        raise RuntimeError("Route file missing </routes> closing tag")
    injection = emergency_type + route_definition + vehicle_definition
    contents = contents.replace("</routes>", f"{injection}</routes>")
    route_file.write_text(contents)


def write_sumocfg(cfg_file: Path, net_file: Path, route_file: Path) -> None:
    cfg_file.write_text(
        """
<configuration>
    <input>
        <net-file value="network.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
</configuration>
""".strip()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate congestion-specific SUMO scenarios")
    parser.add_argument("--level", choices=sorted(DEFAULT_VPH.keys()), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--net-size", type=parse_net_size, default=parse_net_size("3x3"))
    parser.add_argument("--edge-length", type=float, default=150.0)
    parser.add_argument("--vph", type=int, help="Vehicle-per-hour demand override")
    parser.add_argument("--ev", dest="ev", action="store_true", default=True, help="Inject emergency vehicle")
    parser.add_argument("--no-ev", dest="ev", action="store_false", help="Disable emergency vehicle injection")
    parser.add_argument("--ev-prefix", default="ev", help="Prefix for emergency vehicle id")
    parser.add_argument("--outdir", type=Path, default=Path("data/scenarios"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    level_dir = args.outdir / args.level
    level_dir.mkdir(parents=True, exist_ok=True)
    net_file = level_dir / "network.net.xml"
    route_file = level_dir / "routes.rou.xml"
    cfg_file = level_dir / "scenario.sumocfg"

    vph = args.vph or DEFAULT_VPH[args.level]
    print(f"[EVPS] Generating network for {args.level} congestion (VPH={vph})...")
    generate_network(net_file, args.net_size, args.edge_length)
    print("[EVPS] Generating routes via randomTrips...")
    generate_routes(net_file, route_file, args.seed, vph)
    if args.ev:
        print("[EVPS] Injecting deterministic emergency vehicle route...")
        inject_emergency_vehicle(route_file, net_file, args.seed, args.ev_prefix)
    write_sumocfg(cfg_file, net_file, route_file)
    print(f"[EVPS] Scenario assets written to {level_dir}")


if __name__ == "__main__":
    main()
