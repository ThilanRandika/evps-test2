"""Utility helpers for importing SUMO Python modules."""
from __future__ import annotations

import os
import sys


class MissingSUMOHomeError(RuntimeError):
    """Raised when SUMO_HOME is not configured."""


def _ensure_tools_on_path() -> str:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise MissingSUMOHomeError(
            "SUMO_HOME environment variable is not set. Please install SUMO and set "
            "SUMO_HOME before running EVPS tooling."
        )

    tools_path = os.path.join(sumo_home, "tools")
    if not os.path.isdir(tools_path):
        raise MissingSUMOHomeError(
            f"SUMO tools directory not found at '{tools_path}'. Ensure SUMO is installed correctly."
        )

    if tools_path not in sys.path:
        sys.path.append(tools_path)
    return tools_path


def ensure_sumo_imports() -> None:
    """Ensure SUMO's Python tools are importable."""

    _ensure_tools_on_path()


# Attempt to make tools importable immediately to fail fast during module import.
try:
    ensure_sumo_imports()
except MissingSUMOHomeError as exc:  # pragma: no cover - executed only when misconfigured
    raise


def check_binary(name: str) -> str:
    """Return the absolute path to a SUMO binary using sumolib.checkBinary."""

    ensure_sumo_imports()
    from sumolib import checkBinary  # type: ignore

    return checkBinary(name)


__all__ = ["ensure_sumo_imports", "check_binary", "MissingSUMOHomeError"]
