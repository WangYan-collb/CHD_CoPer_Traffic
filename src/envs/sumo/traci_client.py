from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil


class SumoUnavailableError(RuntimeError):
    """Raised when SUMO or TraCI is not available on the local machine."""


def load_traci():
    if not os.environ.get("SUMO_HOME"):
        raise SumoUnavailableError("SUMO_HOME is not set; install SUMO and export SUMO_HOME first")
    try:
        return importlib.import_module("traci")
    except ModuleNotFoundError as exc:
        raise SumoUnavailableError("Python package 'traci' is not installed") from exc


def sumo_binary(gui: bool = False) -> str:
    binary_name = "sumo-gui" if gui else "sumo"
    binary = shutil.which(binary_name)
    if binary:
        return binary
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise SumoUnavailableError("SUMO_HOME is not set; install SUMO and export SUMO_HOME first")
    candidate = Path(sumo_home) / "bin" / binary_name
    if os.name == "nt":
        candidate = candidate.with_suffix(".exe")
    if not candidate.exists():
        raise SumoUnavailableError(f"SUMO binary not found: {candidate}")
    return str(candidate)


def build_sumo_command(sumocfg: str | Path, gui: bool = False) -> list[str]:
    return [sumo_binary(gui=gui), "-c", str(sumocfg), "--start", "--quit-on-end"]
