from __future__ import annotations

import importlib
import os


class SumoUnavailableError(RuntimeError):
    """Raised when SUMO or TraCI is not available on the local machine."""


def load_traci():
    if not os.environ.get("SUMO_HOME"):
        raise SumoUnavailableError("SUMO_HOME is not set; install SUMO and export SUMO_HOME first")
    try:
        return importlib.import_module("traci")
    except ModuleNotFoundError as exc:
        raise SumoUnavailableError("Python package 'traci' is not installed") from exc
