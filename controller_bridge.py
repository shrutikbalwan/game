"""Shared bridge between Virtual Steering Wheel and Racing Game.

Uses a memory-mapped file for ultra-low-latency inter-process communication
without requiring window focus.
"""

import json
import os
import time
from dataclasses import dataclass

BRIDGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_control_bridge.json")
BRIDGE_TEMP_FILE = f"{BRIDGE_FILE}.tmp"


@dataclass
class ControlInput:
    """The current control state detected by the hand tracker."""
    steer_left: bool = False
    steer_right: bool = False
    accelerate: bool = False
    brake: bool = False
    calibrated: bool = False
    timestamp: float = 0.0


def write_control(control: ControlInput) -> None:
    """Atomically write control state to the shared bridge file.

    Replacing a complete temporary file prevents the reader from observing a
    partially written JSON document while both applications are running.
    """
    data = {
        "steer_left": control.steer_left,
        "steer_right": control.steer_right,
        "accelerate": control.accelerate,
        "brake": control.brake,
        "calibrated": control.calibrated,
        "timestamp": time.time(),
    }
    try:
        with open(BRIDGE_TEMP_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(BRIDGE_TEMP_FILE, BRIDGE_FILE)
    except OSError:
        pass  # Will retry on next frame


def _boolean(data: dict, key: str) -> bool:
    """Accept only real JSON booleans, not truthy strings or numbers."""
    value = data.get(key, False)
    return value if isinstance(value, bool) else False


def _timestamp(data: dict) -> float:
    """Return a finite numeric timestamp or the safe default."""
    value = data.get("timestamp", 0.0)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 0.0
    value = float(value)
    return value if value >= 0.0 else 0.0


def read_control() -> ControlInput:
    """Read the latest control state from the shared bridge file."""
    try:
        with open(BRIDGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return ControlInput()
        return ControlInput(
            steer_left=_boolean(data, "steer_left"),
            steer_right=_boolean(data, "steer_right"),
            accelerate=_boolean(data, "accelerate"),
            brake=_boolean(data, "brake"),
            calibrated=_boolean(data, "calibrated"),
            timestamp=_timestamp(data),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return ControlInput()


def cleanup_bridge() -> None:
    """Remove the bridge file during shutdown."""
    try:
        for path in (BRIDGE_FILE, BRIDGE_TEMP_FILE):
            try:
                os.remove(path)
            except FileNotFoundError:
                continue
    except OSError:
        pass

