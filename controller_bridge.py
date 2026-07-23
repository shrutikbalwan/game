"""Shared bridge between Virtual Steering Wheel and Racing Game.

Uses a memory-mapped file for ultra-low-latency inter-process communication
without requiring window focus.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

BRIDGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_control_bridge.json")


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
    """Write control state to the shared bridge file."""
    data = {
        "steer_left": control.steer_left,
        "steer_right": control.steer_right,
        "accelerate": control.accelerate,
        "brake": control.brake,
        "calibrated": control.calibrated,
        "timestamp": time.time(),
    }
    try:
        with open(BRIDGE_FILE, "w") as f:
            json.dump(data, f)
    except (OSError, IOError):
        pass  # Will retry on next frame


def read_control() -> ControlInput:
    """Read the latest control state from the shared bridge file."""
    try:
        with open(BRIDGE_FILE, "r") as f:
            data = json.load(f)
        return ControlInput(
            steer_left=data.get("steer_left", False),
            steer_right=data.get("steer_right", False),
            accelerate=data.get("accelerate", False),
            brake=data.get("brake", False),
            calibrated=data.get("calibrated", False),
            timestamp=data.get("timestamp", 0.0),
        )
    except (OSError, IOError, json.JSONDecodeError):
        return ControlInput()


def cleanup_bridge() -> None:
    """Remove the bridge file during shutdown."""
    try:
        if os.path.exists(BRIDGE_FILE):
            os.remove(BRIDGE_FILE)
    except (OSError, IOError):
        pass

