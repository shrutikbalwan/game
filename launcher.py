"""Launcher: runs the Virtual Steering Wheel (hand tracking) and the Racing Game together.

Usage:
    python launcher.py

The hand tracking window and game window will both open. Click on the GAME window
to give it focus, then use your hands in front of the webcam to drive.
Press Ctrl+C in this terminal to stop everything.
"""

import subprocess
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def start_component(script_name: str) -> subprocess.Popen:
    """Start a repository script with the active Python interpreter."""
    script_path = PROJECT_DIR / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"Required component is missing: {script_path}")
    return subprocess.Popen([sys.executable, str(script_path)], cwd=PROJECT_DIR)


def stop_process(name: str, process: subprocess.Popen) -> None:
    """Terminate a child process and escalate if it does not exit promptly."""
    if process.poll() is not None:
        return
    print(f"Terminating {name}...")
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        print(f"Force killing {name}...")
        process.kill()
        process.wait()


def main() -> None:
    print("=" * 60)
    print("  Virtual Steering Wheel + Racing Game Launcher")
    print("=" * 60)
    print()
    print("Both applications will start now.")
    print("1. The hand-tracking camera window will open.")
    print("2. The racing game window will open.")
    print("3. CLICK on the GAME window to give it focus.")
    print("4. Use your hands in front of the webcam to drive!")
    print()
    print("  - Tilt hands left/right  ->  Steer (A/D)")
    print("  - Bring wrists together   ->  Accelerate (W)")
    print("  - Make a fist             ->  Brake (S)")
    print()
    print("Press Ctrl+C in this terminal to stop everything.")
    print("=" * 60)
    print()

    processes = []

    try:
        # Start the racing game first so it's ready
        game_proc = start_component("game.py")
        processes.append(("game", game_proc))

        # Small delay so the game window opens first
        time.sleep(0.5)

        # Start the hand tracking
        tracking_proc = start_component("main.py")
        processes.append(("tracking", tracking_proc))

        # If either component exits, stop the other instead of leaving an
        # orphaned camera or game process running.
        while all(proc.poll() is None for _, proc in processes):
            time.sleep(0.1)

        for name, proc in processes:
            return_code = proc.poll()
            if return_code not in (None, 0):
                print(f"{name.capitalize()} exited with status {return_code}.")

    except KeyboardInterrupt:
        print("\nShutting down...")
    except (FileNotFoundError, OSError) as error:
        print(f"Unable to start applications: {error}")
    finally:
        for name, proc in processes:
            stop_process(name, proc)

    print("All applications closed.")


if __name__ == "__main__":
    main()

