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
        game_proc = subprocess.Popen(
            [sys.executable, "game.py"],
            cwd=r"c:\Users\shrut\Downloads\New folder",
        )
        processes.append(("game", game_proc))

        # Small delay so the game window opens first
        time.sleep(0.5)

        # Start the hand tracking
        tracking_proc = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=r"c:\Users\shrut\Downloads\New folder",
        )
        processes.append(("tracking", tracking_proc))

        # Wait for both processes to complete
        for name, proc in processes:
            proc.wait()

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        for name, proc in processes:
            if proc.poll() is None:
                print(f"Terminating {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    print(f"Force killing {name}...")
                    proc.kill()

    print("All applications closed.")


if __name__ == "__main__":
    main()

