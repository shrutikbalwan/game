# Virtual Steering Wheel Racing Game

A top-down Pygame racer that can be driven with hand gestures captured from a
webcam. `main.py` detects gestures with MediaPipe and writes a small local
control-state file; `game.py` reads that state and falls back to the keyboard
when hand tracking is unavailable.

## Features

- Three-lane arcade driving with traffic, scoring, collision detection, and restart support
- Webcam steering, acceleration, and braking gestures
- Keyboard fallback for playing without a camera
- A launcher that starts the tracker and game together
- A lightweight JSON bridge between the two processes

## Requirements

- Python 3.9 or newer
- A webcam for gesture controls (optional for keyboard play)
- `pygame`, `opencv-python`, `mediapipe`, and `numpy`

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install pygame opencv-python mediapipe numpy
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install pygame opencv-python mediapipe numpy
```

## Run

Start both applications:

```bash
python launcher.py
```

To play with the keyboard only:

```bash
python game.py
```

To run only the hand tracker:

```bash
python main.py
```

## Controls

| Action | Keyboard | Gesture |
| --- | --- | --- |
| Steer | `A` / `D` | Tilt both hands left / right |
| Accelerate | `W` | Bring both wrists together |
| Brake | `S` | Make a fist |
| Restart after a collision | `R` | - |
| Quit the game | `Q` or close the window | - |

Hold both hands steady during startup calibration. The preview shows the
detected steering angle and active controls.

## Troubleshooting

- If the webcam cannot be opened, close other camera applications and check OS camera permissions.
- If MediaPipe cannot initialize, the tracker provides simulated controls for testing the remaining pipeline.
- If the game does not react to gestures, confirm the tracker says it is calibrated and that both scripts run from the same repository checkout.
- Press `Ctrl+C` in the launcher terminal to stop both child processes.

The bridge file (`_control_bridge.json`) is generated at runtime and removed
during a clean tracker shutdown.
