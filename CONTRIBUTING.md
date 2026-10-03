# Contributing

Thanks for improving the Virtual Steering Wheel Racing Game.

## Local setup

1. Fork or clone the repository.
2. Create a virtual environment with `python -m venv .venv`.
3. Activate it and run `python -m pip install -r requirements.txt`.
4. Use `python game.py` for keyboard-only development, or `python launcher.py`
   when testing the complete camera-to-game path.

## Making a change

- Create a focused branch from `master`.
- Keep generated files such as `__pycache__` and `_control_bridge.json` out of commits.
- Preserve keyboard fallback behavior when changing gesture controls.
- Avoid hardware-specific paths; resolve files relative to the repository.
- Add or update documentation when commands or controls change.

## Validation

Before opening a pull request, run:

```bash
python -m compileall -q .
```

For gameplay changes, also launch `game.py` and verify steering, acceleration,
braking, collision, restart, and quit behavior. Camera-related changes should be
tested both with a working webcam and with the camera unavailable.

## Pull requests

Describe the user-visible behavior, how it was tested, and any hardware or OS
limitations. Keep unrelated refactors in separate pull requests so changes are
easy to review and revert.
