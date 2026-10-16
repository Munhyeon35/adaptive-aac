# Adaptive AAC

Local research prototype based on the linked capstone proposal. Python 3.11+ standard-library backend, SQLite, and browser-native HTML/CSS/JavaScript. No runtime packages or Node build are required.

- Run from any directory: `python3 /Users/mh/repository/adaptive-aac/run.py`; tests: `python3 -m unittest discover -s tests -v` (also pytest-compatible); evaluation: `python3 -m scripts.evaluate`.
- Optional local Qwen + QLoRA: `python3 setup_local.py --serve`. Keep dependencies in `.venv-mlx/` and model/data/weights in ignored `var/`. Never report training complete without actual saved adapter weights and successful generation validation.
- Keep all vocabulary IDs stable. Validate all model output against the vocabulary and use deterministic fallback on every provider error.
- Preserve user control: never auto-select, auto-speak, or record an unconfirmed message. Always keep the full vocabulary accessible.
- Cloud recommendations are opt-in per local profile. Never expose keys in frontend responses or log messages/history.
- Personal history belongs in ignored `var/`, not fixtures. Synthetic evaluation data must be labeled as synthetic.
- Python uses four-space indentation; JavaScript uses two. Test domain behavior and API failure paths; browser tests cover core communication and accessibility interactions.
