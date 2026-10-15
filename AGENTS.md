# Adaptive AAC

Local research prototype based on the linked capstone proposal. Python 3.11+ standard-library backend, SQLite, and browser-native HTML/CSS/JavaScript. No runtime packages or Node build are required.

- Run: `python3 -m aac`; tests: `python3 -m unittest discover -s tests -v` (also pytest-compatible); evaluation: `python3 -m scripts.evaluate`.
- Keep all vocabulary IDs stable. Validate all model output against the vocabulary and use deterministic fallback on every provider error.
- Preserve user control: never auto-select, auto-speak, or record an unconfirmed message. Always keep the full vocabulary accessible.
- Cloud recommendations are opt-in per local profile. Never expose keys in frontend responses or log messages/history.
- Personal history belongs in ignored `var/`, not fixtures. Synthetic evaluation data must be labeled as synthetic.
- Python uses four-space indentation; JavaScript uses two. Test domain behavior and API failure paths; browser tests cover core communication and accessibility interactions.
