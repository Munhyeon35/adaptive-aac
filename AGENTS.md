# Adaptive AAC

Local research prototype based on the linked capstone proposal. Python 3.11+ standard-library backend, SQLite, and browser-native HTML/CSS/JavaScript. No runtime packages or Node build are required.

- Run from the clone: `python run.py` (or an absolute path to `run.py`); tests: `python -m unittest discover -s tests -v`; evaluation: `python -m scripts.evaluate`.
- Optional local Qwen: `python setup_local.py --serve` selects MLX on Apple Silicon and Transformers elsewhere. CUDA uses QLoRA; CPU uses LoRA. Intel Mac needs Python 3.11/3.12. Keep dependencies in `.venv-mlx/` or `.venv-torch/` and model/data/weights in ignored `var/`. Never report training complete without actual saved adapter weights and successful generation validation.
- Keep subprocess communication portable to Windows (no `select()` on pipes), use UTF-8 for text artifacts, and preserve existing MLX manifests. Run `AAC_TEST_TORCH=1` with the PyTorch environment for the optional tiny-model integration test; CI runs it on each supported OS.
- Keep all vocabulary IDs stable. Validate all model output against the vocabulary and use deterministic fallback on every provider error.
- Preserve user control: never auto-select, auto-speak, or record an unconfirmed message. Always keep the full vocabulary accessible.
- Cloud recommendations are opt-in per local profile. Never expose keys in frontend responses or log messages/history.
- Personal history belongs in ignored `var/`, not fixtures. Synthetic evaluation data must be labeled as synthetic.
- Python uses four-space indentation; JavaScript uses two. Test domain behavior and API failure paths; browser tests cover core communication and accessibility interactions.
