"""Reproducible *simulated* interaction costs; never a clinical efficacy claim."""

import argparse
import json
from pathlib import Path
from statistics import mean

from aac.config import ROOT
from aac.prediction import recommend
from aac.vocabulary import VOCABULARY, validate_ids


def evaluate(scenarios, history=()):
    rows = []
    for scenario in scenarios:
        target = validate_ids(scenario["symbols"], allow_empty=False)
        selected, adaptive, static, hits = [], 0, 0, 0
        for symbol_id in target:
            board = [s["id"] for s in VOCABULARY]
            # Both boards show six tiles. Static starts on page 1 after a selection.
            static += board.index(symbol_id) // 6 + 1
            predictions = recommend(selected, history, context=scenario.get("context", "any"))["symbols"]
            if symbol_id in [s["id"] for s in predictions]:
                adaptive += 1
                hits += 1
            else:
                # Escape to a paginated full board: open, page navigation, select.
                adaptive += 1 + board.index(symbol_id) // 6 + 1
            selected.append(symbol_id)
        rows.append({"name": scenario["name"], "symbols": target,
                     "adaptive_actions": adaptive, "static_actions": static,
                     "top6_hits": hits, "prediction_steps": len(target)})
    if not rows:
        raise ValueError("At least one scenario is required")
    avg_static = mean(r["static_actions"] for r in rows)
    avg_adaptive = mean(r["adaptive_actions"] for r in rows)
    return {"kind": "synthetic-simulation", "engine": "local-deterministic",
            "assumptions": "6 tiles/page; static restarts at page 1; adaptive miss opens full board; excludes sentence confirmation, typing, visual search time. UI full-board scroll is approximated as page actions.",
            "scenarios": len(rows), "top6_hit_rate": round(sum(r["top6_hits"] for r in rows) / sum(r["prediction_steps"] for r in rows), 4),
            "mean_adaptive_actions": round(avg_adaptive, 3), "mean_static_actions": round(avg_static, 3),
            "simulated_reduction_percent": round((1 - avg_adaptive / avg_static) * 100, 2), "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/scenarios.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(json.loads(args.input.read_text())["scenarios"])
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
