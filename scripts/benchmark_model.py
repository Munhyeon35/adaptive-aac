"""Model-serving benchmark: output validity, ranking quality, and latency per recommendation engine.

Scope: only the model path the server uses (same bounded allowed list, same worker subprocess,
same acceptance rule). It is NOT the AAC interaction evaluation in scripts/evaluate.py, which
simulates communication cost and belongs to the evaluation role.

Metrics are computed from raw engine output. Invalid output counts as a miss and is never
repaired with the deterministic fallback, so a model cannot gain from the rule engine.
Separately, served_hit_at_k scores the screen the user actually sees: recommend() places the
accepted model IDs first and fills the rest from the deterministic order (or shows only the
deterministic order after an error). Compare raw metrics to judge the model, served metrics to
judge the product.

Run with the backend's environment so the worker can import its framework, e.g.
    .venv-torch/bin/python -m scripts.benchmark_model
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from statistics import mean
import time

from aac.config import ROOT
from aac.prediction import model_candidates, validate_proposal
from aac.vocabulary import validate_ids

ENGINE_ERRORS = (ValueError, OSError, TimeoutError, KeyError, TypeError)
WARMUP_REQUEST = {"selected": [], "context": "any"}


class DeterministicEngine:
    """Baseline: the ranked allowed list itself, i.e. what the server shows without a model."""
    source = "deterministic"

    def recommend(self, _selected, _history, candidates, _context, limit):
        return candidates[:limit]

    def close(self):
        pass


def load_cases(path):
    """Read held-out chat rows produced by scripts.local_dataset (selected, context, targets)."""
    cases = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        messages = json.loads(line)["messages"]
        payload = json.loads(messages[1]["content"])
        targets = json.loads(messages[-1]["content"])["ids"]
        context = payload.get("context", "any")
        if context not in ("any", "home", "school", "meal"):
            raise ValueError(f"Invalid context on line {number}")
        cases.append({"selected": validate_ids(payload["selected"]),
                      "targets": validate_ids(targets, allow_empty=False), "context": context})
    if not cases:
        raise ValueError("No benchmark cases found")
    return cases


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _call(engine, selected, context, limit):
    # Rebuilt with the current serving rule instead of trusting the allowed list stored in the file.
    allowed = model_candidates(selected, context=context)
    started = time.monotonic()
    try:
        ids, error = validate_proposal(engine.recommend(selected, [], allowed, context, limit), allowed, limit), None
    except ENGINE_ERRORS as exc:
        ids, error = [], type(exc).__name__
    elapsed = time.monotonic() - started
    # The provider pauses 10 s after an error to protect the UI. A benchmark measures each
    # request independently, so it lifts that pause instead of recording cascaded failures.
    if error and hasattr(engine, "_retry_at"):
        engine._retry_at = 0
    return allowed, ids, error, elapsed


def run_engine(name, engine, cases, limit=6, ks=(1, 3, 6)):
    _allowed, _ids, warm_error, cold_start = _call(engine, WARMUP_REQUEST["selected"], WARMUP_REQUEST["context"], limit)
    # The warm-up only isolates model-load time. An invalid warm-up answer is a quality result,
    # not an outage, so cases still run; a broken worker shows up as errors on every case.
    summary = {"engine": name, "cold_start_seconds": round(cold_start, 3), "warmup_error": warm_error}
    results, latencies, errors = [], [], {}
    for case in cases:
        allowed, ids, error, elapsed = _call(engine, case["selected"], case["context"], limit)
        latencies.append(elapsed)
        if error:
            errors[error] = errors.get(error, 0) + 1
        targets = set(case["targets"])
        served = (ids + [i for i in allowed if i not in ids])[:limit]
        results.append({"selected": case["selected"], "context": case["context"], "target_ids": case["targets"],
                        "predicted_ids": ids, "valid": error is None, "error": error,
                        "target_in_allowed": bool(targets & set(allowed)),
                        "first_hit_rank": next((rank for rank, i in enumerate(ids, 1) if i in targets), None),
                        "served_ids": served,
                        "served_first_hit_rank": next((rank for rank, i in enumerate(served, 1) if i in targets), None),
                        "seconds": round(elapsed, 3)})
    count = len(results)
    summary.update(
        status="ok", cases=count, limit=limit,
        valid_output_rate=round(sum(r["valid"] for r in results) / count, 4),
        errors=errors,
        **{f"hit_at_{k}": round(sum(1 for r in results if r["first_hit_rank"] and r["first_hit_rank"] <= k) / count, 4)
           for k in ks if k <= limit},
        **{f"served_hit_at_{k}": round(sum(1 for r in results if r["served_first_hit_rank"]
                                           and r["served_first_hit_rank"] <= k) / count, 4)
           for k in ks if k <= limit},
        mrr=round(sum(1 / r["first_hit_rank"] for r in results if r["first_hit_rank"]) / count, 4),
        # Collapse check: an engine that always answers one ID scores well on a skewed split.
        distinct_top1=len({r["predicted_ids"][0] for r in results if r["predicted_ids"]}),
        constant_top1_hit_rate=round(max(sum(i in r["target_ids"] for r in results)
                                         for i in {i for r in results for i in r["target_ids"]}) / count, 4),
        # Upper bound for any engine: a target outside the allowed list cannot be recommended.
        target_in_allowed_rate=round(sum(r["target_in_allowed"] for r in results) / count, 4),
        latency_seconds={"mean": round(mean(latencies), 3), "p50": round(percentile(latencies, 0.5), 3),
                         "p95": round(percentile(latencies, 0.95), 3), "max": round(max(latencies), 3)})
    return summary, results


def build_engines(names, manifest, device):
    from aac.local_model import LocalModelProvider
    from aac.runtime import select_backend
    backend = select_backend(manifest.get("backend", "mlx")) if manifest else None
    for name in names:
        if name == "deterministic":
            yield name, (lambda: DeterministicEngine())
        elif name in {"base", "adapter"}:
            if not manifest:
                raise SystemExit("No var/active-model.json. Run setup_local.py first or use --engines deterministic.")
            if name == "adapter" and not manifest.get("adapter_path"):
                raise SystemExit("The active model has no adapter to benchmark.")
            adapter = manifest.get("adapter_path") if name == "adapter" else None
            yield name, (lambda adapter=adapter: LocalModelProvider(
                manifest["model_path"], adapter, backend=backend, device=device,
                quantization=manifest.get("quantization", "none")))
        else:
            raise SystemExit(f"Unknown engine: {name}")


def print_table(summaries):
    columns = ("engine", "valid_output_rate", "hit_at_1", "hit_at_6", "served_hit_at_3", "served_hit_at_6",
               "distinct_top1", "cold_start_seconds")
    print("  ".join(f"{c:>18}" for c in columns + ("p50_s", "p95_s")))
    for row in summaries:
        if row["status"] != "ok":
            print(f"{row['engine']:>18}  unavailable: {row['error']}")
            continue
        latency = row["latency_seconds"]
        print("  ".join(f"{row.get(c, '-'):>18}" for c in columns) + f"  {latency['p50']:>18}  {latency['p95']:>18}")
    ok = [row for row in summaries if row["status"] == "ok"]
    if ok:
        print(f"Constant-answer baseline hit@1: {ok[0]['constant_top1_hit_rate']} "
              "(an engine near this with distinct_top1=1 has collapsed)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, nargs="+", help="Held-out JSONL file(s) (default: test split of the active run)")
    parser.add_argument("--engines", default="deterministic,base,adapter",
                        help="Comma-separated: deterministic, base (no adapter), adapter")
    parser.add_argument("--limit", type=int, default=6, choices=range(1, 17), metavar="1-16")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=os.environ.get("AAC_LOCAL_DEVICE", "auto"))
    parser.add_argument("--output", type=Path, help="Default: var/benchmarks/model-<UTC time>.json")
    parser.add_argument("--run-dir", type=Path, help="Benchmark a saved run (its model.json) instead of the active model")
    args = parser.parse_args()

    manifest_path = args.run_dir / "model.json" if args.run_dir else ROOT / "var/active-model.json"
    if args.run_dir and not manifest_path.is_file():
        parser.error(f"{manifest_path} is missing; the run did not finish training and validation.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    cases_paths = args.cases or ([Path(manifest["run_dir"]) / "data/test.jsonl"] if manifest and manifest.get("run_dir") else [])
    if not cases_paths or not all(path.is_file() for path in cases_paths):
        parser.error("No held-out cases. Pass --cases path/to/test.jsonl.")
    cases = [case for path in cases_paths for case in load_cases(path)]
    split_manifest = cases_paths[0].parent / "manifest.json"
    split_info = json.loads(split_manifest.read_text(encoding="utf-8")) if split_manifest.is_file() else {}

    summaries, details = [], {}
    for name, factory in build_engines([n.strip() for n in args.engines.split(",") if n.strip()], manifest, args.device):
        print(f"Benchmarking {name} on {len(cases)} cases…", flush=True)
        try:
            engine = factory()
        except (OSError, ValueError) as error:
            summaries.append({"engine": name, "status": "unavailable", "error": str(error)})
            continue
        try:
            summary, results = run_engine(name, engine, cases, args.limit)
        finally:
            engine.close()  # One model in memory at a time.
        summaries.append(summary)
        details[name] = results

    report = {
        "kind": "model-serving-benchmark",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Model path only (validity, ranking, latency). Not the AAC interaction-cost evaluation.",
        "cases_paths": [str(path) for path in cases_paths], "cases": len(cases),
        "data_kind": split_info.get("kind", "unknown"), "data_notice": split_info.get("notice"),
        "caveat": ("setup_local.py also uses this test split to decide activation; "
                   "use an untouched holdout for a final report."),
        "model": {k: manifest.get(k) for k in ("backend", "model_repo", "model_revision", "quantization", "adapter_path")}
        if manifest else None,
        "summaries": summaries, "results": details,
    }
    output = args.output or ROOT / "var/benchmarks" / f"model-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_table(summaries)
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
