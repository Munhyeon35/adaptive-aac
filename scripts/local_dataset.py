"""Small synthetic/reviewed-data splits with no identical prompts across splits."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

from aac.local_model import local_messages
from aac.prediction import model_candidates
from aac.vocabulary import validate_ids


def build_examples(records):
    grouped = {}
    for record in records:
        sequence = validate_ids(record["symbols"], allow_empty=False)
        context = record.get("context", "any")
        if context not in ("any", "home", "school", "meal"):
            raise ValueError("Invalid context")
        for index, next_id in enumerate(sequence):
            targets = grouped.setdefault((tuple(sequence[:index]), context), [])
            if next_id not in targets:
                targets.append(next_id)
    for (selected, context), targets in grouped.items():
        # Same bounded allowed list as serving. Targets are always included so reviewed
        # Mulberry choices remain learnable; local_messages sorts IDs, so no position leak.
        targets = targets[:6]
        allowed = model_candidates(list(selected), context=context, required=targets)
        messages = local_messages(list(selected), [], allowed, context)
        # A partial intention can have several correct next choices.
        yield {"messages": messages + [{"role": "assistant", "content": json.dumps({"ids": targets})}]}


def prompt_key(example):
    return json.dumps(example["messages"][:-1], ensure_ascii=False, sort_keys=True)


def prepare_splits(input_path, output_dir, seed=42):
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    records = payload.get("records", payload.get("scenarios"))
    if not isinstance(records, list):
        raise ValueError("Expected records or scenarios")
    families = defaultdict(list)
    for record in records:
        sequence = validate_ids(record["symbols"], allow_empty=False)
        # The same intention with more/negation and different contexts stays together.
        families[tuple(sequence[:2])].append(record)
    keys = sorted(families)
    if len(keys) < 6:
        raise ValueError("At least six distinct intention families are required for train/valid/test splits")
    random.Random(seed).shuffle(keys)
    holdout = max(1, len(keys) // 5)
    assignments = {"train": keys[2 * holdout:], "valid": keys[holdout:2 * holdout], "test": keys[:holdout]}
    seen, outputs, dropped = set(), {}, {}
    for split, family_keys in assignments.items():
        examples = list(build_examples([record for key in family_keys for record in families[key]]))
        rows = [row for row in examples if prompt_key(row) not in seen]
        dropped[split] = len(examples) - len(rows)
        if not rows:
            raise ValueError(f"No independent {split} prompts remain; provide more diverse records")
        outputs[split] = rows
        seen.update(prompt_key(row) for row in rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, rows in outputs.items():
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        (output / f"{split}.jsonl").write_text(content, encoding="utf-8")
        hashes[split] = hashlib.sha256(content.encode()).hexdigest()
    manifest = {"kind": payload.get("kind", "reviewed-user-export"), "seed": seed,
                "source_records": len(records), "family_counts": {k: len(v) for k, v in assignments.items()},
                "example_counts": {k: len(v) for k, v in outputs.items()}, "shared_prompts_removed": dropped,
                "sha256": hashes,
                "notice": "Development-only split by first two symbols, not a participant-held-out clinical evaluation."}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
