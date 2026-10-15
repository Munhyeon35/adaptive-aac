"""Convert reviewed local exports into next-symbol supervised chat JSONL."""

import argparse
import json
from pathlib import Path

from aac.llm import PREDICTION_INSTRUCTIONS, prediction_input
from aac.vocabulary import BY_ID, validate_ids


def examples(records):
    seen = set()
    for record in records:
        sequence = validate_ids(record["symbols"], allow_empty=False)
        context = record.get("context", "any")
        for index, next_id in enumerate(sequence):
            prefix = sequence[:index]
            key = (tuple(prefix), next_id, context)
            if key in seen:
                continue
            seen.add(key)
            allowed = [i for i in BY_ID if i not in prefix]
            yield {"messages": [
                {"role": "system", "content": PREDICTION_INSTRUCTIONS},
                {"role": "user", "content": json.dumps(prediction_input(prefix, [], allowed, context, 6), ensure_ascii=False)},
                {"role": "assistant", "content": json.dumps({"ids": [next_id]}, ensure_ascii=False)},
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Exported aac-interactions.json or synthetic scenarios.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    records = payload.get("records", payload.get("scenarios"))
    if not isinstance(records, list):
        parser.error("Expected records or scenarios list")
    rows = list(examples(records))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"Wrote {len(rows)} next-symbol examples to {args.output}. Review and split by participant/session before training.")


if __name__ == "__main__":
    main()
