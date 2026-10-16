"""Transparent template examples for a smoke-training run, not collected AAC data."""

from aac.vocabulary import VOCABULARY


def synthetic_records():
    rows = []
    settings = {"food": ("eat", ["any", "meal"], [[], ["more"], ["not"]]),
                "drink": ("drink", ["any", "meal"], [[], ["more"], ["not"]]),
                "place": ("go", ["any", "home"], [[], ["not"]]),
                "object": ("want", ["any", "school"], [[], ["not"]])}
    for symbol in VOCABULARY:
        if symbol["category"] not in settings:
            continue
        action, contexts, modifiers = settings[symbol["category"]]
        for context in contexts:
            for modifier in modifiers:
                rows.append({"symbols": [action, symbol["id"], *modifier], "context": context})
    for context in ("any", "home"):
        rows.extend({"symbols": ["hurt", part], "context": context} for part in ("head", "stomach"))
        rows.extend({"symbols": sequence, "context": context} for sequence in
                    (["tired", "rest"], ["cold", "want", "blanket"], ["hot", "want", "fan"],
                     ["help"], ["thanks"], ["yes"], ["no"], ["stop"]))
    return {"kind": "synthetic-template-smoke-training", "records": rows,
            "notice": "Generated from explicit vocabulary/grammar templates. No real-user records; not clinical or independent efficacy evidence."}
