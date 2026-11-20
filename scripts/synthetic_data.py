"""Transparent template examples for a smoke-training run, not collected AAC data."""

from aac.vocabulary import VOCABULARY

# Feeling-led chains give the model prompts whose answer is not a modifier. Without them
# most prompts were "[action, item] -> more/not" and adapters collapsed to answering "not".
CHAINS = {
    "hungry": [["hungry", "eat", symbol] for symbol in ("rice", "bread", "noodles", "soup", "egg", "apple")],
    "thirsty": [["thirsty", "drink", symbol] for symbol in ("water", "milk", "juice", "tea")],
    "tired": [["tired", "rest"], ["tired", "go", "bed"], ["tired", "go", "home"]],
    "hot": [["hot", "want", "fan"], ["hot", "drink", "water"]],
    "cold": [["cold", "want", "blanket"], ["cold", "go", "home"]],
    "hurt": [["hurt", "head", "help"], ["hurt", "stomach", "help"], ["hurt", "go", "hospital"]],
    "sad": [["sad", "help"], ["sad", "go", "home"]],
    "play": [["play", "outside"], ["play", "want", "toy"]],
    "bathroom": [["go", "bathroom", "want", "tissue"]],
}
CHAIN_CONTEXTS = {"hungry": ("any", "meal"), "thirsty": ("any", "meal"), "tired": ("any", "home"),
                  "hot": ("any", "school"), "cold": ("any", "home"), "hurt": ("any", "school"),
                  "sad": ("any", "home"), "play": ("home", "school"), "bathroom": ("any", "school")}


def synthetic_records():
    rows = []
    # action, contexts, and the single context that also gets more/negation variants
    settings = {"food": ("eat", ["any", "meal"], "meal", [["more"], ["not"]]),
                "drink": ("drink", ["any", "meal"], "meal", [["more"], ["not"]]),
                "place": ("go", ["any", "home"], "any", [["not"]]),
                "object": ("want", ["any", "school"], "any", [["not"]])}
    for symbol in VOCABULARY:
        if symbol["category"] not in settings:
            continue
        action, contexts, modifier_context, modifiers = settings[symbol["category"]]
        for context in contexts:
            rows.append({"symbols": [action, symbol["id"]], "context": context})
        rows.extend({"symbols": [action, symbol["id"], *modifier], "context": modifier_context} for modifier in modifiers)
    for name, sequences in CHAINS.items():
        for context in CHAIN_CONTEXTS[name]:
            rows.extend({"symbols": sequence, "context": context} for sequence in sequences)
    for context in ("any", "home"):
        rows.extend({"symbols": [symbol], "context": context} for symbol in ("help", "thanks", "yes", "no", "stop"))
    return {"kind": "synthetic-template-smoke-training", "records": rows,
            "notice": "Generated from explicit vocabulary/grammar templates. No real-user records; not clinical or independent efficacy evidence."}
