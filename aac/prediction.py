"""Deterministic, history-aware ranking and bounded model assistance."""

from .vocabulary import BY_ID, STARTERS, VOCABULARY


def rank_symbols(selected, history=(), category=None, context="any"):
    scores = {s["id"]: 0.0 for s in VOCABULARY if s["id"] not in selected}
    for index, symbol_id in enumerate(STARTERS):
        if symbol_id in scores:
            scores[symbol_id] += (len(STARTERS) - index) * (2 if not selected else 0.12)
    for position, previous_id in enumerate(selected):
        previous = BY_ID[previous_id]
        weight = 1 + position / max(len(selected), 1)
        for candidate_id in scores:
            candidate = BY_ID[candidate_id]
            if candidate_id in previous["tags"]:
                scores[candidate_id] += 8 * weight
            if candidate["category"] in previous["tags"]:
                scores[candidate_id] += 5 * weight
            if previous_id in candidate["tags"]:
                scores[candidate_id] += 4 * weight
            if set(previous["tags"]) & set(candidate["tags"]):
                scores[candidate_id] += weight
    # Only user-confirmed sequences contribute to personalisation.
    for entry in history[:100]:
        sequence = entry["symbols"]
        for candidate_id in scores:
            if candidate_id in sequence:
                scores[candidate_id] += 0.15
        if sequence[:len(selected)] == selected and len(sequence) > len(selected):
            next_id = sequence[len(selected)]
            if next_id in scores:
                scores[next_id] += 3
    context_categories = {"meal": {"food", "drink"}, "home": {"object", "place"}, "school": {"object"}}
    for symbol_id in scores:
        if BY_ID[symbol_id]["category"] in context_categories.get(context, set()):
            scores[symbol_id] += 1
    # Stable tie order keeps motor location predictable between identical requests.
    return sorted((i for i in scores if not category or BY_ID[i]["category"] == category),
                  key=lambda i: -scores[i])


def recommend(selected, history=(), category=None, context="any", excluded=(), limit=6, provider=None):
    ranked = rank_symbols(selected, history, category, context)
    candidates = [i for i in ranked if i not in excluded]
    source, reason = "local", None
    if provider and candidates:
        try:
            proposed = provider.recommend(selected, history, candidates, context, limit)
            if (not isinstance(proposed, list) or not proposed or len(proposed) > limit
                    or any(not isinstance(i, str) or i not in candidates for i in proposed)
                    or len(proposed) != len(set(proposed))):
                raise ValueError("Invalid model recommendations")
            candidates = proposed + [i for i in candidates if i not in proposed]
            source = getattr(provider, "source", "llm")
        except (ValueError, OSError, TimeoutError, KeyError, TypeError):
            source, reason = "fallback", "추천 연결이 잠시 어려워 기본 추천을 보여드려요."
    return {"symbols": [BY_ID[i] for i in candidates[:limit]], "source": source,
            "notice": reason, "remaining": max(0, len(candidates) - limit)}
