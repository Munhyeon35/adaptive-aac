"""Deterministic, history-aware ranking and bounded model assistance."""

from datetime import datetime, timezone
import math

from .vocabulary import BY_ID, STARTERS, VOCABULARY


# Model prompts stay far below the 2048-token local limit even with the Mulberry board.
MODEL_CANDIDATE_LIMIT = 80


# Personal usage. Every term saturates, so one heavily used symbol lifts its own rank without
# flooding unrelated screens; a consistent choice overtakes untouched semantic matches
# (worth ~10) after about four to five confirmations.
USAGE_HALF_LIFE_DAYS = 30
FREQUENCY_WEIGHT = 1.0    # x log1p(uses): mild, also lets used Mulberry symbols into the board
TRANSITION_WEIGHT = 12.0  # x n/(n+2): "what this person picks after the last tile"
START_WEIGHT = 12.0       # x n/(n+2): first tile when nothing is selected
PREFIX_WEIGHT = 6.0       # x n/(n+1): exact multi-tile prefix continuation


def _decay(created_at, now):
    try:
        moment = datetime.fromisoformat(created_at)
    except (TypeError, ValueError):
        return 1.0  # Undated rows (fixtures, simulations) count fully.
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - moment).total_seconds() / 86400)
    return 0.5 ** (age_days / USAGE_HALF_LIFE_DAYS)


def usage_profile(history, now=None):
    """Recency-weighted usage statistics for one profile.

    This dict is the contract for a future DB aggregate: a store may return the same keys
    directly (any key may be omitted). Values are decayed use counts.
      symbols:       {symbol_id: w}                      symbol used in a confirmed message
      starts:        {symbol_id: w}                      first symbol of a message
      transitions:   {previous_id: {next_id: w}}         consecutive pair
      continuations: {(prefix ids...): {next_id: w}}     full prefix of length >= 2
    """
    now = now or datetime.now(timezone.utc)
    usage = {"symbols": {}, "starts": {}, "transitions": {}, "continuations": {}}
    for entry in history[:100]:
        sequence = [i for i in entry["symbols"] if i in BY_ID]
        if not sequence:
            continue
        weight = _decay(entry.get("created_at"), now)
        for symbol_id in set(sequence):
            usage["symbols"][symbol_id] = usage["symbols"].get(symbol_id, 0) + weight
        usage["starts"][sequence[0]] = usage["starts"].get(sequence[0], 0) + weight
        for index in range(1, len(sequence)):
            following = usage["transitions"].setdefault(sequence[index - 1], {})
            following[sequence[index]] = following.get(sequence[index], 0) + weight
            if index >= 2:
                following = usage["continuations"].setdefault(tuple(sequence[:index]), {})
                following[sequence[index]] = following.get(sequence[index], 0) + weight
    return usage


def _apply_usage(scores, selected, usage):
    for symbol_id, uses in usage.get("symbols", {}).items():
        if symbol_id in scores:
            scores[symbol_id] += FREQUENCY_WEIGHT * math.log1p(uses)
    if not selected:
        following, weight, smoothing = usage.get("starts", {}), START_WEIGHT, 2
    else:
        following, weight, smoothing = usage.get("transitions", {}).get(selected[-1], {}), TRANSITION_WEIGHT, 2
    for symbol_id, uses in following.items():
        if symbol_id in scores:
            scores[symbol_id] += weight * uses / (uses + smoothing)
    if len(selected) >= 2:
        for symbol_id, uses in usage.get("continuations", {}).get(tuple(selected), {}).items():
            if symbol_id in scores:
                scores[symbol_id] += PREFIX_WEIGHT * uses / (uses + 1)


def score_symbols(selected, history=(), context="any", usage=None):
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
    _apply_usage(scores, selected, usage if usage is not None else usage_profile(history))
    context_categories = {"meal": {"food", "drink"}, "home": {"object", "place"}, "school": {"object"}}
    for symbol_id in scores:
        if BY_ID[symbol_id]["category"] in context_categories.get(context, set()):
            scores[symbol_id] += 1
    return scores


def _order(scores, category=None):
    # Stable tie order keeps motor location predictable between identical requests.
    return sorted((i for i in scores if not category or BY_ID[i]["category"] == category),
                  key=lambda i: -scores[i])


def rank_symbols(selected, history=(), category=None, context="any", usage=None):
    return _order(score_symbols(selected, history, context, usage), category)


def model_candidates(selected, history=(), category=None, context="any", excluded=(),
                     limit=MODEL_CANDIDATE_LIMIT, required=(), scores=None):
    """Bounded allowed list shared by serving and training prompts.

    Uncategorised Mulberry symbols carry no semantic signal and would otherwise
    fill the prompt in file order ("A", "Advent", ...). They are offered to a
    model only after the user's confirmed history gives them a positive score.
    The full vocabulary stays reachable through the deterministic board.
    """
    scores = score_symbols(selected, history, context) if scores is None else scores
    pool = [i for i in _order(scores, category) if i not in excluded
            and (BY_ID[i]["category"] is not None or scores[i] > 0)][:limit]
    pool += [i for i in required if i in BY_ID and i not in pool and i not in selected]
    return pool


def validate_proposal(proposed, allowed, limit):
    """Serving acceptance rule for model output; benchmarks reuse it so both judge identically."""
    if (not isinstance(proposed, list) or not proposed or len(proposed) > limit
            or any(not isinstance(i, str) or i not in allowed for i in proposed)
            or len(proposed) != len(set(proposed))):
        raise ValueError("Invalid model recommendations")
    return proposed


def recommend(selected, history=(), category=None, context="any", excluded=(), limit=6, provider=None, usage=None):
    scores = score_symbols(selected, history, context, usage)
    candidates = [i for i in _order(scores, category) if i not in excluded]
    source, reason = "local", None
    allowed = model_candidates(selected, history, category, context, excluded, scores=scores) if provider else []
    if allowed:
        try:
            proposed = validate_proposal(provider.recommend(selected, history, allowed, context, limit), allowed, limit)
            candidates = proposed + [i for i in candidates if i not in proposed]
            source = getattr(provider, "source", "llm")
        except (ValueError, OSError, TimeoutError, KeyError, TypeError):
            source, reason = "fallback", "추천 연결이 잠시 어려워 기본 추천을 보여드려요."
    return {"symbols": [BY_ID[i] for i in candidates[:limit]], "source": source,
            "notice": reason, "remaining": max(0, len(candidates) - limit)}
