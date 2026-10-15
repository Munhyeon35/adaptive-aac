"""Conservative Korean realization: all selected concepts must remain visible."""

from .vocabulary import BY_ID


def particle(word, consonant, vowel):
    char = ord(word[-1])
    has_final = 0xAC00 <= char <= 0xD7A3 and (char - 0xAC00) % 28 != 0
    return word + (consonant if has_final else vowel)


def sentence_options(selected):
    if not selected:
        raise ValueError("먼저 그림을 선택해 주세요.")
    remaining = list(selected)
    phrases = []
    negative = "not" in remaining
    more = "more" in remaining
    if negative:
        remaining.remove("not")
    if more:
        remaining.remove("more")
    actions = [i for i in remaining if BY_ID[i]["category"] == "action"]
    nouns = [i for i in remaining if BY_ID[i]["category"] in {"food", "drink", "place", "object"}]
    consumed_modifiers = False
    if len(actions) == 1:
        action = actions[0]
        compatible = {"eat": {"food"}, "drink": {"drink"}, "go": {"place"},
                      "want": {"food", "drink", "object"}}.get(action, set())
        if nouns and all(BY_ID[i]["category"] in compatible for i in nouns):
            subject = ", ".join(BY_ID[i]["label"] for i in nouns)
            subject = subject + "에" if action == "go" else particle(subject, "을", "를")
            verb = {"eat": "먹고 싶어요", "drink": "마시고 싶어요", "go": "가고 싶어요", "want": "주세요"}[action]
            if negative:
                verb = {"eat": "먹고 싶지 않아요", "drink": "마시고 싶지 않아요", "go": "가고 싶지 않아요", "want": "원하지 않아요"}[action]
            phrases.append(f"{subject} {'더 ' if more else ''}{verb}.")
            for item in nouns + [action]:
                remaining.remove(item)
            consumed_modifiers = True
        elif not nouns and action in {"eat", "drink", "go", "rest", "play"}:
            verb = BY_ID[action]["label"]
            if negative:
                verb = verb.replace("고 싶어요", "고 싶지 않아요")
            phrases.append(f"{'더 ' if more else ''}{verb}.")
            remaining.remove(action)
            consumed_modifiers = True
    if "hurt" in remaining and any(i in remaining for i in ("head", "stomach")) and not negative:
        parts = [i for i in ("head", "stomach") if i in remaining]
        phrases.append(particle(", ".join(BY_ID[i]["label"] for i in parts), "이", "가") + " 아파요.")
        for item in parts + ["hurt"]:
            remaining.remove(item)
    # Unusual combinations are kept verbatim rather than guessing an intention.
    phrases.extend(BY_ID[i]["label"] + "." for i in remaining)
    if not consumed_modifiers:
        if more:
            phrases.append("더.")
        if negative:
            phrases.append("싫어요 / 안 할래요.")
    sentence = " ".join(phrases)
    return [sentence, "저는 " + sentence] if len(phrases) == 1 and consumed_modifiers else [sentence]


def generate_sentence(selected, provider=None):
    options = sentence_options(selected)
    source = "local"
    sentence = options[0]
    if provider and len(options) > 1:
        try:
            chosen = provider.sentence(selected, options)
            if chosen not in options:
                raise ValueError("Ungrounded sentence")
            sentence, source = chosen, "llm"
        except (ValueError, OSError, TimeoutError, KeyError, TypeError):
            source = "fallback"
    return {"sentence": sentence, "alternatives": options, "source": source}
