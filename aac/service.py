"""Domain service behind the HTTP API. Validates domain rules, independent of the web framework."""

import uuid

from .prediction import recommend
from .sentences import generate_sentence
from .vocabulary import CATEGORIES, CATEGORY_IDS, VOCABULARY, validate_ids

CONTEXTS = {"any", "home", "school", "meal"}
MODES = {"adaptive", "static"}


def bounded_int(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("숫자 범위를 확인해 주세요.")
    return value


def name_value(value):
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 30:
        raise ValueError("이름은 1~30자로 입력해 주세요.")
    return value.strip()


def context_value(value):
    if not isinstance(value, str) or value not in CONTEXTS:
        raise ValueError("상황을 확인해 주세요.")
    return value


class AacService:
    def __init__(self, store, provider=None, local_provider=None):
        self.store, self.provider, self.local_provider = store, provider, local_provider

    def bootstrap(self):
        return {"vocabulary": VOCABULARY, "categories": CATEGORIES, "cloud_available": bool(self.provider),
                "local_model_available": bool(self.local_provider),
                "recommendation_timeout_ms": max(35000, int(getattr(self.local_provider, "timeout", 25) * 1000) + 5000)}

    def create_profile(self, name="나"):
        return self.store.create_profile(name_value(name))

    def profile(self, profile_id):
        return self.store.profile(profile_id)

    def update_profile(self, profile_id, name, cloud_enabled):
        self.store.profile(profile_id)
        if type(cloud_enabled) is not bool:
            raise ValueError("추천 설정을 확인해 주세요.")
        if cloud_enabled and not self.provider:
            raise ValueError("서버에 API 키가 설정되어 있지 않아요.")
        return self.store.update_profile(profile_id, name_value(name), cloud_enabled)

    def history(self, profile_id):
        self.store.profile(profile_id)
        return {"messages": self.store.history(profile_id), "stats": self.store.stats(profile_id)}

    def clear_history(self, profile_id):
        self.store.profile(profile_id)
        self.store.clear_history(profile_id)
        return {"cleared": True}

    def export(self, profile_id):
        self.store.profile(profile_id)
        history = self.store.history(profile_id, limit=10000)
        return {"schema_version": 1, "kind": "user-confirmed-aac-interactions",
                "records": [{k: v for k, v in row.items() if k not in {"profile_id", "id"}} for row in history],
                "notice": "직접 수정한 문장도 포함됩니다. 학습에 사용하기 전에 검토하세요."}

    def recommend(self, profile_id, selected=(), category=None, context="any", excluded=()):
        provider = self._provider(profile_id)
        selected = validate_ids(list(selected))
        context = context_value(context)
        if category is not None and (not isinstance(category, str) or category not in CATEGORY_IDS):
            raise ValueError("분류를 확인해 주세요.")
        excluded = validate_ids(list(excluded), maximum=len(VOCABULARY))
        # The first board shows more starters; after a selection a short list keeps choices quick.
        limit = 6 if selected else 16
        return recommend(selected, self.store.history(profile_id), category, context, excluded,
                         limit=limit, provider=provider)

    def sentence(self, profile_id, selected, context="any"):
        provider = self._provider(profile_id)
        selected = validate_ids(list(selected), allow_empty=False)
        context_value(context)
        return generate_sentence(selected, provider)

    def confirm(self, profile_id, message_id, selected, sentence, mode, context, selection_count, elapsed_ms,
                cards_seen=None):
        self.store.profile(profile_id)
        selected = validate_ids(list(selected), allow_empty=False)
        context = context_value(context)
        if not isinstance(sentence, str) or not 1 <= len(sentence.strip()) <= 300:
            raise ValueError("문장은 1~300자로 입력해 주세요.")
        if mode not in MODES:
            raise ValueError("선택 모드를 확인해 주세요.")
        try:
            message_id = str(uuid.UUID(str(message_id)))
        except (ValueError, TypeError, AttributeError):
            raise ValueError("메시지 식별자를 확인해 주세요.") from None
        if cards_seen is not None:
            cards_seen = bounded_int(cards_seen, 0, 1000000)
        return self.store.confirm(profile_id, message_id, selected, sentence.strip(), mode, context,
                                  bounded_int(selection_count, len(selected), 10000),
                                  bounded_int(elapsed_ms, 0, 86400000), cards_seen=cards_seen)

    def _provider(self, profile_id):
        # Cloud is used only after the profile opts in; otherwise the local model (if any) answers.
        profile = self.store.profile(profile_id)
        return self.provider if profile["cloud_enabled"] and self.provider else self.local_provider
