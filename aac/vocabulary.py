"""Small, auditable vocabulary shared by prediction, generation, and the interface."""

CATEGORIES = [
    {"id": "action", "label": "하고 싶어요", "icon": "👋"},
    {"id": "food", "label": "먹을 것", "icon": "🍚"},
    {"id": "drink", "label": "마실 것", "icon": "🥛"},
    {"id": "place", "label": "갈 곳", "icon": "🏠"},
    {"id": "feeling", "label": "몸과 마음", "icon": "💛"},
    {"id": "object", "label": "필요한 것", "icon": "🧸"},
    {"id": "social", "label": "내 생각", "icon": "💬"},
]

# id, label, pictogram, category, semantic associations
_ROWS = [
    ("eat", "먹고 싶어요", "🍽️", "action", ["food", "hungry"]),
    ("drink", "마시고 싶어요", "🥤", "action", ["drink", "thirsty"]),
    ("go", "가고 싶어요", "🚶", "action", ["place"]),
    ("want", "주세요", "🤲", "action", ["object", "food", "drink"]),
    ("help", "도와주세요", "🙋", "action", ["feeling", "object"]),
    ("rest", "쉬고 싶어요", "🛋️", "action", ["tired", "home", "bed"]),
    ("rice", "밥", "🍚", "food", ["eat", "hungry"]),
    ("bread", "빵", "🍞", "food", ["eat", "hungry"]),
    ("snack", "과자", "🍪", "food", ["eat", "hungry"]),
    ("apple", "사과", "🍎", "food", ["eat", "hungry"]),
    ("banana", "바나나", "🍌", "food", ["eat", "hungry"]),
    ("noodles", "국수", "🍜", "food", ["eat", "hungry"]),
    ("soup", "국", "🥣", "food", ["eat", "hungry"]),
    ("egg", "달걀", "🥚", "food", ["eat", "hungry"]),
    ("water", "물", "💧", "drink", ["drink", "thirsty"]),
    ("milk", "우유", "🥛", "drink", ["drink", "thirsty"]),
    ("juice", "주스", "🧃", "drink", ["drink", "thirsty"]),
    ("tea", "차", "🍵", "drink", ["drink", "thirsty"]),
    ("bathroom", "화장실", "🚻", "place", ["go"]),
    ("home", "집", "🏠", "place", ["go", "rest"]),
    ("outside", "밖", "🌳", "place", ["go", "play"]),
    ("school", "학교", "🏫", "place", ["go"]),
    ("hospital", "병원", "🏥", "place", ["go", "hurt"]),
    ("bed", "침대", "🛏️", "place", ["go", "rest", "tired"]),
    ("hungry", "배고파요", "😋", "feeling", ["food", "eat"]),
    ("thirsty", "목말라요", "😮", "feeling", ["drink"]),
    ("hurt", "아파요", "🤕", "feeling", ["body", "help"]),
    ("tired", "피곤해요", "😴", "feeling", ["rest", "bed"]),
    ("happy", "기뻐요", "😊", "feeling", []),
    ("sad", "슬퍼요", "😢", "feeling", ["help"]),
    ("hot", "더워요", "🥵", "feeling", ["water", "fan"]),
    ("cold", "추워요", "🥶", "feeling", ["blanket"]),
    ("stomach", "배", "🫃", "feeling", ["body", "hurt"]),
    ("head", "머리", "🧑", "feeling", ["body", "hurt"]),
    ("toy", "장난감", "🧸", "object", ["want", "play"]),
    ("book", "책", "📖", "object", ["want"]),
    ("phone", "휴대폰", "📱", "object", ["want"]),
    ("blanket", "담요", "🧣", "object", ["want", "cold"]),
    ("tissue", "휴지", "🧻", "object", ["want", "bathroom"]),
    ("fan", "선풍기", "🌀", "object", ["want", "hot"]),
    ("yes", "네", "👍", "social", []),
    ("no", "아니요", "👎", "social", []),
    ("not", "싫어요 / 안 할래요", "🙅", "social", []),
    ("more", "더", "➕", "social", ["food", "drink"]),
    ("thanks", "고마워요", "🫶", "social", []),
    ("stop", "그만할래요", "✋", "social", []),
    ("play", "놀고 싶어요", "🪁", "action", ["toy", "outside"]),
]

VOCABULARY = [dict(id=i, label=l, icon=e, category=c, tags=t) for i, l, e, c, t in _ROWS]
BY_ID = {symbol["id"]: symbol for symbol in VOCABULARY}
CATEGORY_IDS = {category["id"] for category in CATEGORIES}
STARTERS = ["eat", "drink", "go", "want", "hurt", "help"]


def validate_ids(value, *, maximum=12, allow_empty=True):
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"그림은 최대 {maximum}개까지 선택할 수 있어요.")
    if not allow_empty and not value:
        raise ValueError("먼저 그림을 선택해 주세요.")
    if any(not isinstance(item, str) or item not in BY_ID for item in value):
        raise ValueError("등록되지 않은 그림입니다.")
    if len(value) != len(set(value)):
        raise ValueError("같은 그림을 두 번 선택할 수 없어요.")
    return value
