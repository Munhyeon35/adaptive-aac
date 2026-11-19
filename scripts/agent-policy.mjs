const normalize = (phrase) => phrase.normalize("NFC").trim().replace(/^(저는|나는)\s+/, "")
  .replace(/[.!?。！？]+$/, "").replace(/\s+/g, " ");

const scoreReasons = {
  1: "전달한 문장이 없어요.",
  2: "과제 의도에 맞는 문장으로 확인되지 않았어요.",
  3: "의도의 일부만 전달했어요.",
  4: "핵심 의도를 전달했지만 일부 세부 내용이 빠졌어요.",
  5: "원하는 뜻을 모두 전달했어요.",
};

export function rateIntent(task, delivered) {
  const phrase = normalize(delivered);
  const score = !phrase ? 1 : ([5, 4, 3].find((level) =>
    task.score_phrases[level]?.some((option) => normalize(option) === phrase)) ?? 2);
  return {score, score_reason: scoreReasons[score]};
}

export function seededRandom(seed, key) {
  let state = seed >>> 0;
  for (const char of key) state = Math.imul(state ^ char.codePointAt(0), 16777619);
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value ^= value + Math.imul(value ^ (value >>> 7), 61 | value);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}
