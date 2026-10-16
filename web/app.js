"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  vocabulary: [], categories: [], byId: new Map(), profile: null, cloudAvailable: false, localModelAvailable: false,
  selected: [], recommendations: [], excluded: [], category: null, context: "any",
  mode: "adaptive", page: 0, clicks: 0, started: null, revision: 0, request: 0,
  messageId: crypto.randomUUID(), saved: false, generating: false, confirming: false,
};

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function announce(text) { $("status").textContent = text; }
function showError(text = "") { $("error").textContent = text; $("error").hidden = !text; }
function stored(key) { try { return localStorage.getItem(key); } catch { return null; } }
function persist(key, value) { try { localStorage.setItem(key, value); } catch { showError("브라우저 저장 공간을 사용할 수 없어 새로 열면 사용자 설정이 초기화돼요."); } }

async function api(path, body) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), path === "/api/recommend" ? 35000 : 12000);
  try {
    const response = await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      headers: {"Content-Type": "application/json", "X-Profile-ID": state.profile?.id || ""},
      body: body === undefined ? undefined : JSON.stringify(body), signal: controller.signal,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "요청을 처리하지 못했어요.");
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("연결이 오래 걸리고 있어요. 다시 시도해 주세요.");
    if (error instanceof TypeError) throw new Error("서버에 연결할 수 없어요. 서버가 실행 중인지 확인해 주세요.");
    throw error;
  } finally { clearTimeout(timeout); }
}

function touch() {
  if (state.started === null) state.started = performance.now();
  state.clicks += 1;
}

function invalidateSentence() {
  state.revision += 1;
  state.saved = false;
  state.messageId = crypto.randomUUID();
  $("sentence-panel").hidden = true;
  $("sentence").value = "";
}

function reset() {
  state.selected = []; state.excluded = []; state.page = 0; state.category = null;
  state.clicks = 0; state.started = null;
  invalidateSentence(); renderCategories(); renderSelected(); updateBoard(); showError();
  if ("speechSynthesis" in window) speechSynthesis.cancel();
}

function choose(id) {
  if (state.selected.includes(id)) return;
  if (state.selected.length >= 12) { showError("그림은 12개까지 선택할 수 있어요. 문장을 전달한 뒤 새 이야기를 시작해 주세요."); return; }
  touch(); state.selected.push(id); state.excluded = []; state.page = 0;
  invalidateSentence(); renderSelected(); updateBoard(); renderAll(); showError();
  announce(`${state.byId.get(id).label} 선택. ${state.selected.length}개의 그림을 골랐어요.`);
}

function remove(id) {
  touch(); state.selected = state.selected.filter((item) => item !== id);
  state.excluded = []; state.page = 0;
  invalidateSentence(); renderSelected(); updateBoard(); renderAll();
  announce("그림을 취소했어요.");
}

function symbolButton(symbol, all = false) {
  const button = node("button", "symbol-card");
  button.dataset.category = symbol.category;
  button.dataset.symbol = symbol.id;
  button.setAttribute("aria-label", `${symbol.label}${state.selected.includes(symbol.id) ? " (선택됨)" : " 선택"}`);
  const icon = node("span", "symbol-icon", symbol.icon); icon.setAttribute("aria-hidden", "true");
  button.append(icon, node("span", "", symbol.label));
  if (all) { button.setAttribute("aria-pressed", String(state.selected.includes(symbol.id))); button.disabled = state.selected.includes(symbol.id); }
  button.addEventListener("click", () => choose(symbol.id));
  return button;
}

function renderCategories() {
  const categories = [{id: null, label: "모두"}, ...state.categories];
  $("categories").replaceChildren(...categories.map((category) => {
    const button = node("button", "category", category.label);
    button.setAttribute("aria-pressed", String(category.id === state.category));
    button.addEventListener("click", () => {
      touch(); state.category = category.id; state.excluded = []; state.page = 0;
      renderCategories(); updateBoard();
      [...$("categories").children].find((b) => b.textContent === category.label)?.focus();
    });
    return button;
  }));
}

function renderSelected() {
  if (state.selected.length) {
    $("selected").replaceChildren(...state.selected.map((id) => {
      const symbol = state.byId.get(id), button = node("button", "token");
      button.setAttribute("aria-label", `${symbol.label} 선택 취소`);
      button.append(node("span", "", symbol.icon), node("span", "", symbol.label), node("span", "remove", "✕"));
      button.addEventListener("click", () => remove(id));
      return button;
    }));
  } else {
    const empty = node("div", "empty-selection");
    empty.append(node("span", "", "💬"), node("p", "", "나의 이야기가 여기에 모여요."));
    $("selected").replaceChildren(empty);
  }
  $("selection-count").textContent = `${state.selected.length} / 12개`;
  $("undo").disabled = $("clear").disabled = state.selected.length === 0;
  $("generate").disabled = state.selected.length === 0 || state.generating;
  for (const mode of ["adaptive", "static"]) {
    $(`mode-${mode}`).disabled = state.selected.length > 0;
    $(`mode-${mode}`).title = state.selected.length ? "새 이야기를 시작하면 방식을 바꿀 수 있어요." : "";
  }
}

function renderGrid(symbols) {
  const focusedId = $("symbols").contains(document.activeElement) ? document.activeElement.dataset.symbol : null;
  $("symbols").replaceChildren(...symbols.map((symbol) => symbolButton(symbol, state.mode === "static")));
  if (!symbols.length) $("symbols").append(node("p", "empty-grid", "이 분류의 그림을 모두 봤어요.\n다른 분류나 전체 그림을 선택해 보세요."));
  $("symbols").setAttribute("aria-busy", "false");
  if (focusedId) {
    const same = [...$("symbols").children].find((button) => button.dataset.symbol === focusedId);
    (same || $("symbols").querySelector("button") || $("all-open")).focus({preventScroll: true});
  }
}

async function updateBoard() {
  const request = ++state.request;
  $("grid-title").textContent = state.selected.length ? "다음으로 어떤 말을 더할까요?" : "이런 말부터 시작해 볼까요?";
  $("recommend-notice").textContent = "";
  if (state.mode === "static") {
    const available = state.vocabulary.filter((s) => !state.category || s.category === state.category);
    const pages = Math.max(1, Math.ceil(available.length / 6));
    state.page %= pages;
    renderGrid(available.slice(state.page * 6, (state.page + 1) * 6));
    $("source").textContent = `기본 그림판 · ${state.page + 1}/${pages}`;
    $("refresh").textContent = "다음 그림 →"; $("refresh").disabled = pages <= 1;
    return;
  }
  $("symbols").setAttribute("aria-busy", "true");
  $("refresh").disabled = true;
  $("source").textContent = "추천 준비 중…";
  try {
    const result = await api("/api/recommend", {selected: [...state.selected], excluded: [...state.excluded], category: state.category, context: state.context});
    if (request !== state.request) return;
    state.recommendations = result.symbols.map((s) => s.id);
    renderGrid(result.symbols);
    $("source").textContent = result.source === "local_llm" ? "내 기기 AI 추천" : result.source === "llm" ? "클라우드 AI 추천" : "기본 맞춤 추천";
    $("recommend-notice").textContent = result.notice || "";
    $("refresh").textContent = result.remaining ? "↻ 다른 그림 추천" : "↻ 첫 추천 다시 보기";
    $("refresh").disabled = false;
  } catch (error) {
    if (request !== state.request) return;
    $("symbols").setAttribute("aria-busy", "false");
    $("source").textContent = "연결 확인 필요";
    $("recommend-notice").textContent = `${error.message} 전체 그림은 계속 사용할 수 있어요.`;
    $("refresh").disabled = false; $("refresh").textContent = "↻ 추천 다시 시도";
  }
}

function renderAll() {
  if (!$("all-dialog").open) return;
  const query = $("search").value.trim().toLocaleLowerCase();
  const symbols = state.vocabulary.filter((s) => s.label.includes(query) || s.id.includes(query));
  const focusId = document.activeElement?.dataset.symbol;
  $("all-symbols").replaceChildren(...symbols.map((s) => symbolButton(s, true)));
  if (!symbols.length) $("all-symbols").append(node("p", "empty-grid", "맞는 그림이 없어요. 다른 이름으로 찾아보세요."));
  $("all-count").textContent = `${symbols.length}개의 그림 · 선택한 뒤 닫기를 눌러 이야기를 이어가세요.`;
  if (focusId) {
    const buttons = [...$("all-symbols").children];
    const next = buttons.find((b) => !b.disabled && b.dataset.symbol === focusId) || buttons.find((b) => !b.disabled);
    (next || $("search")).focus({preventScroll: true});
  }
}

async function generate() {
  if (!state.selected.length || state.generating) return;
  state.generating = true; const revision = state.revision;
  $("generate").disabled = true; $("generate").textContent = "문장을 만들고 있어요…"; showError();
  try {
    const result = await api("/api/sentence", {selected: [...state.selected]});
    if (revision !== state.revision) return;
    $("sentence").value = result.sentence; $("sentence-panel").hidden = false;
    $("sentence").focus(); announce("문장을 만들었어요. 내용을 확인해 주세요.");
  } catch (error) { if (revision === state.revision) showError(error.message); }
  finally { state.generating = false; $("generate").textContent = "✦ 문장으로 만들기"; renderSelected(); }
}

function displayOutput(sentence, symbols) {
  $("output-text").textContent = sentence;
  $("output-symbols").textContent = symbols.map((i) => state.byId.get(i)?.icon || "").join(" ");
  $("speech-status").textContent = "";
  $("speak").disabled = !("speechSynthesis" in window);
  if ($("speak").disabled) $("speech-status").textContent = "이 브라우저는 음성 읽기를 지원하지 않아요. 화면의 문장을 보여 주세요.";
  if (!$("output-dialog").open) $("output-dialog").showModal();
}

async function confirmSentence() {
  const sentence = $("sentence").value.trim();
  if (!sentence) { $("sentence").focus(); showError("전달할 문장을 입력해 주세요."); return; }
  if (state.confirming) return;
  displayOutput(sentence, state.selected);
  if (state.saved) { $("save-status").textContent = "내 이야기에 저장했어요."; return; }
  state.confirming = true; $("confirm").disabled = true;
  $("save-status").textContent = "내 이야기에 저장하고 있어요…";
  const revision = state.revision;
  try {
    await api("/api/confirm", {id: state.messageId, selected: [...state.selected], sentence,
      mode: state.mode, context: state.context, selection_count: state.clicks,
      elapsed_ms: Math.min(86400000, Math.round(performance.now() - (state.started ?? performance.now())))});
    if (revision === state.revision) state.saved = true;
    $("save-status").textContent = "내 이야기에 저장했어요.";
  } catch (error) { $("save-status").textContent = `저장하지 못했어요. ${error.message} 이 창을 닫고 전달하기를 다시 눌러 주세요.`; }
  finally { state.confirming = false; $("confirm").disabled = false; }
}

function speak() {
  if (!("speechSynthesis" in window)) return;
  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance($("output-text").textContent);
  utterance.lang = "ko-KR"; utterance.rate = 0.85;
  const koreanVoice = speechSynthesis.getVoices().find((voice) => voice.lang.startsWith("ko"));
  if (koreanVoice) utterance.voice = koreanVoice;
  utterance.onstart = () => { $("speech-status").textContent = "내 이야기를 읽고 있어요."; };
  utterance.onend = () => { $("speech-status").textContent = "이야기를 다 읽었어요."; };
  utterance.onerror = (event) => { $("speech-status").textContent = ["canceled", "interrupted"].includes(event.error) ? "읽기를 멈췄어요." : "음성을 읽지 못했어요. 기기의 한국어 음성 설정을 확인해 주세요."; };
  speechSynthesis.speak(utterance);
}

async function openHistory() {
  $("history-dialog").showModal(); $("history-list").textContent = "이야기를 불러오고 있어요…";
  try {
    const result = await api("/api/history");
    $("history-list").replaceChildren(...result.messages.map((message) => {
      const button = node("button", "history-item");
      button.append(node("p", "", message.sentence), node("small", "", `${new Date(message.created_at).toLocaleString("ko-KR")} · 다시 보여주기`));
      button.addEventListener("click", () => { $("history-dialog").close(); displayOutput(message.sentence, message.symbols); $("save-status").textContent = "이전에 전한 이야기예요."; });
      return button;
    }));
    if (!result.messages.length) $("history-list").append(node("p", "subtle", "아직 저장한 이야기가 없어요. 그림을 골라 첫 이야기를 만들어 보세요."));
    const table = node("table"), head = node("tr");
    for (const title of ["방식", "메시지", "평균 선택", "평균 시간"]) head.append(node("th", "", title));
    table.append(head);
    for (const row of result.stats) {
      const tr = node("tr");
      for (const value of [row.mode === "adaptive" ? "맞춤 추천" : "기본 그림판", row.messages, row.average_selections, `${row.average_seconds}초`]) tr.append(node("td", "", value));
      table.append(tr);
    }
    $("stats").replaceChildren(table);
  } catch (error) { $("history-list").textContent = error.message; }
}

function bind() {
  $("undo").addEventListener("click", () => { if (state.selected.length) remove(state.selected.at(-1)); });
  $("clear").addEventListener("click", reset);
  $("generate").addEventListener("click", generate);
  $("confirm").addEventListener("click", confirmSentence);
  $("sentence").addEventListener("input", () => { state.saved = false; state.messageId = crypto.randomUUID(); state.revision += 1; });
  $("context").addEventListener("change", () => { touch(); state.context = $("context").value; state.excluded = []; updateBoard(); });
  $("refresh").addEventListener("click", () => {
    touch();
    if (state.mode === "static") state.page += 1;
    else {
      state.excluded = [...new Set([...state.excluded, ...state.recommendations])];
      const available = state.vocabulary.filter((s) => !state.selected.includes(s.id) && (!state.category || s.category === state.category));
      if (available.every((s) => state.excluded.includes(s.id))) state.excluded = [];
    }
    updateBoard();
  });
  for (const mode of ["adaptive", "static"]) $(`mode-${mode}`).addEventListener("click", () => {
    if (state.selected.length) return;
    state.mode = mode;
    for (const m of ["adaptive", "static"]) $(`mode-${m}`).setAttribute("aria-pressed", String(m === mode));
    reset();
  });
  $("all-open").addEventListener("click", () => { touch(); $("all-dialog").showModal(); $("search").value = ""; renderAll(); $("search").focus(); });
  $("search").addEventListener("input", renderAll);
  document.querySelectorAll("[data-close]").forEach((button) => button.addEventListener("click", () => $(button.dataset.close).close()));
  $("speak").addEventListener("click", speak);
  $("stop-speech").addEventListener("click", () => { if ("speechSynthesis" in window) speechSynthesis.cancel(); $("speech-status").textContent = "읽기를 멈췄어요."; });
  $("output-dialog").addEventListener("close", () => { if ("speechSynthesis" in window) speechSynthesis.cancel(); });
  $("new-message").addEventListener("click", () => { $("output-dialog").close(); reset(); $("board").focus(); });
  $("history-open").addEventListener("click", openHistory);
  $("settings-open").addEventListener("click", () => {
    $("profile-name").value = state.profile.name; $("cloud-enabled").checked = state.profile.cloud_enabled;
    $("cloud-enabled").disabled = !state.cloudAvailable; $("large-text").checked = document.documentElement.classList.contains("large");
    $("settings-status").textContent = state.localModelAvailable ? "로컬 AI가 설정되어 있어요. 아래 설정은 클라우드 AI 사용 여부예요." : state.cloudAvailable ? "" : "로컬 모델 설치 전에는 기본 추천을 사용해요.";
    $("delete-confirm").hidden = true; $("settings-dialog").showModal();
  });
  $("settings-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      state.profile = await api("/api/profile", {name: $("profile-name").value, cloud_enabled: $("cloud-enabled").checked});
      document.documentElement.classList.toggle("large", $("large-text").checked);
      persist("aac.large", String($("large-text").checked)); $("settings-dialog").close(); updateBoard();
    } catch (error) { $("settings-status").textContent = error.message; }
  });
  $("export").addEventListener("click", async () => {
    try {
      const data = await api("/api/export");
      const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: "application/json"}));
      const anchor = node("a"); anchor.href = url; anchor.download = "aac-interactions.json"; anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { $("settings-status").textContent = error.message; }
  });
  $("delete-history").addEventListener("click", () => { $("delete-confirm").hidden = false; });
  $("delete-cancel").addEventListener("click", () => { $("delete-confirm").hidden = true; });
  $("delete-confirm-yes").addEventListener("click", async () => {
    try { await api("/api/history/clear", {}); $("delete-confirm").hidden = true; $("settings-status").textContent = "이야기 기록을 지웠어요."; updateBoard(); }
    catch (error) { $("settings-status").textContent = error.message; }
  });
}

async function start() {
  try {
    const bootstrap = await api("/api/bootstrap");
    state.vocabulary = bootstrap.vocabulary; state.categories = bootstrap.categories;
    state.byId = new Map(state.vocabulary.map((symbol) => [symbol.id, symbol])); state.cloudAvailable = bootstrap.cloud_available;
    state.localModelAvailable = bootstrap.local_model_available;
    const profileId = stored("aac.profile");
    if (profileId) {
      state.profile = {id: profileId};
      try { state.profile = await api("/api/profile"); }
      catch (error) { if (!error.message.includes("사용자 정보를")) throw error; state.profile = null; }
    }
    if (!state.profile) { state.profile = await api("/api/profiles", {name: "나"}); persist("aac.profile", state.profile.id); }
    document.documentElement.classList.toggle("large", stored("aac.large") === "true");
    bind(); renderCategories(); renderSelected(); await updateBoard();
  } catch (error) { showError(`${error.message} 페이지를 새로 열어 다시 시도해 주세요.`); }
}

start();
