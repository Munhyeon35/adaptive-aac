// Synthetic AAC user agents driving visible controls in the shipping browser app.
import assert from "node:assert/strict";
import {mkdir, readFile, writeFile} from "node:fs/promises";
import {dirname} from "node:path";
import {parseArgs} from "node:util";
import {rateIntent, seededRandom} from "./agent-policy.mjs";
import {withBrowser} from "./browser-harness.mjs";

const board = "#symbols";
const fullBoard = "#all-symbols";
const frame = "new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve(true))))";
const modes = ["adaptive", "static"];

function visibleCards(selector) {
  const grid = document.querySelector(selector);
  return [...grid.querySelectorAll(".symbol-card:not(:disabled)")].flatMap((card) => {
    const bounds = card.getBoundingClientRect();
    const x = Math.round(bounds.x + bounds.width / 2);
    const y = Math.round(bounds.y + bounds.height / 2);
    return card.contains(document.elementFromPoint(x, y))
      ? [{label: card.lastElementChild.textContent.trim(), id: card.dataset.symbol, x, y}]
      : [];
  });
}

function scrollPanel(selector, move) {
  const grid = document.querySelector(selector);
  if (selector === "#all-symbols") {
    const available = grid.scrollHeight - grid.clientHeight - grid.scrollTop > 1;
    if (available && move) grid.scrollTop += Math.floor(grid.clientHeight * 0.7);
    return available;
  }
  const available = grid.getBoundingClientRect().bottom > innerHeight &&
    document.scrollingElement.scrollHeight - innerHeight - scrollY > 1;
  if (available && move) window.scrollBy(0, Math.floor(innerHeight * 0.7));
  return available;
}

class UserAgent {
  constructor(page, persona, task, seed, timeout) {
    this.page = page;
    this.persona = persona;
    this.task = task;
    this.timeout = timeout;
    this.random = seededRandom(seed, `${persona.id}:${task.id}`);
    this.behaviorMs = 0;
    this.cardsInspected = 0;
    this.misTaps = 0;
    this.trace = [];
    this.sources = new Set();
    this.exhausted = false;
  }

  spend(ms) {
    if (this.behaviorMs + ms <= this.persona.budget_ms) {
      this.behaviorMs += ms;
      return true;
    }
    this.exhausted = true;
    return false;
  }

  async pointer(point) {
    await this.page.send("Input.dispatchMouseEvent", {type: "mousePressed", ...point, button: "left", clickCount: 1});
    await this.page.send("Input.dispatchMouseEvent", {type: "mouseReleased", ...point, button: "left", clickCount: 1});
  }

  async control(selector, counted = true) {
    if (counted && !this.spend(this.persona.action_ms)) return false;
    const point = await this.page.evaluate(`(() => {
      const control = document.querySelector(${JSON.stringify(selector)});
      if (!control || control.disabled) return null;
      control.scrollIntoView({block: "nearest"});
      const r = control.getBoundingClientRect();
      const x = Math.round(r.x + r.width / 2), y = Math.round(r.y + r.height / 2);
      return control.contains(document.elementFromPoint(x, y)) ? {x, y} : null;
    })()`);
    if (!point) throw new Error(`Control is unavailable: ${selector}`);
    await this.pointer(point);
    return true;
  }

  async waitForBoard() {
    await this.page.until('document.querySelector("#symbols")?.getAttribute("aria-busy") === "false"', this.timeout);
    this.sources.add(await this.page.evaluate(`(() => {
      const caption = document.querySelector("#source").textContent.split(" · ")[0];
      if (caption === "내 기기 AI 추천") return "local_llm";
      if (caption === "기본 그림판") return "static";
      if (caption === "연결 확인 필요") return "connection_error";
      return document.querySelector("#recommend-notice").textContent ? "fallback" : "rules";
    })()`));
  }

  async scan(selector, target) {
    const name = selector === board ? "board" : "all_vocabulary";
    if (selector === board) {
      await this.page.evaluate('document.querySelector("#symbols").scrollIntoView({block: "start"})');
    }
    const inspected = new Set();
    const panel = JSON.stringify(selector);
    for (;;) {
      await this.page.evaluate(frame);
      const cards = await this.page.evaluate(`(${visibleCards})(${panel})`);
      const labels = [];
      let match;
      for (const card of cards) {
        if (inspected.has(card.id)) continue;
        if (!this.spend(this.persona.inspect_ms_per_card)) break;
        inspected.add(card.id);
        labels.push(card.label);
        this.cardsInspected += 1;
        if (card.label === target) { match = card; break; }
      }
      if (labels.length) this.trace.push({action: "inspect", surface: name, labels});
      if (match) {
        const neighbor = cards.filter((card) => card.id !== match.id)
          .sort((a, b) => Math.hypot(a.x - match.x, a.y - match.y) - Math.hypot(b.x - match.x, b.y - match.y))[0];
        return {match, neighbor};
      }
      if (this.exhausted || !await this.page.evaluate(`(${scrollPanel})(${panel}, false)`)) return null;
      if (!this.spend(this.persona.action_ms)) return null;
      await this.page.evaluate(`(${scrollPanel})(${panel}, true)`);
      this.trace.push({action: "scroll", surface: name});
    }
  }

  async find(target) {
    await this.waitForBoard();
    let cards = await this.scan(board, target);
    if (cards || this.exhausted) return cards;
    if (await this.page.evaluate('!document.querySelector("#refresh").disabled')) {
      if (!await this.control("#refresh")) return null;
      this.trace.push({action: "next_board"});
      await this.waitForBoard();
      cards = await this.scan(board, target);
      if (cards || this.exhausted) return cards;
    }
    if (!await this.control("#all-open")) return null;
    this.trace.push({action: "open_all_vocabulary"});
    return await this.scan(fullBoard, target);
  }

  async closeVocabulary() {
    const open = await this.page.evaluate('document.querySelector("#all-dialog").open');
    return !open || await this.control('[data-close="all-dialog"]');
  }

  async choose(target) {
    const slip = this.random() < this.persona.mis_tap_probability;
    let cards = await this.find(target);
    if (cards && slip && cards.neighbor) {
      if (!this.spend(this.persona.action_ms)) return false;
      await this.pointer({x: cards.neighbor.x, y: cards.neighbor.y});
      this.misTaps += 1;
      this.trace.push({action: "mis_tap", label: cards.neighbor.label});
      if (!await this.closeVocabulary() || !await this.control("#undo")) return false;
      this.trace.push({action: "undo"});
      cards = await this.find(target);
    }
    if (!cards || !this.spend(this.persona.action_ms)) {
      await this.closeVocabulary();
      return false;
    }
    await this.pointer({x: cards.match.x, y: cards.match.y});
    this.trace.push({action: "select", label: target});
    assert.equal(await this.page.evaluate(
      'document.querySelector("#selected .token:last-of-type")?.lastElementChild.previousElementSibling.textContent',
    ), target, "The visible target must be added to the composer.");
    return await this.closeVocabulary();
  }

  async compose() {
    for (const target of this.task.goal_cards) {
      if (!await this.choose(target)) {
        this.trace.push({action: this.exhausted ? "budget_exhausted" : "target_unavailable", label: target});
        break;
      }
    }
    if (this.exhausted || !await this.page.evaluate('!document.querySelector("#generate").disabled')) {
      return {sentence: "", completed_ms: null};
    }
    if (!await this.control("#generate")) return {sentence: "", completed_ms: null};
    const status = await this.page.until(`!document.querySelector("#sentence-panel").hidden ? "ready" :
      !document.querySelector("#error").hidden && "generate_failed"`, this.timeout);
    if (status !== "ready") return {sentence: "", completed_ms: null};
    const preview = await this.page.evaluate('document.querySelector("#sentence").value');
    // The agent can decline a clearly wrong draft. Keyboard writing is disabled in these personas.
    if (!this.spend(this.persona.action_ms) || rateIntent(this.task, preview).score < 3 || !await this.control("#confirm")) {
      const reason = this.exhausted ? "budget_exhausted" : "draft_does_not_match_intent";
      this.trace.push({action: "not_confirmed", reason, draft: preview});
      return {sentence: "", completed_ms: null};
    }
    const sentence = await this.page.until('document.querySelector("#output-dialog").open && document.querySelector("#output-text").textContent');
    return {sentence, completed_ms: Math.round(performance.now() - this.started)};
  }

  async run(mode) {
    await this.page.navigate();
    await this.waitForBoard();
    if (mode === "static") await this.control("#mode-static", false);
    if (this.task.context !== "any") {
      await this.page.evaluate(`(() => {
        const context = document.querySelector("#context");
        context.value = ${JSON.stringify(this.task.context)};
        context.dispatchEvent(new Event("change", {bubbles: true}));
      })()`);
      await this.waitForBoard();
      // Begin the measured trial on the requested context with a fresh composer and exposure counter.
      await this.control(`#mode-${mode}`, false);
    }
    this.sources.clear();
    await this.waitForBoard();
    this.started = performance.now();
    const {sentence, completed_ms} = await this.compose();
    const observed_ms = Math.round(performance.now() - this.started);
    let record = null;
    if (sentence) {
      const saved = await this.page.until(`document.querySelector("#save-status").textContent === "내 이야기에 저장했어요." ? "saved" :
        document.querySelector("#save-status").textContent.startsWith("저장하지 못했어요.") && "save_failed"`);
      if (saved === "saved") {
        // Read persisted metrics after the trial, independently of any agent decision.
        record = await this.page.evaluate(`fetch("/api/history", {headers: {"X-Profile-ID": localStorage.getItem("aac.profile")}})
          .then(async (response) => (await response.json()).messages[0])`);
        assert.equal(record.sentence, sentence);
        assert.equal(record.mode, mode);
        assert.equal(record.context, this.task.context);
        assert.equal(record.cards_seen > 0, true, "The app must persist visible card exposures.");
      }
    }
    return {
      persona: this.persona.id, task: this.task.id, mode, intended_sentence: this.task.intent,
      delivered_sentence: sentence || null, ...rateIntent(this.task, sentence),
      cards_inspected: this.cardsInspected, app_cards_seen: record?.cards_seen ?? null,
      completion_ms: completed_ms, attempt_ms: observed_ms, app_elapsed_ms: record?.elapsed_ms ?? null,
      simulated_behavior_ms: this.behaviorMs, mis_taps: this.misTaps,
      saved: Boolean(record), board_sources: [...this.sources], trace: this.trace,
    };
  }
}

const average = (values) => values.length ? Math.round(100 * values.reduce((a, b) => a + b, 0) / values.length) / 100 : null;

function summarize(results) {
  return Object.fromEntries(modes.flatMap((mode) => {
    const cases = results.filter((result) => result.mode === mode);
    if (!cases.length) return [];
    return [[mode, {
      attempts: cases.length, completed: cases.filter((row) => row.delivered_sentence).length,
      fully_expressed: cases.filter((row) => row.score === 5).length,
      mean_score_out_of_5: average(cases.map((row) => row.score)),
      mean_cards_inspected: average(cases.map((row) => row.cards_inspected)),
      mean_app_cards_seen_saved: average(cases.map((row) => row.app_cards_seen).filter((value) => value !== null)),
      mean_completion_ms_completed: average(cases.map((row) => row.completion_ms).filter((value) => value !== null)),
      mean_simulated_behavior_ms: average(cases.map((row) => row.simulated_behavior_ms)),
    }]];
  }));
}

async function main() {
  const {values} = parseArgs({options: {
    output: {type: "string"}, seed: {type: "string", default: "42"}, persona: {type: "string"},
    task: {type: "string"}, mode: {type: "string"}, "local-manifest": {type: "string"},
    help: {type: "boolean"},
  }});
  if (values.help) {
    console.log("CHROMIUM_PATH=/path/to/chrome node scripts/evaluate-agent.mjs [--output FILE] [--seed INT] [--persona ID] [--task ID] [--mode adaptive|static] [--local-manifest FILE]");
    return;
  }
  const seed = Number(values.seed);
  if (!/^\d+$/.test(values.seed) || !Number.isSafeInteger(seed) || seed > 0xFFFFFFFF) {
    throw new Error("Seed must be a 32-bit unsigned integer.");
  }
  const corpus = JSON.parse(await readFile(new URL("../data/agent-scenarios.json", import.meta.url), "utf8"));
  const personas = corpus.personas.filter(({id}) => !values.persona || id === values.persona);
  const tasks = corpus.tasks.filter(({id}) => !values.task || id === values.task);
  const selectedModes = modes.filter((mode) => !values.mode || mode === values.mode);
  if (!personas.length || !tasks.length || !selectedModes.length) throw new Error("Unknown persona, task, or mode.");
  const localManifest = values["local-manifest"];
  const results = await withBrowser(async (page) => {
    const rows = [];
    for (const persona of personas) {
      for (const task of tasks) {
        for (const mode of selectedModes) {
          const agent = new UserAgent(page, persona, task, seed, localManifest ? 140000 : 10000);
          rows.push(await agent.run(mode));
        }
      }
    }
    assert.deepEqual(page.errors, []);
    return rows;
  }, {localManifest});
  const result = {
    kind: "synthetic-aac-browser-evaluation", seed, engine: localManifest ? "local-model-with-fallback" : "rules",
    viewport: {width: 1365, height: 1000},
    sources: corpus.sources, personas, summary: summarize(results), results,
  };
  const json = JSON.stringify(result, null, 2) + "\n";
  if (values.output) {
    await mkdir(dirname(values.output), {recursive: true});
    await writeFile(values.output, json);
    console.log(JSON.stringify(result.summary, null, 2));
  } else console.log(json);
}

await main();
