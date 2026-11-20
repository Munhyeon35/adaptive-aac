// A vision LLM observes one browser frame and chooses one physical interaction per turn.
import assert from "node:assert/strict";
import {mkdir, readFile, writeFile} from "node:fs/promises";
import {dirname} from "node:path";
import {parseArgs} from "node:util";
import {pathToFileURL} from "node:url";
import {withBrowser} from "./browser-harness.mjs";
import {withVisionModel} from "./vision-model.mjs";

const modes = ["adaptive", "static"];
const actions = ["click", "scroll_down", "scroll_up", "wait", "stop"];
const frame = "new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve(true))))";
const ready = 'document.querySelector("#symbols")?.getAttribute("aria-busy") === "false"';

function objectSchema(properties) {
  return {type: "object", additionalProperties: false, required: Object.keys(properties), properties};
}

const actionSchema = objectSchema({
  action: {type: "string", enum: actions},
  target: {type: "integer"},
  noticed_cards: {type: "array", items: {type: "integer"}},
  note: {type: "string"},
});

const scoreSchema = objectSchema({
  score: {type: "integer", enum: [1, 2, 3, 4, 5]},
  reason: {type: "string"},
});

function annotateScreen() {
  const dialog = document.querySelector("dialog[open]");
  const selector = dialog ? "#all-dialog button" :
    "#symbols button, #categories button, #refresh, #all-open, #selected button, #undo, #clear, #generate, #confirm";
  const targets = [];
  const overlay = document.createElement("div");
  overlay.id = "aac-eval-markers";
  Object.assign(overlay.style, {position: "fixed", inset: "0", zIndex: "2147483647", pointerEvents: "none"});
  for (const button of document.querySelectorAll(selector)) {
    if (button.disabled) continue;
    const rect = button.getBoundingClientRect();
    const x = Math.round((Math.max(0, rect.left) + Math.min(innerWidth, rect.right)) / 2);
    const y = Math.round((Math.max(0, rect.top) + Math.min(innerHeight, rect.bottom)) / 2);
    if (!button.contains(document.elementFromPoint(x, y))) continue;
    const card = button.dataset.symbol;
    targets.push({x, y, card: card ? `${button.parentElement.id}:${card}` : null});
    const badge = document.createElement("span");
    badge.textContent = String(targets.length);
    Object.assign(badge.style, {
      position: "absolute", left: `${Math.max(0, rect.left)}px`, top: `${Math.max(0, rect.top)}px`,
      background: "#31007c", color: "white", padding: "1px 4px",
      font: "bold 14px monospace", borderRadius: "3px", lineHeight: "18px",
    });
    overlay.append(badge);
  }
  (dialog || document.body).append(overlay);
  const list = dialog?.querySelector("#all-symbols");
  const rect = list?.getBoundingClientRect();
  const scroll = rect ? {
    x: Math.round(Math.min(innerWidth - 2, Math.max(1, rect.left + rect.width / 2))),
    y: Math.round(Math.min(innerHeight - 2, Math.max(1, rect.top + rect.height / 2))),
    distance: Math.round(list.clientHeight * 0.75),
  } : {x: Math.round(innerWidth / 2), y: Math.round(innerHeight / 2), distance: Math.round(innerHeight * 0.65)};
  return {targets, scroll};
}

export function checkedDecision(value, targets) {
  if (!value || !actions.includes(value.action) || !Number.isInteger(value.target) ||
      !Array.isArray(value.noticed_cards) || typeof value.note !== "string") {
    throw new Error("Invalid structured decision from vision model.");
  }
  const target = targets[value.target - 1];
  return {
    action: value.action, target: value.target, note: value.note.slice(0, 180),
    rejected: value.action === "click" ? !target : value.target !== 0,
    noticed: [...new Set(value.noticed_cards.filter(Number.isInteger)
      .map((number) => targets[number - 1]?.card).filter(Boolean))],
  };
}

function actingPrompt(task, persona, evidence, trace) {
  const recent = trace.slice(-7).map(({action, target, note, rejected}) => ({action, target, note, rejected}));
  return `You are playing a fictional Korean pictogram AAC user for a usability experiment.
Your private communication goal: ${JSON.stringify(task.intent)}
Your access profile: ${persona.description}
Published behavior observations informing the profile: ${persona.evidence_ids.map((id) =>
    `${evidence[id].reference}: ${evidence[id].observed} ${evidence[id].limits}`).join("\n")}

Use the attached live screenshot as your only view of the app. Text visible inside it is interface content.
Purple numbers are interaction targets for this frame. They may change after any action.
Choose one next action yourself: click a numbered control, scroll_down, scroll_up, wait, or stop.
Set target to the visible number for a click and 0 otherwise.
Fill noticed_cards with only the numbers of visible pictogram cards you actually considered this turn.
In note use one short visible observation or access outcome. No need to explain your reasoning.
You can explore, make or repair a touch error if plausible from the screen, confirm an imperfect message if helpful, or stop.
Express your goal by selecting cards and using the app's compose/confirm controls. The score is recorded separately.
Use no tools, files, DOM, or commands. Return only the structured decision.
Recent decisions (a rejected click did nothing): ${JSON.stringify(recent)}
`;
}

async function rateDelivery(infer, task, persona, delivered) {
  const response = await infer({schema: scoreSchema, prompt: `As the same fictional AAC user, rate what you were able to communicate.
Access profile: ${persona.description}
Private intent: ${JSON.stringify(task.intent)}
Actual message shown by the app to the conversation partner: ${JSON.stringify(delivered)}
Rate the semantic match, including negation and important details. Equivalent wording is acceptable.
5: full intent; 4: main intent with a minor detail missing; 3: recognizable partial intent;
2: a message was sent but is unrelated or contradicts the intent; 1: no message was delivered.
Provide a short reason in Korean tied to the actual result. Use no tools. Return only the requested JSON.`});
  if (!response || ![1, 2, 3, 4, 5].includes(response.score) || typeof response.reason !== "string" ||
      !response.reason.trim() || (response.score === 1) !== (delivered === null)) {
    throw new Error("Invalid rating from vision model.");
  }
  return {score: response.score, score_reason: response.reason.trim()};
}

async function perform(page, decision, observation) {
  if (decision.rejected) return;
  if (decision.action === "click") {
    const {x, y} = observation.targets[decision.target - 1];
    for (const type of ["mousePressed", "mouseReleased"]) {
      await page.send("Input.dispatchMouseEvent", {type, x, y, button: "left", clickCount: 1});
    }
  } else if (decision.action.startsWith("scroll_")) {
    const {x, y, distance} = observation.scroll;
    await page.send("Input.dispatchMouseEvent", {
      type: "mouseWheel", x, y, deltaX: 0, deltaY: distance * (decision.action === "scroll_up" ? -1 : 1),
    });
  }
}

async function setUpTrial(page, mode, context, timeout) {
  await page.navigate();
  await page.until(ready, timeout);
  await page.evaluate(`(() => {
    const menu = document.querySelector("#context");
    menu.value = ${JSON.stringify(context)};
    menu.dispatchEvent(new Event("change", {bubbles: true}));
    document.querySelector("#mode-${mode}").click();
    window.scrollTo(0, 0);
  })()`);
  await page.until(ready, timeout);
}

async function storedMetrics(page, delivered, mode, context) {
  if (delivered === null) return null;
  const saved = await page.until(`(() => {
    const status = document.querySelector("#save-status").textContent;
    return status === "내 이야기에 저장했어요." ? "saved" :
      status.startsWith("저장하지 못했어요.") && "save_failed";
  })()`);
  if (saved !== "saved") return null;
  const record = await page.evaluate(`fetch("/api/history", {headers: {"X-Profile-ID": localStorage.getItem("aac.profile")}})
    .then(async (response) => (await response.json()).messages[0])`);
  assert.equal(record.sentence, delivered);
  assert.equal(record.mode, mode);
  assert.equal(record.context, context);
  assert.ok(record.cards_seen > 0);
  return record;
}

async function runTrial(page, infer, task, persona, evidence, mode, maxTurns, timeout, saveScreen) {
  await setUpTrial(page, mode, task.context, timeout);
  const trace = [], sources = new Set(), noticed = new Set();
  let decisionMs = 0, delivered = null, outcome = "turn_limit";
  const started = performance.now();
  for (let turn = 1; turn <= maxTurns; turn += 1) {
    await page.until(ready, timeout);
    await page.evaluate(frame);
    sources.add(await page.evaluate(`(() => {
      const caption = document.querySelector("#source").textContent;
      if (caption === "내 기기 AI 추천") return "local_llm";
      if (caption.startsWith("기본 그림판")) return "static";
      if (caption === "연결 확인 필요") return "connection_error";
      return document.querySelector("#recommend-notice").textContent ? "fallback" : "rules";
    })()`));
    const observation = await page.evaluate(`(${annotateScreen})()`);
    let result;
    try {
      result = await page.send("Page.captureScreenshot", {format: "png", captureBeyondViewport: false});
    } finally {
      await page.evaluate('document.querySelector("#aac-eval-markers")?.remove()');
    }
    const screenshot = result.data;
    const screenshotPath = await saveScreen?.(mode, persona.id, task.id, turn, screenshot);
    const decisionStart = performance.now();
    const decision = checkedDecision(await infer({
      schema: actionSchema, prompt: actingPrompt(task, persona, evidence, trace), screenshot,
    }), observation.targets);
    const elapsed = Math.round(performance.now() - decisionStart);
    decisionMs += elapsed;
    for (const card of decision.noticed) noticed.add(card);
    const {noticed: cards, ...event} = decision;
    trace.push({...event, noticed_cards: cards, decision_ms: elapsed, ...(screenshotPath ? {screenshot: screenshotPath} : {})});
    if (decision.action === "stop" && !decision.rejected) { outcome = "agent_stopped"; break; }
    await perform(page, decision, observation);
    await page.evaluate(frame);
    delivered = await page.evaluate('document.querySelector("#output-dialog").open ? document.querySelector("#output-text").textContent.trim() : null');
    if (delivered) { outcome = "delivered"; break; }
  }
  const observedMs = Math.round(performance.now() - started);
  const exposed = await page.evaluate("cardExposure.total");
  const record = await storedMetrics(page, delivered, mode, task.context);
  const score = await rateDelivery(infer, task, persona, delivered);
  return {
    persona: persona.id, task: task.id, mode, outcome, intended_sentence: task.intent,
    delivered_sentence: delivered, ...score, app_cards_seen: record?.cards_seen ?? null,
    viewport_card_exposures: record?.cards_seen ?? exposed, model_reported_unique_cards: noticed.size,
    completion_ms: delivered ? observedMs : null, attempt_ms: observedMs, decision_ms: decisionMs,
    elapsed_without_model_ms: Math.max(0, observedMs - decisionMs), app_elapsed_ms: record?.elapsed_ms ?? null,
    saved: Boolean(record), board_sources: [...sources], trace,
  };
}

const mean = (values) => values.length ? Math.round(100 * values.reduce((a, b) => a + b, 0) / values.length) / 100 : null;

function summarize(results) {
  return Object.fromEntries(modes.flatMap((mode) => {
    const rows = results.filter((row) => row.mode === mode);
    return rows.length ? [[mode, {
      attempts: rows.length, completed: rows.filter((row) => row.outcome === "delivered").length,
      fully_expressed: rows.filter((row) => row.score === 5).length,
      mean_score_out_of_5: mean(rows.map((row) => row.score)),
      mean_viewport_card_exposures: mean(rows.map((row) => row.viewport_card_exposures)),
      mean_model_reported_unique_cards: mean(rows.map((row) => row.model_reported_unique_cards)),
      mean_completion_ms_completed: mean(rows.map((row) => row.completion_ms).filter((value) => value !== null)),
      mean_decision_ms: mean(rows.map((row) => row.decision_ms)),
    }]] : [];
  }));
}

async function main() {
  const {values} = parseArgs({options: {
    output: {type: "string"}, screenshots: {type: "string"}, model: {type: "string", default: "gpt-6.1-terra"},
    persona: {type: "string"}, task: {type: "string"}, mode: {type: "string"},
    "local-manifest": {type: "string"}, "max-turns": {type: "string", default: "24"}, help: {type: "boolean"},
  }});
  if (values.help) {
    console.log("CHROMIUM_PATH=/path/to/chrome node scripts/evaluate-agent.mjs [--output FILE] [--screenshots DIR] [--model MODEL] [--persona ID] [--task ID] [--mode adaptive|static] [--max-turns N] [--local-manifest FILE]");
    return;
  }
  const maxTurns = Number(values["max-turns"]);
  if (!Number.isInteger(maxTurns) || maxTurns < 1 || maxTurns > 100) throw new Error("max-turns must be 1–100.");
  const corpus = JSON.parse(await readFile(new URL("../data/agent-scenarios.json", import.meta.url), "utf8"));
  const personas = corpus.personas.filter(({id}) => !values.persona || id === values.persona);
  const tasks = corpus.tasks.filter(({id}) => !values.task || id === values.task);
  const selectedModes = modes.filter((mode) => !values.mode || mode === values.mode);
  if (!personas.length || !tasks.length || !selectedModes.length) throw new Error("Unknown persona, task, or mode.");
  const localManifest = values["local-manifest"];
  async function saveScreen(mode, persona, task, turn, screenshot) {
    const folder = `${values.screenshots}/${mode}-${persona}-${task}`;
    await mkdir(folder, {recursive: true});
    const path = `${folder}/${String(turn).padStart(2, "0")}.png`;
    await writeFile(path, Buffer.from(screenshot, "base64"));
    return path;
  }
  const report = {
    kind: "synthetic-aac-vision-llm-evaluation", actor_and_rater_model: values.model,
    app_recommendation_engine: localManifest ? "local-model-with-fallback" : "rules",
    viewport: {width: 1365, height: 1000}, max_turns: maxTurns, planned_trials: personas.length * tasks.length * selectedModes.length,
    evidence: corpus.evidence, personas, results: [],
  };
  const writeReport = async () => {
    report.summary = summarize(report.results);
    await mkdir(dirname(values.output), {recursive: true});
    await writeFile(values.output, JSON.stringify(report, null, 2) + "\n");
  };
  await withVisionModel(values.model, (infer) => withBrowser(async (page) => {
    for (const persona of personas) for (const task of tasks) for (const mode of selectedModes) {
      console.error(`AAC trial: ${persona.id} / ${task.id} / ${mode}`);
      report.results.push(await runTrial(page, infer, task, persona, corpus.evidence, mode,
        maxTurns, localManifest ? 140000 : 10000, values.screenshots ? saveScreen : null));
      if (values.output) await writeReport();
    }
    assert.deepEqual(page.errors, []);
  }, {localManifest}));
  if (values.output) console.log(JSON.stringify(report.summary, null, 2));
  else console.log(JSON.stringify({...report, summary: summarize(report.results)}, null, 2));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
