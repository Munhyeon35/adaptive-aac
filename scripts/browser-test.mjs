// Optional browser integration check. Requires Node 20+ and CHROMIUM_PATH.
// CDP pipes and a Python HTTP bridge allow testing without opening TCP ports.
import {spawn} from "node:child_process";
import {mkdtemp, mkdir, writeFile, rm} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";
import assert from "node:assert/strict";
import {createInterface} from "node:readline";

if (!process.env.CHROMIUM_PATH) throw new Error("Set CHROMIUM_PATH to your Chromium or Chrome headless executable.");
const profile = await mkdtemp(join(tmpdir(), "aac-chromium-"));
const chromium = spawn(process.env.CHROMIUM_PATH, ["--headless", "--remote-debugging-pipe", "--no-first-run", "--disable-background-networking", `--user-data-dir=${profile}`, "about:blank"], {stdio: ["ignore", "ignore", "pipe", "pipe", "pipe"]});
const bridge = spawn(process.env.PYTHON || "python3", ["-m", "scripts.browser_bridge"], {stdio: ["pipe", "pipe", "inherit"]});
const pending = new Map(), bridgePending = new Map();
let sequence = 0, bridgeSequence = 0, buffered = "", browserError = "";
const errors = [];
chromium.stderr.on("data", (chunk) => { browserError += chunk; });
chromium.on("error", (error) => errors.push(error.message));
createInterface({input: bridge.stdout}).on("line", (line) => {
  const response = JSON.parse(line);
  bridgePending.get(response.id)?.(response);
  bridgePending.delete(response.id);
});
function backend(request) {
  const id = ++bridgeSequence;
  return new Promise((resolve) => { bridgePending.set(id, resolve); bridge.stdin.write(JSON.stringify({id, ...request}) + "\n"); });
}
function cdp(method, params = {}, sessionId) {
  const id = ++sequence;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method}. ${browserError.slice(-1500)}`)); }, 10000);
    pending.set(id, {resolve: (value) => { clearTimeout(timer); resolve(value); }, reject: (error) => { clearTimeout(timer); reject(error); }});
    chromium.stdio[3].write(JSON.stringify({id, method, params, ...(sessionId ? {sessionId} : {})}) + "\0");
  });
}
chromium.stdio[4].on("data", (chunk) => {
  buffered += chunk.toString();
  let split;
  while ((split = buffered.indexOf("\0")) >= 0) {
    const raw = buffered.slice(0, split); buffered = buffered.slice(split + 1);
    if (!raw) continue;
    const message = JSON.parse(raw);
    if (message.id) {
      const promise = pending.get(message.id); pending.delete(message.id);
      if (message.error) promise?.reject(new Error(message.error.message)); else promise?.resolve(message.result);
    } else if (message.method === "Fetch.requestPaused") {
      const {requestId, request} = message.params;
      backend({path: new URL(request.url).pathname, method: request.method, headers: request.headers, body: request.postData || ""}).then((result) =>
        cdp("Fetch.fulfillRequest", {requestId, responseCode: result.status, responseHeaders: result.headers, body: result.body}, message.sessionId)
      ).catch((error) => errors.push(error.message));
    } else if (message.method === "Runtime.exceptionThrown") errors.push(message.params.exceptionDetails.text);
  }
});

try {
  const {targetId} = await cdp("Target.createTarget", {url: "about:blank"});
  const {sessionId} = await cdp("Target.attachToTarget", {targetId, flatten: true});
  const send = (method, params) => cdp(method, params, sessionId);
  await send("Page.enable"); await send("Runtime.enable");
  await send("Fetch.enable", {patterns: [{urlPattern: "*"}]});
  await send("Emulation.setDeviceMetricsOverride", {width: 1365, height: 1000, deviceScaleFactor: 1, mobile: false});
  await send("Page.navigate", {url: "http://localhost:8765/"});
  async function evaluate(expression) {
    const response = await send("Runtime.evaluate", {expression, returnByValue: true, awaitPromise: true});
    if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
    return response.result.value;
  }
  async function until(expression) {
    for (let i = 0; i < 100; i += 1) {
      if (await evaluate(expression)) return;
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    throw new Error(`Condition not reached: ${expression}`);
  }
  const click = (selector) => evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
  await until('document.querySelectorAll("#symbols button").length === 6 && document.querySelector("#symbols").getAttribute("aria-busy") === "false"');
  await mkdir("artifacts", {recursive: true});
  async function screenshot(name) {
    const shot = await send("Page.captureScreenshot", {format: "png", captureBeyondViewport: true});
    await writeFile(`artifacts/${name}.png`, Buffer.from(shot.data, "base64"));
  }
  await screenshot("desktop");
  await click('[data-symbol="eat"]');
  await until('document.querySelector("#symbols [data-symbol=bread]") !== null');
  await click('#symbols [data-symbol="bread"]');
  await click("#generate");
  await until('document.querySelector("#sentence").value.includes("빵을 먹고 싶어요")');
  await click("#confirm");
  await until('document.querySelector("#save-status").textContent === "내 이야기에 저장했어요."');
  assert.equal(await evaluate('document.querySelector("#output-text").textContent'), "빵을 먹고 싶어요.");
  await screenshot("message");
  await click("#new-message");
  await click("#history-open");
  await until('document.querySelectorAll(".history-item").length === 1');
  await click('[data-close="history-dialog"]');
  await click("#all-open");
  await evaluate('document.querySelector("#search").value = "화장실"; document.querySelector("#search").dispatchEvent(new Event("input"));');
  assert.equal(await evaluate('document.querySelectorAll("#all-symbols button").length'), 1);
  await click('#all-symbols [data-symbol="bathroom"]');
  await click('[data-close="all-dialog"]');
  assert.equal(await evaluate('document.querySelectorAll("#selected .token").length'), 1);
  await click("#undo");
  assert.equal(await evaluate('document.querySelectorAll("#selected .token").length'), 0);
  await click("#mode-static");
  assert.equal(await evaluate('document.querySelector("#mode-static").getAttribute("aria-pressed")'), "true");
  await click("#refresh");
  assert.ok((await evaluate('document.querySelector("#source").textContent')).includes("2/"));
  await click("#mode-adaptive");
  await send("Emulation.setDeviceMetricsOverride", {width: 390, height: 844, deviceScaleFactor: 1, mobile: true});
  await until('document.querySelector("#symbols").getAttribute("aria-busy") === "false"');
  assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth'), true);
  await screenshot("mobile");
  await click("#settings-open");
  assert.equal(await evaluate('document.querySelector("#cloud-enabled").disabled'), true);
  await click("#delete-history"); await click("#delete-confirm-yes");
  await until('document.querySelector("#settings-status").textContent.includes("기록을 지웠어요")');
  await click('[data-close="settings-dialog"]'); await click("#history-open");
  await until('document.querySelector("#history-list").textContent.includes("아직 저장한")');
  assert.deepEqual(errors, []);
  console.log("PASS: browser communication, persistence, full vocabulary search, undo, static mode, mobile layout, consent, deletion. Screenshots: artifacts/*.png");
} finally {
  chromium.kill(); bridge.kill();
  await new Promise((resolve) => setTimeout(resolve, 200));
  await rm(profile, {recursive: true, force: true});
}
