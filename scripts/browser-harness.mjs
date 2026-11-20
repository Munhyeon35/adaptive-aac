// Use Chromium DevTools pipes and the real Python HTTP handler without opening a TCP port.
import {spawn} from "node:child_process";
import {once} from "node:events";
import {mkdtemp, rm} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {createInterface} from "node:readline";
import {setTimeout as delay} from "node:timers/promises";

const origin = "http://localhost:8765";

export async function withBrowser(run, {localManifest, visible = false} = {}) {
  const executable = process.env.CHROMIUM_PATH;
  if (!executable) throw new Error("Set CHROMIUM_PATH to your Chrome or Chromium executable.");
  const profile = await mkdtemp(join(tmpdir(), "aac-chromium-"));
  const chromium = spawn(executable, [
    ...(visible ? ["--window-size=1365,1000"] : ["--headless"]), "--remote-debugging-pipe", "--no-first-run", "--disable-background-networking",
    `--user-data-dir=${profile}`, "about:blank",
  ], {stdio: ["ignore", "ignore", "pipe", "pipe", "pipe"]});
  const bridge = spawn(process.env.PYTHON || "python3", [
    "-m", "scripts.browser_bridge", ...(localManifest ? ["--local-manifest", localManifest] : []),
  ], {stdio: ["pipe", "pipe", "inherit"], env: {...process.env, PYTHONIOENCODING: "utf-8"}});
  let browserError = "", buffer = "", nextId = 0;
  const errors = [];
  const pending = new Map();
  const bridgePending = new Map();
  let closing = false, failure;

  function fail(message) {
    if (closing || failure) return;
    failure = new Error(message);
    for (const respond of pending.values()) respond({error: failure});
    pending.clear();
  }

  for (const [child, label] of [[chromium, "Chromium"], [bridge, "Python bridge"]]) {
    child.on("error", (error) => fail(`${label}: ${error.message}`));
    child.on("exit", (code, signal) => fail(`${label} exited (${signal || code}). ${browserError}`));
  }
  for (const stream of [chromium.stdio[3], bridge.stdin]) {
    stream.on("error", (error) => fail(error.message));
  }
  chromium.stderr.on("data", (chunk) => { browserError = (browserError + chunk).slice(-1500); });
  createInterface({input: bridge.stdout}).on("line", (line) => {
    const response = JSON.parse(line);
    bridgePending.get(response.id)?.(response);
    bridgePending.delete(response.id);
  });
  function cdp(method, params = {}, sessionId) {
    if (failure) return Promise.reject(failure);
    const id = ++nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        pending.delete(id);
        reject(new Error(`CDP timeout: ${method}. ${errors.join(" ")} ${browserError}`));
      }, 10000);
      pending.set(id, (response) => {
        clearTimeout(timer);
        if (response.error) reject(new Error(response.error.message));
        else resolve(response.result);
      });
      chromium.stdio[3].write(JSON.stringify({id, method, params, sessionId}) + "\0");
    });
  }

  chromium.stdio[4].setEncoding("utf8");
  chromium.stdio[4].on("data", (chunk) => {
    buffer += chunk;
    let split;
    while ((split = buffer.indexOf("\0")) >= 0) {
      const raw = buffer.slice(0, split);
      buffer = buffer.slice(split + 1);
      if (!raw) continue;
      const message = JSON.parse(raw);
      if (message.id) {
        pending.get(message.id)?.(message);
        pending.delete(message.id);
      } else if (message.method === "Fetch.requestPaused") {
        const {requestId, request} = message.params;
        const url = new URL(request.url);
        if (url.origin !== origin) {
          cdp("Fetch.failRequest", {requestId, errorReason: "BlockedByClient"}, message.sessionId)
            .catch((error) => errors.push(error.message));
          continue;
        }
        const id = ++nextId;
        bridgePending.set(id, (result) => cdp("Fetch.fulfillRequest", {
          requestId, responseCode: result.status, responseHeaders: result.headers, body: result.body,
        }, message.sessionId).catch((error) => errors.push(error.message)));
        bridge.stdin.write(JSON.stringify({
          id, path: url.pathname, method: request.method, headers: request.headers, body: request.postData || "",
        }) + "\n");
      } else if (message.method === "Runtime.exceptionThrown") {
        errors.push(message.params.exceptionDetails.exception?.description || message.params.exceptionDetails.text);
      }
    }
  });

  try {
    const {targetId} = await cdp("Target.createTarget", {url: "about:blank"});
    const {sessionId} = await cdp("Target.attachToTarget", {targetId, flatten: true});
    const send = (method, params) => cdp(method, params, sessionId);
    await send("Page.enable");
    await send("Runtime.enable");
    await send("Fetch.enable", {patterns: [{urlPattern: "*"}]});
    await send("Emulation.setDeviceMetricsOverride", {width: 1365, height: 1000, deviceScaleFactor: 1, mobile: false});
    async function navigate() {
      await send("Storage.clearDataForOrigin", {origin, storageTypes: "local_storage"});
      await send("Page.navigate", {url: origin + "/"});
    }
    async function evaluate(expression) {
      const response = await send("Runtime.evaluate", {expression, returnByValue: true, awaitPromise: true});
      if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
      return response.result.value;
    }
    async function until(expression, timeoutMs = 10000) {
      const deadline = performance.now() + timeoutMs;
      do {
        const value = await evaluate(expression);
        if (value) return value;
        await delay(30);
      } while (performance.now() < deadline);
      throw new Error(`Condition not reached: ${expression}. ${errors.join(" ")}`);
    }
    return await run({send, evaluate, until, errors, navigate});
  } finally {
    closing = true;
    chromium.kill();
    bridge.stdin.end();
    await Promise.all([chromium, bridge].map(async (child) => {
      if (child.exitCode !== null || child.signalCode !== null || !child.pid) return;
      await Promise.race([once(child, "exit"), delay(3000, undefined, {ref: false})]);
      if (child.exitCode === null && child.signalCode === null) {
        child.kill("SIGKILL");
        await once(child, "exit");
      }
    }));
    await rm(profile, {recursive: true, force: true, maxRetries: 3, retryDelay: 100});
  }
}
