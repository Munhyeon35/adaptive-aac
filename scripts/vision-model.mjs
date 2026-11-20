import {spawn} from "node:child_process";
import {mkdtemp, readFile, rm, writeFile} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";

function runCodex(executable, args, prompt) {
  return new Promise((resolve, reject) => {
    const env = {...process.env};
    delete env.CODEX_THREAD_ID;
    const child = spawn(executable, args, {
      stdio: ["pipe", "pipe", "pipe"], timeout: 90000, windowsHide: true, env,
    });
    let events = "", diagnostics = "";
    child.stdout.setEncoding("utf8").on("data", (chunk) => {
      events += chunk;
      if (events.length > 1024 * 1024) child.kill();
    });
    child.stderr.on("data", (chunk) => { diagnostics = (diagnostics + chunk).slice(-1000); });
    child.stdin.on("error", (error) => { if (error.code !== "EPIPE") reject(error); });
    child.on("error", reject);
    child.on("close", (code, signal) => {
      if (code !== 0) {
        reject(new Error(`Codex vision request failed (${signal || code}). ${diagnostics.trim()}`));
        return;
      }
      try {
        const items = events.split("\n").filter(Boolean).map((line) => JSON.parse(line));
        if (items.some((event) => event.item && !["agent_message", "reasoning"].includes(event.item.type))) {
          throw new Error("Vision model attempted to use a tool outside the screenshot interface.");
        }
        if (!items.some((event) => event.type === "turn.completed")) throw new Error("Vision model turn did not complete.");
        resolve();
      } catch (error) {
        reject(error);
      }
    });
    child.stdin.end(prompt, "utf8");
  });
}

export async function withVisionModel(model, run) {
  const folder = await mkdtemp(join(tmpdir(), "aac-vision-"));
  const executable = process.env.AAC_CODEX || "codex";
  try {
    return await run(async ({prompt, screenshot, schema}) => {
      const schemaFile = join(folder, "response-schema.json");
      const output = join(folder, "decision.json");
      const image = join(folder, "screen.png");
      await writeFile(schemaFile, JSON.stringify(schema));
      await rm(output, {force: true});
      if (screenshot) await writeFile(image, Buffer.from(screenshot, "base64"));
      await runCodex(executable, [
        "exec", "--ignore-user-config", "--ignore-rules", "--ephemeral", "--sandbox", "read-only",
        "--skip-git-repo-check", "--color", "never", "--json", "-C", folder,
        "-c", "mcp_servers={}", "--model", model,
        "--output-schema", schemaFile, "--output-last-message", output,
        ...(screenshot ? ["--image", image] : []), "--", "-",
      ], prompt);
      return JSON.parse(await readFile(output, "utf8"));
    });
  } finally {
    await rm(folder, {recursive: true, force: true});
  }
}
