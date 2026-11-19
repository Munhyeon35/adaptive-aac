import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {test} from "node:test";
import {rateIntent, seededRandom} from "./agent-policy.mjs";

const {tasks} = JSON.parse(readFileSync(new URL("../data/agent-scenarios.json", import.meta.url), "utf8"));

test("intent scoring accepts paraphrases and distinguishes omissions, opposite polarity and no output", () => {
  const refusal = tasks.find(({id}) => id === "refusal");
  assert.equal(rateIntent(refusal, "저는 빵을 안 먹을래요!").score, 5);
  assert.equal(rateIntent(refusal, "먹고 싶지 않아요.").score, 3);
  assert.equal(rateIntent(refusal, "빵을 먹고 싶어요.").score, 2);
  assert.equal(rateIntent(refusal, "").score, 1);
  const moreWater = tasks.find(({id}) => id === "more_water");
  assert.equal(rateIntent(moreWater, "물을 마시고 싶어요.").score, 4);
  assert.ok(tasks.every((task) => rateIntent(task, task.intent).score === 5));
});

test("each paired trial uses the same random stream for motor errors", () => {
  const sample = (seed) => {
    const draw = seededRandom(seed, "motor_recovery:refusal");
    return Array.from({length: 10}, draw);
  };
  assert.deepEqual(sample(42), sample(42));
  assert.notDeepEqual(sample(42), sample(43));
});
