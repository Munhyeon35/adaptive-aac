import assert from "node:assert/strict";
import {test} from "node:test";
import {checkedDecision} from "./evaluate-agent.mjs";

const targets = [{card: "symbols:bread"}, {card: null}, {card: "symbols:water"}];

test("only visible controls can receive model clicks", () => {
  const raw = {action: "click", target: 1, noticed_cards: [1, 1, 2, 3, 99], note: "Two cards visible"};
  assert.deepEqual(checkedDecision(raw, targets), {
    action: "click", target: 1, note: raw.note, rejected: false, noticed: ["symbols:bread", "symbols:water"],
  });
  assert.equal(checkedDecision({...raw, target: 99}, targets).rejected, true);
  assert.equal(checkedDecision({...raw, target: 0}, targets).rejected, true);
});

test("non-click actions cannot smuggle coordinates or commands", () => {
  const raw = {action: "scroll_down", target: 0, noticed_cards: [], note: "Need to scroll"};
  assert.equal(checkedDecision(raw, targets).rejected, false);
  assert.equal(checkedDecision({...raw, target: 3}, targets).rejected, true);
  assert.throws(() => checkedDecision({...raw, action: "execute"}, targets), /Invalid structured decision/);
});
