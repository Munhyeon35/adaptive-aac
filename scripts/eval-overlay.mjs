// These observers and controls exist only in the evaluator's browser.
function trackCards() {
  if (window.aacEvalCards) return;
  const previews = new Map();
  const seen = new Set();
  let grid, presentation, total = 0;

  function note(button) {
    if (!button || button.disabled || button.parentElement !== grid) return;
    const id = button.dataset.symbol;
    if (!id || seen.has(id)) return;
    seen.add(id);
    total += 1;
    if (!previews.has(id)) {
      const icon = button.querySelector(".symbol-icon");
      const image = icon?.querySelector("img")?.getAttribute("src");
      previews.set(id, {
        label: button.lastElementChild.textContent.trim(),
        image: image?.startsWith("/symbols/") ? image : null,
        icon: icon?.textContent.trim() || "",
      });
    }
  }

  function record(entries) {
    for (const entry of entries) if (entry.intersectionRatio >= 0.5) note(entry.target);
  }

  const observer = new IntersectionObserver(record, {threshold: 0.5});
  function watch() {
    record(observer.takeRecords());
    observer.disconnect();
    const dialog = document.querySelector("dialog[open]");
    const id = !dialog ? "symbols" : dialog.id === "all-dialog" ? "all-symbols" : null;
    grid = !document.hidden && id ? document.getElementById(id) : null;
    if (!grid || grid.getAttribute("aria-busy") === "true") return;
    const cards = [...grid.querySelectorAll("button[data-symbol]")];
    const key = JSON.stringify([
      id, cards.map((card) => card.dataset.symbol), document.querySelector("#selected").textContent,
      document.querySelector("#categories [aria-pressed=true]")?.textContent,
      document.querySelector("#all-page-label")?.textContent,
    ]);
    if (key !== presentation) {
      presentation = key;
      seen.clear();
    }
    for (const card of cards) if (!card.disabled) observer.observe(card);
  }

  const changes = new MutationObserver(watch);
  for (const id of ["symbols", "all-symbols"]) {
    changes.observe(document.getElementById(id), {childList: true, attributes: true, attributeFilter: ["aria-busy"]});
  }
  for (const dialog of document.querySelectorAll("dialog")) {
    changes.observe(dialog, {attributes: true, attributeFilter: ["open"]});
  }
  document.addEventListener("visibilitychange", watch);
  document.addEventListener("click", (event) => note(event.target.closest("button[data-symbol]")), true);
  window.aacEvalCards = () => {
    record(observer.takeRecords());
    return {total, previews: [...previews.values()]};
  };
  watch();
}

function renderToast({durationMs, completed, count, previews, score}) {
  document.getElementById("aac-eval-result")?.remove();
  const make = (tag, text, styles = {}) => {
    const element = document.createElement(tag);
    if (text !== null) element.textContent = text;
    Object.assign(element.style, styles);
    return element;
  };
  const toast = make("section", null, {
    position: "fixed", inset: "12px 12px auto auto", margin: "0",
    width: "min(420px, calc(100vw - 24px))", maxHeight: "min(450px, calc(100dvh - 24px))",
    overflow: "auto", padding: "16px", color: "#19352a", background: "#fffdf6",
    border: "1px solid #c5d5c7", borderRadius: "16px", boxShadow: "0 8px 34px #12261e36",
    font: '15px/1.45 system-ui, sans-serif', boxSizing: "border-box",
  });
  toast.id = "aac-eval-result";
  toast.setAttribute("popover", "manual");
  toast.setAttribute("aria-label", "평가 결과");
  const heading = make("div", null, {
    display: "flex", justifyContent: "space-between", alignItems: "center", gap: "8px",
  });
  heading.append(make("strong", "평가 결과", {fontSize: "20px"}));
  const close = make("button", "닫기", {padding: "4px 9px", minHeight: "32px"});
  close.type = "button";
  close.addEventListener("click", () => toast.remove());
  heading.append(close);
  const seconds = (durationMs / 1000).toLocaleString("ko-KR", {maximumFractionDigits: 2});
  const summary = make("p", `${completed ? "완료 시간" : "미완료 · 시도 시간"} ${seconds}초 · 카드 노출 ${count}회`,
    {margin: "8px 0 0", fontWeight: "700"});
  summary.setAttribute("role", "status");
  const details = make("p", `의도 전달 ${score}/5 · 서로 다른 카드 ${previews.length}장`,
    {margin: "4px 0 10px", color: "#4a6154", fontSize: "13px"});
  toast.append(heading, summary, details);
  const grid = make("ul", null, {
    display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: "6px",
    maxHeight: "min(240px, 31dvh)", overflowY: "auto", listStyle: "none", margin: "0", padding: "0",
  });
  grid.setAttribute("aria-label", "노출된 이미지 카드");
  for (const card of previews) {
    const tile = make("li", null, {
      display: "grid", justifyItems: "center", gap: "3px", padding: "5px", textAlign: "center",
      border: "1px solid #dedfd2", borderRadius: "8px", background: "white", overflowWrap: "anywhere",
    });
    if (card.image) {
      const image = make("img", null, {width: "30px", height: "30px", objectFit: "contain"});
      image.src = card.image;
      image.alt = "";
      tile.append(image);
    } else {
      const icon = make("span", card.icon || "◇", {fontSize: "20px"});
      icon.setAttribute("aria-hidden", "true");
      tile.append(icon);
    }
    tile.append(make("span", card.label, {fontSize: "11px"}));
    grid.append(tile);
  }
  if (previews.length) toast.append(grid);
  (document.querySelector("dialog[open]") || document.body).append(toast);
  toast.showPopover();
}

export function trackEvalCards(page) {
  return page.evaluate(`(${trackCards})()`);
}

export function readEvalCards(page) {
  return page.evaluate("window.aacEvalCards()");
}

export function showEvalToast(page, result) {
  return page.evaluate(`(${renderToast})(${JSON.stringify(result)})`);
}
