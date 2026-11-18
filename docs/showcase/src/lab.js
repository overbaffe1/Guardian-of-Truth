"use strict";

const { cases, packing, scores } = JSON.parse(document.querySelector("#lab-data").textContent);
const $ = (id) => document.getElementById(id);
const number = new Intl.NumberFormat("ru-RU", { useGrouping: "always" });
const statuses = { whole: "Целиком", short: "Сокращён", marker: "Только маркер", gone: "Удалён" };

function button(text, index, change) {
  const control = document.createElement("button");
  control.type = "button";
  control.textContent = text;
  control.addEventListener("click", () => change(index));
  return control;
}

function press(group, index) {
  [...group.children].forEach((child, i) => child.setAttribute("aria-pressed", String(i === index)));
}

let caseIndex = 1;
const caseGroup = $("cases");
for (const [i, item] of cases.entries()) caseGroup.append(button(item.title, i, showCase));

function showCase(index) {
  caseIndex = index;
  const item = cases[index];
  press(caseGroup, index);
  $("case-prompt").textContent = item.prompt;
  $("case-response").textContent = item.response;
  $("verdict").dataset.tone = "idle";
  $("label").textContent = "—";
  $("verdict-title").textContent = "Ожидает сверки";
  $("verdict-note").textContent = "Откройте записанный вердикт для этого хода.";
  $("signal-list").replaceChildren();
}

$("inspect").addEventListener("click", () => {
  const item = cases[caseIndex];
  $("verdict").dataset.tone = item.label ? "alert" : "clear";
  $("label").textContent = `=${item.label}`;
  $("verdict-title").textContent = item.label ? "Найдена галлюцинация" : "Правила не нашли ошибки";
  $("verdict-note").textContent = item.note;
  $("signal-list").replaceChildren(...item.signals.map((signal) => {
    const badge = document.createElement("li");
    badge.textContent = `${signal.severity} · ${signal.kind}`;
    badge.title = signal.message;
    return badge;
  }));
});
showCase(caseIndex);

let frameIndex = packing.proof_index - 1;
const budgetSlider = $("budget");
const timeline = $("timeline");
for (const [i, row] of packing.entries.entries()) {
  const item = button(String(row.number).padStart(2, "0"), i, showFrame);
  if (row.proof) item.classList.add("is-proof");
  timeline.append(item);
}
for (const item of packing.caps) {
  const tick = document.createElement("span");
  tick.textContent = number.format(item.budget);
  $("budget-ticks").append(tick);
}
$("raw-chars").textContent = number.format(packing.raw_chars);
$("packed-reply").textContent = packing.reply;

function showFrame(index) {
  frameIndex = index;
  const before = packing.entries[index];
  const packed = packing.caps[Number(budgetSlider.value)].rows[index];
  press(timeline, index);
  $("frame-no").textContent = `№${String(before.number).padStart(2, "0")}`;
  $("frame-state").textContent = `· ${statuses[packed.state]}`;
  $("frame-before").textContent = before.body;
  $("frame-after").textContent = packed.body || "В упакованном журнале этой записи нет.";
  $("frame-before").scrollTop = 0;
  $("frame-after").scrollTop = 0;
}

function showBudget() {
  const cap = packing.caps[Number(budgetSlider.value)];
  $("budget-value").textContent = number.format(cap.budget);
  budgetSlider.setAttribute("aria-valuetext", `${number.format(cap.budget)} символов`);
  $("packed-chars").textContent = number.format(cap.chars);
  $("payloads").textContent = `${cap.payloads} / ${packing.entries.length}`;
  $("proof").textContent = cap.proof_present ? "◆ УЛИКА D-17 НА МЕСТЕ" : "◇ УЛИКА D-17 ПОТЕРЯНА";
  for (const [i, entry] of packing.entries.entries()) {
    const cell = timeline.children[i];
    const state = cap.rows[i].state;
    cell.className = `${state}${entry.proof ? " is-proof" : ""}`;
    cell.setAttribute("aria-label", `Кадр ${entry.number}: ${statuses[state]}${entry.proof ? "; источник D-17" : ""}`);
    cell.title = cell.getAttribute("aria-label");
  }
  showFrame(frameIndex);
}
budgetSlider.addEventListener("input", showBudget);
showBudget();

const types = { tp: "Найдены", fn: "Пропущены", fp: "Ложные тревоги", tn: "Верно чистые" };
const modeGroup = $("modes");
for (const [i, mode] of scores.regimes.entries()) modeGroup.append(button(mode.title, i, showScore));
$("cohort").textContent = `${scores.total} ХОДОВ / ${scores.positives} С ГАЛЛЮЦИНАЦИЯМИ`;

function showScore(index) {
  const mode = scores.regimes[index];
  press(modeGroup, index);
  $("f1").textContent = mode.f1.toFixed(3);
  $("precision-recall").textContent = `Precision ${mode.precision.toFixed(3)} · Recall ${mode.recall.toFixed(3)}`;
  const cells = document.createDocumentFragment();
  const labels = document.createDocumentFragment();
  for (const [key, title] of Object.entries(types)) {
    const count = mode.matrix[key];
    for (let i = 0; i < count; i++) {
      const cell = document.createElement("span");
      cell.className = key;
      cell.title = title;
      cells.append(cell);
    }
    const caption = document.createElement("li");
    const swatch = document.createElement("i");
    swatch.className = `swatch ${key}`;
    caption.append(swatch, `${title}: ${count}`);
    labels.append(caption);
  }
  $("matrix").replaceChildren(cells);
  $("legend").replaceChildren(labels);
}
showScore(scores.regimes.findIndex((mode) => mode.key === "cpu"));
