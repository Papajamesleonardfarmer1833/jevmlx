/* Static showcase: renders precomputed decision data (data/demo.json). No model runs here. */

const state = { data: null, active: 0 };

async function boot() {
  const demo = document.getElementById("demo");
  try {
    const res = await fetch("data/demo.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.data = await res.json();
  } catch (err) {
    demo.innerHTML = `<p class="error">Could not load data/demo.json (${err.message}).<br>` +
      `If you are viewing this from the filesystem, run a local server first: <code>python -m http.server</code></p>`;
    return;
  }

  document.getElementById("model-name").textContent = state.data.model;
  renderTabs();
  render(0);
}

function renderTabs() {
  const tabs = document.getElementById("preset-tabs");
  tabs.innerHTML = "";
  state.data.presets.forEach((preset, i) => {
    const b = document.createElement("button");
    b.className = "tab" + (i === state.active ? " active" : "");
    b.setAttribute("role", "tab");
    b.textContent = `${preset.title} (${preset.num_fields} fields)`;
    b.addEventListener("click", () => render(i));
    tabs.appendChild(b);
  });
}

function fmtMs(ms) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

function confColor(p) {
  if (p >= 0.9) return "var(--green)";
  if (p >= 0.7) return "var(--amber)";
  return "var(--red)";
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function render(index) {
  state.active = index;
  renderTabs();
  const preset = state.data.presets[index];
  const p = preset.parallel;
  const n = preset.naive;

  const fieldRows = p.fields.map((f) => {
    const top = f.top.slice(1, 3).map((t) => `${esc(t.choice)} ${(t.p * 100).toFixed(1)}%`).join(" · ");
    return `<tr>
      <td class="fname">${esc(f.name)}<div class="ftype">${esc(f.type)}</div></td>
      <td class="fval">${esc(f.value)}</td>
      <td>
        <div class="conf">
          <div class="conf-bar"><div class="conf-fill" style="width:${(f.confidence * 100).toFixed(1)}%;background:${confColor(f.confidence)}"></div></div>
          <span class="conf-num">${(f.confidence * 100).toFixed(1)}%</span>
        </div>
        ${top ? `<div class="alts">runners-up: ${top}</div>` : ""}
      </td>
    </tr>`;
  }).join("");

  const jsonStr = esc(JSON.stringify(p.json, null, 2));

  const naiveOk = n.schema_match;
  const naiveBadge = naiveOk
    ? `<span class="pill good">valid JSON</span>`
    : `<span class="pill bad">malformed / incomplete JSON</span>`;

  document.getElementById("demo").innerHTML = `
    <div class="demo-meta">
      <span class="pill">parallel <strong>${fmtMs(p.elapsed_ms)}</strong> (prefill ${fmtMs(p.prefill_ms)} + batched pass ${fmtMs(p.suffix_eval_ms)})</span>
      <span class="pill">naive <strong>${fmtMs(n.elapsed_ms)}</strong> · ${n.total_tokens} tokens · ${n.tokens_per_second} tok/s</span>
      <span class="pill ${naiveBadge ? "" : ""}">${naiveBadge}</span>
      <span class="pill good">parallel: schema-valid</span>
    </div>

    <details class="context">
      <summary>Show the input context (${preset.context.length} chars)</summary>
      <pre>${esc(preset.context)}</pre>
    </details>

    <table class="field-table">
      <thead><tr><th style="width:22%">field</th><th style="width:20%">decision</th><th>confidence</th></tr></thead>
      <tbody>${fieldRows}</tbody>
    </table>

    <div class="json">
      <div class="pane">
        <div class="pane-head">
          <span>Assembled JSON — never generated, so never malformed</span>
          <button class="copy-btn" id="copy-json">copy</button>
        </div>
        <pre id="json-pre">${jsonStr}</pre>
      </div>
      <div class="pane">
        <div class="pane-head">
          <span>Naive baseline — same model, writes the JSON itself</span>
        </div>
        <pre>${esc(n.raw_text)}</pre>
      </div>
    </div>
  `;

  const btn = document.getElementById("copy-json");
  if (btn) {
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(JSON.stringify(p.json, null, 2));
        btn.textContent = "copied";
        setTimeout(() => (btn.textContent = "copy"), 1400);
      } catch {
        btn.textContent = "copy failed";
      }
    });
  }
}

boot();
