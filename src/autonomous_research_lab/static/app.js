"use strict";
const $ = (id) => document.getElementById(id);
const token = document.querySelector('meta[name="arl-token"]').content;
const state = {overview: null, selected: {programs: null, experiments: null, attempts: null}, signatures: {}, tab: "programs"};
const terminal = new Set(["completed", "failed", "stopped", "timed_out", "interrupted", "limit_exceeded"]);
const node = (tag, text, className) => {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
};
function error(message) {
  $("global-error").textContent = message;
  $("global-error").classList.toggle("hidden", !message);
}
async function api(path, body) {
  const options = body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json", "X-ARL-Token": token}, body: JSON.stringify(body)};
  const response = await fetch(path, {cache: "no-store", ...options});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
function button(label, action, className = "ghost") {
  const result = node("button", label, className);
  result.type = "button";
  result.addEventListener("click", async () => {
    result.disabled = true;
    error("");
    try { await action(); } catch (exception) { error(exception.message); }
    finally { result.disabled = false; }
  });
  return result;
}
function jsonBlock(label, value, open = false) {
  const details = node("details", undefined, "record-section");
  details.dataset.section = label;
  details.open = open;
  details.append(node("summary", label), node("pre", JSON.stringify(value, null, 2)));
  return details;
}
function card(title, subtitle) {
  const result = node("article", undefined, "card");
  result.append(node("h2", title));
  if (subtitle) result.append(node("p", subtitle, "subtle mono"));
  return result;
}
function badge(value) { return node("span", value, "badge " + (terminal.has(value) ? "settled" : "active-state")); }
function switchTab(tab) {
  state.tab = tab;
  document.querySelectorAll(".tab").forEach((item) => {
    const active = item.dataset.tab === tab;
    item.classList.toggle("active", active);
    item.setAttribute("aria-pressed", String(active));
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("hidden", panel.id !== "tab-" + tab));
}
async function select(kind, id) {
  state.selected[kind] = id;
  delete state.signatures[kind];
  switchTab(kind);
  await renderDetail(kind);
  $(kind.slice(0, -1) + "-detail").scrollIntoView({block: "start", behavior: "smooth"});
}
function renderLists() {
  const overview = state.overview;
  const runner = overview.runner;
  const allocation = runner.gpu_uuid ? `GPU ${runner.gpu_uuid}` : "No GPU configured";
  $("runner-status").textContent = runner.configured ? `${runner.image} · ${allocation}` : runner.detail;
  for (const kind of ["programs", "experiments", "attempts"]) {
    const target = $(kind + "-list");
    const fragment = document.createDocumentFragment();
    const rows = overview[kind];
    if (!rows.length) fragment.append(node("p", `No recorded ${kind} yet.`, "empty"));
    for (const row of rows) {
      const title = row.brief?.goal || row.manifest?.name || row.id;
      const entry = node("article", undefined, "list-row");
      const main = node("div");
      main.append(button(title, () => select(kind, row.id), "record-link"), node("p", row.id, "mono subtle"));
      entry.append(main);
      if (row.state) entry.append(badge(row.state));
      fragment.append(entry);
    }
    target.replaceChildren(fragment);
  }
}
async function programView(program) {
  const result = card(program.brief.goal, program.id);
  const status = node("div", undefined, "actions");
  status.append(badge(program.state), node("span", `Phase: ${program.phase}`, "subtle"));
  const actions = [];
  if (program.state === "draft") actions.push("start", "pause", "stop");
  if (program.state === "running") actions.push("pause", "stop");
  if (["paused", "blocked"].includes(program.state)) actions.push("resume", "stop");
  if (program.state === "pausing") actions.push("stop");
  for (const action of actions) status.append(button(action[0].toUpperCase() + action.slice(1), async () => {
    await api(`/api/programs/${program.id}/${action}`, {});
    await refresh();
  }, action === "stop" ? "danger" : "ghost"));
  status.append(button("Report", async () => {
    const report = await api(`/api/programs/${program.id}/report`);
    $("report-body").replaceChildren(node("pre", report.markdown));
    $("report-dialog").classList.remove("hidden");
    $("report-close").focus();
  }));
  result.append(status);
  if (program.stop_reason) result.append(node("p", program.stop_reason, "banner"));
  if (program.active_attempt_id) result.append(button("Inspect active attempt", () => select("attempts", program.active_attempt_id)));
  result.append(jsonBlock("Resource accounting", program.usage, true));
  result.append(jsonBlock("Research brief", program.brief));
  result.append(jsonBlock("Questions and relationships", program.questions));
  result.append(jsonBlock("Designs and alternatives", program.designs));
  result.append(jsonBlock("Decisions and evidence", program.decisions));
  result.append(jsonBlock("Model-authored analyses", program.analyses));
  result.append(jsonBlock("Sources and retrieval records", program.sources));
  const history = node("section", undefined, "record-section");
  history.append(node("h3", "Implementations and attempts"));
  for (const implementation of program.implementations) {
    const line = node("div", undefined, "actions");
    line.append(node("span", `${implementation.id} · ${implementation.status}`, "mono"));
    if (implementation.experiment_id) line.append(button("Code and settings", () => select("experiments", implementation.experiment_id)));
    if (implementation.attempt_id && implementation.status !== "cancelled_before_launch") line.append(button("Execution", () => select("attempts", implementation.attempt_id)));
    history.append(line);
  }
  result.append(history);
  const calls = node("section", undefined, "record-section");
  calls.append(node("h3", "Model calls"), node("p", "Failed and interrupted calls remain charged. Unknown token usage is not zero cost.", "hint"));
  for (const call of program.calls) {
    calls.append(button(`${call.role} · ${call.status} · ${call.id}`, async () => {
      const evidence = await api(`/api/programs/${program.id}/calls/${call.id}`);
      const view = node("div");
      for (const [kind, record] of Object.entries(evidence)) {
        view.append(node("h3", kind + (record?.truncated ? " (bounded preview)" : "")), node("pre", record?.text || "No response recorded."));
      }
      $("report-body").replaceChildren(view);
      $("report-dialog").classList.remove("hidden");
      $("report-close").focus();
    }, "record-link"));
  }
  result.append(calls);
  return result;
}
async function experimentView(experiment) {
  const result = card(experiment.manifest.name, experiment.id);
  result.append(node("p", experiment.manifest.question), node("p", experiment.manifest.procedure));
  result.append(button("Run this saved experiment", async () => {
    const attempt = await api(`/api/experiments/${experiment.id}/run`, {});
    await refresh();
    await select("attempts", attempt.id);
  }));
  result.append(jsonBlock("Manifest and settings", experiment.manifest, true));
  result.append(node("p", `Bundle SHA-256: ${experiment.bundle_hash}`, "mono hint"));
  const files = await api(`/api/experiments/${experiment.id}/code`);
  const source = node("section", undefined, "record-section");
  source.append(node("h3", "Saved source files"));
  const chooser = node("select");
  chooser.setAttribute("aria-label", "Source file");
  for (const name of files.files) {
    const option = node("option", name);
    option.value = name;
    chooser.append(option);
  }
  chooser.value = experiment.manifest.entrypoint;
  const preview = node("pre");
  const note = node("p", "", "hint");
  async function load() {
    const value = await api(`/api/experiments/${experiment.id}/code/` + chooser.value.split("/").map(encodeURIComponent).join("/"));
    preview.textContent = value.text;
    note.textContent = value.truncated ? "Source preview limited to the first 64 KiB." : "Read-only source snapshot.";
  }
  chooser.addEventListener("change", () => load().catch((exception) => error(exception.message)));
  source.append(chooser, note, preview);
  result.append(source);
  if (files.files.length) await load();
  const attempts = state.overview.attempts.filter((attempt) => attempt.experiment_id === experiment.id);
  for (const attempt of attempts) result.append(button(`${attempt.id} · ${attempt.state}`, () => select("attempts", attempt.id), "record-link"));
  return result;
}
async function attemptView(attempt) {
  const result = card("Execution attempt", attempt.id);
  const actions = node("div", undefined, "actions");
  actions.append(badge(attempt.state), button("Source experiment", () => select("experiments", attempt.experiment_id)));
  if (!terminal.has(attempt.state)) actions.append(button("Stop attempt", async () => {
    await api(`/api/attempts/${attempt.id}/stop`, {});
    await refresh();
  }, "danger"));
  result.append(actions);
  if (attempt.error) result.append(node("p", attempt.error, "banner error"));
  result.append(node("p", `Scientific outcome: ${attempt.result?.scientific_outcome || "not assessed"}. Execution status alone does not establish a finding.`, "hint"));
  if (attempt.result) {
    result.append(jsonBlock("Recorded measurements", attempt.result.observations, true));
    result.append(jsonBlock("Resource use and environment", {wall_seconds: attempt.result.wall_seconds, output_bytes: attempt.result.output_bytes, execution: attempt.execution}));
    const artifacts = node("section", undefined, "record-section");
    artifacts.append(node("h3", "Artifacts"));
    for (const artifact of attempt.result.artifacts || []) {
      const line = node("p");
      const link = node("a", `${artifact.path} (${artifact.bytes} bytes)`);
      link.href = `/api/attempts/${attempt.id}/artifacts/` + artifact.path.split("/").map(encodeURIComponent).join("/");
      link.setAttribute("download", "");
      line.append(link, node("small", ` SHA-256: ${artifact.sha256}`, "mono"));
      artifacts.append(line);
    }
    result.append(artifacts);
    if (attempt.result.artifact_warnings?.length) result.append(jsonBlock("Artifact warnings", attempt.result.artifact_warnings, true));
  }
  const logs = await api(`/api/attempts/${attempt.id}/logs`);
  const log = node("details", undefined, "record-section");
  log.dataset.section = "logs";
  log.append(node("summary", logs.truncated ? "Logs (last 64 KiB)" : "Logs"), node("pre", logs.text || "No container log output."));
  result.append(log, jsonBlock("Lifecycle events", attempt.events));
  return result;
}
async function renderDetail(kind) {
  const id = state.selected[kind];
  if (!id) return;
  const record = await api(`/api/${kind}/${id}`);
  const signature = JSON.stringify([record, kind === "experiments" ? state.overview.attempts.filter(a => a.experiment_id === id).map(a => [a.id, a.state]) : null]);
  if (state.signatures[kind] === signature) return;
  const target = $(kind.slice(0, -1) + "-detail");
  const open = new Set([...target.querySelectorAll("details[open]")].map((item) => item.dataset.section));
  const view = kind === "programs" ? await programView(record) : kind === "experiments" ? await experimentView(record) : await attemptView(record);
  if (state.selected[kind] !== id) return;
  for (const details of view.querySelectorAll("details")) if (open.has(details.dataset.section)) details.open = true;
  target.replaceChildren(view);
  state.signatures[kind] = signature;
}
async function refresh() {
  state.overview = await api("/api/overview");
  renderLists();
  await renderDetail(state.tab);
}
function number(id, fallback) {
  const value = $(id).value.trim();
  if (!value && fallback !== undefined) return fallback;
  const parsed = Number(value);
  if (!value || !Number.isFinite(parsed)) throw new Error(`Enter a valid value for ${$(id).labels[0].textContent}.`);
  return parsed;
}
$("brief-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!event.currentTarget.reportValidity()) return;
  $("brief-submit").disabled = true;
  error("");
  try {
    const data = Object.create(null);
    let bytes = 0;
    for (const file of $("brief-data").files) {
      bytes += file.size;
      if (bytes > 256 * 1024) throw new Error("Approved data exceed 256 KiB.");
      if (Object.hasOwn(data, file.name)) throw new Error("Data filenames must be unique.");
      data[file.name] = new TextDecoder("utf-8", {fatal: true}).decode(await file.arrayBuffer());
    }
    const brief = {
      goal: $("brief-goal").value.trim(), scope: $("brief-scope").value.trim(),
      sources: $("brief-sources").value.split("\n").map((item) => item.trim()).filter(Boolean), data_files: data,
      model: {base_url: $("brief-model-url").value.trim(), model: $("brief-model-name").value.trim(), api_key_env: $("brief-model-key-env").value.trim() || null},
      policy: $("brief-policy").value, max_experiments: number("brief-max-experiments", 2), max_model_calls: number("brief-max-calls", 12),
      max_output_tokens_per_call: number("brief-tokens"), max_wall_seconds: number("brief-max-wall", 600), max_repairs: number("brief-repairs"),
      resources: {gpu: $("brief-gpu").checked, wall_seconds: number("brief-run-wall"), memory_bytes: number("brief-memory") * 1024 ** 2,
        output_bytes: number("brief-output") * 1024 ** 2, log_bytes: number("brief-log") * 1024 ** 2, pids: number("brief-pids"), cpus: number("brief-cpus", null)}
    };
    const program = await api("/api/programs", brief);
    $("brief-status").textContent = "Saved. Start the program when the brief and permissions are correct.";
    $("brief-card").open = false;
    await refresh();
    await select("programs", program.id);
  } catch (exception) { error(exception.message); }
  finally { $("brief-submit").disabled = false; }
});
for (const tab of document.querySelectorAll(".tab")) tab.addEventListener("click", () => {
  switchTab(tab.dataset.tab);
  refresh().catch((exception) => error(exception.message));
});
function closeReport() { $("report-dialog").classList.add("hidden"); $("tab-programs-btn").focus(); }
$("report-close").addEventListener("click", closeReport);
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeReport(); });
async function poll() {
  try { if (!state.overview || !document.hidden) await refresh(); } catch (exception) { error(exception.message); }
  setTimeout(poll, 3000);
}
poll();
