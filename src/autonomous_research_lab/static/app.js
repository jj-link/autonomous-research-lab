"use strict";
const $ = (id) => document.getElementById(id);
const token = document.querySelector('meta[name="arl-token"]').content;
const state = {overview: null, selected: {programs: null, experiments: null, attempts: null}, signatures: {}, tab: "programs", codexLoginStarting: false};
const terminal = new Set(["completed", "failed", "stopped", "timed_out", "interrupted", "limit_exceeded"]);
const sections = {
  programs: {title: "Research programs", description: "Questions, experiments, and decisions—kept together.", singular: "Research program", empty: "Your next investigation starts here.", hint: "Create a research brief to give the lab a question, a scope, and clear limits."},
  experiments: {title: "Experiments", description: "The designs and code behind each investigation.", singular: "Saved experiment", empty: "No saved experiments yet.", hint: "Designs and their code snapshots will appear here as the research develops."},
  attempts: {title: "Run history", description: "Every execution, with its measurements and a traceable record.", singular: "Execution attempt", empty: "No runs recorded yet.", hint: "Completed, failed, and interrupted attempts will all remain part of the record."}
};
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
function badge(value) {
  const result = node("span", value.replaceAll("_", " "), "badge");
  result.dataset.state = value;
  return result;
}
function switchTab(tab) {
  state.tab = tab;
  $("section-title").textContent = sections[tab].title;
  $("section-label").textContent = sections[tab].title;
  $("section-description").textContent = sections[tab].description;
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
  renderLists();
  await renderDetail(kind);
  if (state.tab !== kind || state.selected[kind] !== id) return;
  const target = $(kind.slice(0, -1) + "-detail");
  target.tabIndex = -1;
  target.focus({preventScroll: true});
  target.scrollIntoView({block: "start"});
}
function openBrief() {
  switchTab("programs");
  $("brief-card").open = true;
  $("brief-card").scrollIntoView({block: "start"});
  $("brief-goal").focus({preventScroll: true});
}
$("new-program").addEventListener("click", openBrief);
function modelConfig(catalog = false) {
  const provider = $("brief-provider").value;
  const config = {provider};
  if (!catalog) config.model = $("brief-model-name").value.trim();
  if (provider !== "openai-codex") {
    config.base_url = $("brief-model-url").value.trim();
    config.api_key_env = $("brief-model-key-env").value.trim() || null;
  }
  return config;
}
function changeProvider() {
  const provider = $("brief-provider").value;
  const codex = provider === "openai-codex";
  $("model-url-field").classList.toggle("hidden", codex);
  $("model-key-field").classList.toggle("hidden", codex);
  $("brief-model-url").disabled = codex;
  $("brief-model-url").required = !codex;
  $("brief-model-key-env").disabled = codex;
  $("brief-model-url").value = provider === "spark" ? (state.overview?.providers.spark_base_url || "") : provider === "openai-compatible" ? "https://api.openai.com/v1" : "";
  $("brief-model-key-env").value = "";
  $("brief-model-name").value = "";
  $("provider-models").replaceChildren();
  $("model-discovery-status").textContent = "";
  $("brief-tokens").disabled = codex;
  $("brief-tokens").value = codex ? "" : "8192";
  $("brief-tokens").placeholder = codex ? "Provider-managed" : "";
  $("model-token-note").textContent = codex ? "Codex manages output length. Call limits and request timeouts still apply; no numeric output-token cap is claimed." : "Sent as the selected server's requested output-token limit.";
}
function renderProviderStatus(status) {
  const pendingMessage = $("codex-continue").hasAttribute("href")
    ? "Complete sign-in in ChatGPT. Use the link below if its tab did not open."
    : "A sign-in is already pending. Finish it in ChatGPT, or cancel to start again.";
  $("codex-status").textContent = state.codexLoginStarting ? "Opening ChatGPT sign-in…" : status.pending ? pendingMessage : status.error || (status.connected ? "Codex account connected." : "Not connected.");
  $("codex-login").disabled = status.pending || state.codexLoginStarting;
  $("codex-login").classList.toggle("hidden", status.connected);
  $("codex-cancel").classList.toggle("hidden", !status.pending);
  $("codex-logout").classList.toggle("hidden", !status.connected);
  if (!status.pending) {
    $("codex-continue").classList.add("hidden");
    $("codex-continue").removeAttribute("href");
  }
}
for (const [id, action] of [["codex-login", "login"], ["codex-cancel", "cancel"], ["codex-logout", "logout"]]) {
  $(id).addEventListener("click", async () => {
    if (action === "login" && state.codexLoginStarting) return;
    let signInTab = null;
    if (action === "login") state.codexLoginStarting = true;
    $(id).disabled = true;
    error("");
    try {
      if (action === "login") {
        // Reserve the tab inside the click gesture, before awaiting OAuth setup.
        signInTab = window.open("about:blank", "_blank");
        if (signInTab) {
          signInTab.opener = null;
          signInTab.document.title = "ChatGPT sign-in";
          const policy = signInTab.document.createElement("meta");
          policy.name = "referrer";
          policy.content = "no-referrer";
          signInTab.document.head.append(policy);
          signInTab.document.body.textContent = "Preparing ChatGPT sign-in…";
        }
      }
      const response = await api("/api/providers/codex/" + action, {});
      if (action === "login") {
        const url = new URL(response.authorization_url);
        if (url.origin !== "https://auth.openai.com") throw new Error("Unexpected Codex authorization destination.");
        $("codex-continue").href = url.href;
        $("codex-continue").classList.remove("hidden");
        if (signInTab && !signInTab.closed) {
          try { signInTab.location.replace(url.href); }
          catch { signInTab.close(); } // The visible link remains the fallback.
        }
        signInTab = null;
      }
      await refresh();
    } catch (exception) {
      if (signInTab && !signInTab.closed) signInTab.close();
      error(exception.message);
    } finally {
      if (action === "login") state.codexLoginStarting = false;
      if (state.overview) renderProviderStatus(state.overview.providers.codex);
      $(id).disabled = id === "codex-login" && (state.codexLoginStarting || Boolean(state.overview?.providers.codex.pending));
    }
  });
}
$("brief-provider").addEventListener("change", changeProvider);
for (const id of ["brief-model-url", "brief-model-key-env"]) {
  $(id).addEventListener("input", () => {
    $("provider-models").replaceChildren();
    $("brief-model-name").value = "";
    $("model-discovery-status").textContent = "Connection changed; choose a model for this connection.";
  });
}
$("model-discover").addEventListener("click", async () => {
  $("model-discover").disabled = true;
  error("");
  const requested = JSON.stringify(modelConfig(true));
  try {
    const result = await api("/api/providers/models", JSON.parse(requested));
    if (JSON.stringify(modelConfig(true)) !== requested) return;
    $("provider-models").replaceChildren(...result.models.map(id => {
      const option = node("option");
      option.value = id;
      return option;
    }));
    $("model-discovery-status").textContent = result.models.length ? "Catalog loaded. Choose an advertised model in the model field." : "This provider advertised no models.";
  } catch (exception) { error(exception.message); }
  finally { $("model-discover").disabled = false; }
});
changeProvider();
function renderLists() {
  const overview = state.overview;
  const runner = overview.runner;
  renderProviderStatus(overview.providers.codex);
  const allocation = runner.gpu_uuid || "No GPU configured";
  $("runner-status").textContent = runner.configured ? `${runner.image} · ${allocation}` : runner.detail;
  $("runner-label").textContent = runner.configured ? "Runner configured" : "Runner not configured";
  $("runner-label").dataset.configured = String(runner.configured);
  const experimentNames = new Map(overview.experiments.map(row => [row.id, row.manifest.name]));
  for (const kind of ["programs", "experiments", "attempts"]) {
    const target = $(kind + "-list");
    const fragment = document.createDocumentFragment();
    const rows = overview[kind];
    $(kind + "-count").textContent = String(rows.length);
    const signature = JSON.stringify([rows, state.selected[kind], kind === "attempts" ? [...experimentNames] : null]);
    if (state.signatures["list-" + kind] === signature) continue;
    if (!rows.length) {
      const empty = node("div", undefined, "empty");
      const mark = node("span", undefined, "empty-mark");
      mark.setAttribute("aria-hidden", "true");
      empty.append(mark, node("h3", sections[kind].empty), node("p", sections[kind].hint));
      fragment.append(empty);
    }
    for (const row of rows) {
      const title = row.brief?.goal || row.manifest?.name || experimentNames.get(row.experiment_id) || "Execution attempt";
      const entry = node("article", undefined, "list-row" + (state.selected[kind] === row.id ? " selected" : ""));
      const heading = node("div", undefined, "record-row-heading");
      heading.append(node("span", sections[kind].singular, "eyebrow"));
      if (row.state) heading.append(badge(row.state));
      const link = button(title, () => select(kind, row.id), "record-link");
      if (state.selected[kind] === row.id) link.setAttribute("aria-current", "true");
      entry.append(heading, link);
      const context = row.brief?.model?.model || row.manifest?.question;
      if (context) entry.append(node("p", context, "record-context"));
      const footer = node("div", undefined, "record-row-footer");
      const arrow = node("span", "↗", "record-arrow");
      arrow.setAttribute("aria-hidden", "true");
      footer.append(node("span", row.id, "mono"), arrow);
      entry.append(footer);
      fragment.append(entry);
    }
    target.replaceChildren(fragment);
    state.signatures["list-" + kind] = signature;
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
      model: modelConfig(),
      policy: $("brief-policy").value, max_experiments: number("brief-max-experiments", 2), max_model_calls: number("brief-max-calls", 12),
      max_output_tokens_per_call: $("brief-provider").value === "openai-codex" ? null : number("brief-tokens"), max_wall_seconds: number("brief-max-wall", 600), max_repairs: number("brief-repairs"),
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
  try { if (!state.overview || !document.hidden || state.overview.providers.codex.pending) await refresh(); } catch (exception) { error(exception.message); }
  setTimeout(poll, 3000);
}
poll();
