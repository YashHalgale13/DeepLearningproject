/* ═══════════════════════════════════════════════════════════
   ML DASHBOARD — Vanilla JS
════════════════════════════════════════════════════════════ */

const MODEL_META = {
  mnist:     { num: "01", name: "MNIST DIGIT CLASSIFIER",  type: "image",          badge: "IMAGE",  desc: "ANN — classifies handwritten digits 0–9" },
  catdog:    { num: "02", name: "CAT VS DOG CLASSIFIER",   type: "image",          badge: "IMAGE",  desc: "MobileNet CNN — binary image classifier" },
  emotion:   { num: "03", name: "EMOTION DETECTION",       type: "image",          badge: "IMAGE",  desc: "CNN — detects 7 facial emotions" },
  sentiment: { num: "04", name: "SENTIMENT ANALYSIS",      type: "text",           badge: "TEXT",   desc: "RNN — IMDB positive / negative sentiment" },
  caption:   { num: "05", name: "IMAGE CAPTIONING",        type: "image",          badge: "IMAGE",  desc: "VGG16 + LSTM — generates image captions" },
  pos:       { num: "06", name: "POS TAGGER",              type: "text",           badge: "TEXT",   desc: "BiLSTM — Penn Treebank POS tagging" },
  lstm:      { num: "07", name: "LSTM TEXT PREDICTOR",     type: "text_multi",     badge: "TEXT",   desc: "LSTM — next-word sequence prediction" },
  sms:       { num: "08", name: "SMS SPAM DETECTOR",       type: "text",           badge: "TEXT",   desc: "GRU — spam vs ham classifier" },
};

const INPUT_CONFIG = {
  mnist:     { label: "UPLOAD HANDWRITTEN DIGIT IMAGE",  placeholder: "" },
  catdog:    { label: "UPLOAD CAT OR DOG IMAGE",         placeholder: "" },
  emotion:   { label: "UPLOAD FACE IMAGE",               placeholder: "" },
  sentiment: { label: "ENTER MOVIE REVIEW",              placeholder: "Type a movie review here..." },
  caption:   { label: "UPLOAD IMAGE TO CAPTION",         placeholder: "" },
  pos:       { label: "ENTER SENTENCE",                  placeholder: "The quick brown fox jumps over the lazy dog" },
  lstm:      { label: "ENTER SEED TEXT",                 placeholder: "Once upon a time" },
  sms:       { label: "ENTER SMS MESSAGE",               placeholder: "Congratulations! You've won a free prize..." },
};

let currentModel = null;
let selectedFile  = null;

// ─────────────────────────────────────────────────────────────
// Model selection
// ─────────────────────────────────────────────────────────────

function selectModel(modelId) {
  currentModel = modelId;
  selectedFile = null;

  // Sidebar active state
  document.querySelectorAll(".model-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.model === modelId);
  });

  const meta = MODEL_META[modelId];
  const cfg  = INPUT_CONFIG[modelId];

  // Show panel, hide welcome
  document.getElementById("welcome-state").classList.add("hidden");
  document.getElementById("model-panel").classList.remove("hidden");

  // Panel header
  document.getElementById("panel-num").textContent   = meta.num;
  document.getElementById("panel-title").textContent = meta.name;
  document.getElementById("panel-desc").textContent  = meta.desc;
  document.getElementById("panel-type-badge").textContent = meta.badge;

  // Show correct input block
  document.getElementById("image-input").classList.add("hidden");
  document.getElementById("text-input").classList.add("hidden");
  document.getElementById("text-multi-input").classList.add("hidden");
  document.getElementById("text-attention-input").classList.add("hidden");

  if (meta.type === "image") {
    document.getElementById("image-input").classList.remove("hidden");
    document.getElementById("upload-text").textContent = cfg.label;
    clearImage();
  } else if (meta.type === "text") {
    document.getElementById("text-input").classList.remove("hidden");
    document.getElementById("text-label").textContent = cfg.label;
    document.getElementById("text-area").placeholder  = cfg.placeholder;
    document.getElementById("text-area").value = "";
  } else if (meta.type === "text_multi") {
    document.getElementById("text-multi-input").classList.remove("hidden");
    document.getElementById("text-multi-label").textContent = cfg.label;
    document.getElementById("text-multi-area").placeholder  = cfg.placeholder;
    document.getElementById("text-multi-area").value = "";
    document.getElementById("num-words").value = "5";
  } else if (meta.type === "text_attention") {
    document.getElementById("text-attention-input").classList.remove("hidden");
    document.getElementById("text-attention-label").textContent = cfg.label;
    document.getElementById("text-attention-area").placeholder  = cfg.placeholder;
    document.getElementById("text-attention-area").value = "";
    populateBertSelects();
  }

  // Reset output
  hideOutput();
  hideError();

  // Footer
  document.getElementById("active-model-footer").textContent = meta.name;
}

// ─────────────────────────────────────────────────────────────
// File handling
// ─────────────────────────────────────────────────────────────

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (!file) return;
  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("image-preview").src = e.target.result;
    document.getElementById("image-preview-wrap").classList.remove("hidden");
    document.getElementById("image-label").style.display = "none";
  };
  reader.readAsDataURL(file);
  hideOutput();
  hideError();
}

function clearImage() {
  selectedFile = null;
  document.getElementById("file-input").value = "";
  document.getElementById("image-preview").src = "";
  document.getElementById("image-preview-wrap").classList.add("hidden");
  document.getElementById("image-label").style.display = "";
  hideOutput();
  hideError();
}

// Drag & drop support
document.addEventListener("DOMContentLoaded", () => {
  const label = document.getElementById("image-label");
  if (!label) return;

  label.addEventListener("dragover", (e) => {
    e.preventDefault();
    label.style.background = "#000";
    label.style.color = "#fff";
  });

  label.addEventListener("dragleave", () => {
    label.style.background = "";
    label.style.color = "";
  });

  label.addEventListener("drop", (e) => {
    e.preventDefault();
    label.style.background = "";
    label.style.color = "";
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      selectedFile = file;
      const reader = new FileReader();
      reader.onload = (ev) => {
        document.getElementById("image-preview").src = ev.target.result;
        document.getElementById("image-preview-wrap").classList.remove("hidden");
        document.getElementById("image-label").style.display = "none";
      };
      reader.readAsDataURL(file);
    }
  });
});

// ─────────────────────────────────────────────────────────────
// Predict
// ─────────────────────────────────────────────────────────────

async function runPredict() {
  if (!currentModel) return;

  hideOutput();
  hideError();
  setLoading(true);

  try {
    const meta = MODEL_META[currentModel];
    let response;

    if (meta.type === "image") {
      if (!selectedFile) {
        showError("NO IMAGE SELECTED. PLEASE UPLOAD AN IMAGE FIRST.");
        setLoading(false);
        return;
      }
      const form = new FormData();
      form.append("file", selectedFile);
      response = await fetch(`/predict/${currentModel}`, { method: "POST", body: form });

    } else if (meta.type === "text") {
      const text = document.getElementById("text-area").value.trim();
      if (!text) {
        showError("NO TEXT ENTERED. PLEASE TYPE SOMETHING FIRST.");
        setLoading(false);
        return;
      }
      response = await fetch(`/predict/${currentModel}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });

    } else if (meta.type === "text_multi") {
      const text = document.getElementById("text-multi-area").value.trim();
      const num  = parseInt(document.getElementById("num-words").value) || 5;
      if (!text) {
        showError("NO SEED TEXT ENTERED. PLEASE TYPE SOMETHING FIRST.");
        setLoading(false);
        return;
      }
      response = await fetch(`/predict/${currentModel}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, num_words: num })
      });

    } else if (meta.type === "text_attention") {
      const text  = document.getElementById("text-attention-area").value.trim();
      const layer = parseInt(document.getElementById("bert-layer-select").value) || 8;
      const head  = parseInt(document.getElementById("bert-head-select").value)  || 2;
      if (!text) {
        showError("NO TEXT ENTERED. PLEASE PASTE A NEWS ARTICLE FIRST.");
        setLoading(false);
        return;
      }
      if (text.length < 10) {
        showError("TEXT TOO SHORT. PLEASE ENTER AT LEAST 10 CHARACTERS.");
        setLoading(false);
        return;
      }
      response = await fetch(`/predict/${currentModel}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, layer, head })
      });
    }

    const data = await response.json();

    if (!response.ok || data.error) {
      showError("ERROR: " + (data.error || response.statusText));
      setLoading(false);
      return;
    }

    renderOutput(currentModel, data);

  } catch (err) {
    showError("NETWORK ERROR: " + err.message);
  }

  setLoading(false);
}

// ─────────────────────────────────────────────────────────────
// Render output
// ─────────────────────────────────────────────────────────────

function renderOutput(modelId, data) {
  // Hide all result blocks first
  document.getElementById("result-standard").classList.add("hidden");
  document.getElementById("result-caption").classList.add("hidden");
  document.getElementById("result-pos").classList.add("hidden");
  document.getElementById("result-lstm").classList.add("hidden");
  document.getElementById("result-image").classList.add("hidden");
  document.getElementById("result-fakenews").classList.add("hidden");

  document.getElementById("output-section").classList.remove("hidden");

  if (modelId === "caption") {
    document.getElementById("result-caption").classList.remove("hidden");
    document.getElementById("caption-text").textContent   = data.caption || "—";
    document.getElementById("caption-method").textContent = data.method  || "—";
    if (data.image) showResultImage(data.image);

  } else if (modelId === "pos") {
    document.getElementById("result-pos").classList.remove("hidden");
    document.getElementById("pos-count").textContent = data.count + " TOKENS";
    renderPosTokens(data.tokens || []);

  } else if (modelId === "lstm") {
    document.getElementById("result-lstm").classList.remove("hidden");
    renderLstmWords(data.predictions || []);
    document.getElementById("lstm-full").textContent = data.completed_text || "—";

  } else {
    // Standard: label + confidence
    document.getElementById("result-standard").classList.remove("hidden");

    let label = data.prediction || data.sentiment || data.label || data.caption || "—";
    let conf  = data.confidence;

    document.getElementById("result-value").textContent = label;

    if (conf !== undefined && conf !== null) {
      document.getElementById("conf-row").style.display = "";
      document.getElementById("conf-bar-wrap").style.display = "";
      document.getElementById("conf-value").textContent = conf + "%";
      document.getElementById("conf-bar").style.width   = conf + "%";
    } else {
      document.getElementById("conf-row").style.display = "none";
      document.getElementById("conf-bar-wrap").style.display = "none";
    }

    if (data.image) showResultImage(data.image);
  }

  // Fake News — special rich output
  if (modelId === "fakenews") {
    document.getElementById("result-fakenews").classList.remove("hidden");
    renderFakeNews(data);
  }
}

function renderPosTokens(tokens) {
  const container = document.getElementById("pos-tokens");
  container.innerHTML = "";
  tokens.forEach(t => {
    const el = document.createElement("div");
    el.className = "pos-token";
    el.title = t.description;
    el.innerHTML = `
      <span class="pos-word">${escHtml(t.word)}</span>
      <span class="pos-tag">${escHtml(t.tag)}</span>
      <span class="pos-conf">${t.confidence}%</span>
    `;
    container.appendChild(el);
  });
}

function renderLstmWords(words) {
  const container = document.getElementById("lstm-words");
  container.innerHTML = "";
  words.forEach((w, i) => {
    const el = document.createElement("span");
    el.className = "lstm-word";
    el.textContent = w;
    container.appendChild(el);
  });
}

function showResultImage(src) {
  document.getElementById("result-img").src = src;
  document.getElementById("result-image").classList.remove("hidden");
}

// ─────────────────────────────────────────────────────────────
// UI helpers
// ─────────────────────────────────────────────────────────────

function setLoading(on) {
  const btn  = document.getElementById("predict-btn");
  const text = document.getElementById("btn-text");
  if (on) {
    btn.classList.add("loading");
    btn.disabled = true;
    text.textContent = "PREDICTING...";
  } else {
    btn.classList.remove("loading");
    btn.disabled = false;
    text.textContent = "PREDICT";
  }
}

function showError(msg) {
  document.getElementById("error-msg").textContent = msg;
  document.getElementById("error-box").classList.remove("hidden");
}

function hideError() {
  document.getElementById("error-box").classList.add("hidden");
}

function hideOutput() {
  document.getElementById("output-section").classList.add("hidden");
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// ─────────────────────────────────────────────────────────────
// Health check on load
// ─────────────────────────────────────────────────────────────

window.addEventListener("load", async () => {
  try {
    const res = await fetch("/health");
    if (res.ok) {
      document.getElementById("status-dot").style.color = "#00ff41";
    } else {
      document.getElementById("status-dot").style.color = "#ff2222";
    }
  } catch {
    document.getElementById("status-dot").style.color = "#ff2222";
  }
});

// ─────────────────────────────────────────────────────────────
// BERT layer / head dropdowns
// ─────────────────────────────────────────────────────────────

function populateBertSelects() {
  const layerSel = document.getElementById("bert-layer-select");
  const headSel  = document.getElementById("bert-head-select");

  // Only populate once
  if (layerSel.options.length > 0) return;

  for (let i = 0; i < 12; i++) {
    const lo = document.createElement("option");
    lo.value = i;
    lo.textContent = `LAYER ${i}`;
    if (i === 8) lo.selected = true;
    layerSel.appendChild(lo);

    const ho = document.createElement("option");
    ho.value = i;
    ho.textContent = `HEAD ${i}`;
    if (i === 2) ho.selected = true;
    headSel.appendChild(ho);
  }
}

// ─────────────────────────────────────────────────────────────
// Fake News render
// ─────────────────────────────────────────────────────────────

function renderFakeNews(data) {
  const { prediction, confidence, tokens, attention_matrix, selected_layer, selected_head } = data;

  const isFake  = prediction === "Fake";
  const labelEl = document.getElementById("fakenews-label");
  const boxEl   = document.getElementById("fakenews-verdict");

  labelEl.textContent = prediction;
  labelEl.className   = "fakenews-label " + (isFake ? "fake" : "real");
  boxEl.className     = "fakenews-verdict "  + (isFake ? "fake" : "real");

  document.getElementById("fakenews-conf-value").textContent = confidence + "%";
  document.getElementById("fakenews-conf-bar").style.width   = confidence + "%";
  document.getElementById("fakenews-conf-bar").style.background =
    isFake ? "#ff2222" : "#00ff41";

  document.getElementById("fakenews-meta-layer").textContent  = selected_layer;
  document.getElementById("fakenews-meta-head").textContent   = selected_head;
  document.getElementById("fakenews-meta-tokens").textContent = tokens.length;

  drawAttentionHeatmap("fakenews-canvas", tokens, attention_matrix);
}

// ─────────────────────────────────────────────────────────────
// Attention heatmap canvas renderer
// ─────────────────────────────────────────────────────────────

function drawAttentionHeatmap(canvasId, tokens, matrix) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  const n     = tokens.length;
  if (n === 0) return;

  const CELL  = 22;   // px per cell
  const LABEL = 110;  // px for token labels
  const PAD   = 4;

  const totalW = LABEL + n * CELL + PAD;
  const totalH = LABEL + n * CELL + PAD;

  canvas.width  = totalW;
  canvas.height = totalH;

  ctx.clearRect(0, 0, totalW, totalH);

  // Background
  ctx.fillStyle = "#0a0a0a";
  ctx.fillRect(0, 0, totalW, totalH);

  // Draw cells
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      const val = matrix[row][col];
      ctx.fillStyle = attnValueToColor(val);
      ctx.fillRect(
        LABEL + col * CELL,
        LABEL + row * CELL,
        CELL - 1,
        CELL - 1
      );
    }
  }

  // Column labels (top, rotated)
  ctx.save();
  ctx.font = "bold 9px 'Courier New', monospace";
  ctx.fillStyle = "#888";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (let i = 0; i < n; i++) {
    const x = LABEL + i * CELL + CELL / 2;
    const y = LABEL - 4;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(truncAttnToken(tokens[i]), 0, 0);
    ctx.restore();
  }
  ctx.restore();

  // Row labels (left)
  ctx.font = "bold 9px 'Courier New', monospace";
  ctx.fillStyle = "#888";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i < n; i++) {
    const y = LABEL + i * CELL + CELL / 2;
    ctx.fillText(truncAttnToken(tokens[i]), LABEL - 4, y);
  }

  // Axis labels
  ctx.font = "10px 'Courier New', monospace";
  ctx.fillStyle = "#555";
  ctx.textAlign = "center";
  ctx.fillText("KEY (ATTENDED TO) →", LABEL + (n * CELL) / 2, 10);
  ctx.save();
  ctx.translate(10, LABEL + (n * CELL) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("QUERY (ATTENDING FROM) →", 0, 0);
  ctx.restore();
}

/**
 * Map [0,1] → color:
 *   0.0 → near-black  (#0a0a0a)
 *   0.5 → dark green  (#003300)
 *   1.0 → bright green (#00ff41)
 * Matches the dashboard's green-on-black brutalist theme.
 */
function attnValueToColor(val) {
  val = Math.max(0, Math.min(1, val));
  const r = Math.round(0   + val * 0);
  const g = Math.round(10  + val * (255 - 10));
  const b = Math.round(10  + val * (65  - 10));
  return `rgb(${r},${g},${b})`;
}

function truncAttnToken(token, max = 8) {
  if (token.length <= max) return token;
  return token.slice(0, max - 1) + "…";
}
