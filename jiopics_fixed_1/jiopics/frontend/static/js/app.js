/**
 * JIOPICS — AI Collage Studio
 * frontend/static/js/app.js
 *
 * FIXES:
 *   • maxEdge raised 1600→2400, quality 0.85→0.92 — sharper collage output
 *   • Preview and download both use the SAME base64 from Flask (PIL-generated)
 *   • Download button directly uses State.collageDataUrl — no re-render
 *   • Saliency ROI dot uses cx/cy from Flask response (not client-side estimate)
 */

"use strict";

// ═══════════════════════════════════════════════════════════════
//  COMPRESSION CONFIG
// ═══════════════════════════════════════════════════════════════
const COMPRESS = {
  maxEdge: 2400,        // raised from 1600 — more detail reaches Flask
  quality: 0.92,        // raised from 0.85 — less JPEG artefacts
  type:    "image/jpeg",
};

// ═══════════════════════════════════════════════════════════════
//  STATE
// ═══════════════════════════════════════════════════════════════
const State = {
  selectedTemplate: null,
  templatesMeta:    {},
  photos:           [],     // {name, b64, cx, cy, size}
  collageDataUrl:   null,   // PIL-generated base64 — single source of truth
  currentStep:      1,
};

// ═══════════════════════════════════════════════════════════════
//  DOM REFS
// ═══════════════════════════════════════════════════════════════
const $ = id => document.getElementById(id);
const DOM = {
  sections:      () => document.querySelectorAll(".section"),
  stepTabs:      () => document.querySelectorAll(".step-tab"),
  templateGrid:  $("templateGrid"),
  btnNext1:      $("btnNext1"),
  btnGenerate:   $("btnGenerate"),
  fileInput:     $("fileInput"),
  dropzone:      $("dropzone"),
  needPill:      $("needPill"),
  progressFill:  $("progressFill"),
  progressLabel: $("progressLabel"),
  progressRow:   $("progressRow"),
  photoSlots:    $("photoSlots"),
  slotsLabel:    $("slotsLabel"),
  procPanel:     $("procPanel"),
  logConsole:    $("logConsole"),
  resultPanel:   $("resultPanel"),
  collageImg:    $("collageImg"),
  statTemplate:  $("statTemplate"),
  statPhotos:    $("statPhotos"),
  statCanvas:    $("statCanvas"),
  statSlots:     $("statSlots"),
  timingValue:   $("timingValue"),
  roiRow:        $("roiRow"),
  toast:         $("toast"),
  toastIcon:     $("toastIcon"),
  toastMsg:      $("toastMsg"),
};

// ═══════════════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════════════
async function init() {
  await fetchTemplates();
  renderTemplateGrid();
  bindEvents();
  goStep(1);
}

// ═══════════════════════════════════════════════════════════════
//  FETCH TEMPLATES
// ═══════════════════════════════════════════════════════════════
async function fetchTemplates() {
  try {
    const resp = await fetch("/api/templates");
    State.templatesMeta = await resp.json();
  } catch {
    console.warn("Could not fetch templates — using fallback");
  }
}

// ═══════════════════════════════════════════════════════════════
//  RENDER TEMPLATE GRID
// ═══════════════════════════════════════════════════════════════
function renderTemplateGrid() {
  DOM.templateGrid.innerHTML = "";
  Object.entries(State.templatesMeta).forEach(([key, meta]) => {
    const card = document.createElement("div");
    card.className = "tmpl-card";
    card.dataset.key = key;
    card.innerHTML = `
      <div class="check-badge">✓</div>
      <div class="tmpl-preview">
        <img src="/assets/collage_templates/${meta.preview_img}"
             alt="${meta.name}" loading="lazy"
             onerror="this.parentElement.style.background='#E0E8F4'"/>
      </div>
      <div class="tmpl-info">
        <div class="tmpl-name">${meta.name}</div>
        <div class="tmpl-slots-badge">📷 ${meta.slots} Photos</div>
      </div>`;
    card.addEventListener("click", () => selectTemplate(card, key));
    DOM.templateGrid.appendChild(card);
  });
}

// ═══════════════════════════════════════════════════════════════
//  STEP NAVIGATION
// ═══════════════════════════════════════════════════════════════
function goStep(n) {
  if (n === 2 && !State.selectedTemplate) {
    showToast("❗", "Please select a template first");
    return;
  }
  if (n > State.currentStep + 1 && n !== 3) return;

  State.currentStep = n;

  DOM.sections().forEach((s, i) => {
    s.classList.toggle("active", i + 1 === n);
  });

  DOM.stepTabs().forEach((t, i) => {
    t.classList.remove("active", "done");
    const num = t.querySelector(".step-num");
    if (i + 1 < n)       { t.classList.add("done");   num.textContent = "✓"; }
    else if (i + 1 === n) { t.classList.add("active"); num.textContent = i + 1; }
    else                  {                             num.textContent = i + 1; }
  });

  if (n === 2) refreshUploadUI();
}

// ═══════════════════════════════════════════════════════════════
//  TEMPLATE SELECT
// ═══════════════════════════════════════════════════════════════
function selectTemplate(card, key) {
  document.querySelectorAll(".tmpl-card").forEach(c => c.classList.remove("selected"));
  card.classList.add("selected");
  State.selectedTemplate = key;
  State.photos = [];
  DOM.btnNext1.disabled = false;
  showToast("✅", `"${State.templatesMeta[key].name}" selected`);
}

// ═══════════════════════════════════════════════════════════════
//  UPLOAD UI
// ═══════════════════════════════════════════════════════════════
function getNeeded() {
  if (!State.selectedTemplate) return 0;
  return State.templatesMeta[State.selectedTemplate]?.slots ?? 0;
}

function refreshUploadUI() {
  const n      = getNeeded();
  const filled = State.photos.filter(Boolean).length;

  DOM.needPill.textContent = `📁  Needs ${n} photo${n !== 1 ? "s" : ""}`;
  DOM.needPill.classList.toggle("ready", filled >= n);
  DOM.progressRow.style.display = "flex";
  DOM.slotsLabel.style.display  = "block";
  DOM.progressFill.style.width  = `${(filled / n) * 100}%`;
  DOM.progressLabel.textContent = `${filled} / ${n}`;

  buildPhotoSlotUI(n);
  checkGenerateBtn();
}

function buildPhotoSlotUI(n) {
  DOM.photoSlots.innerHTML = "";
  for (let i = 0; i < n; i++) {
    const photo = State.photos[i];
    const wrap  = document.createElement("div");
    wrap.className = "photo-slot";

    const box = document.createElement("div");
    box.className = "slot-box";
    box.addEventListener("click", () => triggerSlotUpload(i));

    if (photo) {
      const img = document.createElement("img");
      img.src = photo.b64;
      box.appendChild(img);

      // ROI dot — uses saliency from Flask, not guessed client-side
      const dot = document.createElement("div");
      dot.className     = "slot-roi-dot";
      dot.style.display = "block";
      dot.style.left    = `${photo.cx * 100}%`;
      dot.style.top     = `${photo.cy * 100}%`;
      box.appendChild(dot);
    } else {
      box.innerHTML = `<span class="slot-icon">+</span><span>${"Slot " + (i + 1)}</span>`;
    }

    const rmv = document.createElement("button");
    rmv.className   = "slot-remove-btn";
    rmv.textContent = "✕";
    rmv.title       = "Remove photo";
    rmv.addEventListener("click", e => { e.stopPropagation(); removePhoto(i); });

    const lbl = document.createElement("div");
    lbl.className   = "slot-num-label";
    lbl.textContent = `Slot ${i + 1}`;

    wrap.appendChild(box);
    wrap.appendChild(rmv);
    wrap.appendChild(lbl);
    DOM.photoSlots.appendChild(wrap);
  }
}

function removePhoto(idx) {
  State.photos[idx] = null;
  refreshUploadUI();
}

function checkGenerateBtn() {
  const n      = getNeeded();
  const filled = State.photos.filter(Boolean).length;
  DOM.btnGenerate.disabled = filled < n;
}

// ═══════════════════════════════════════════════════════════════
//  CLIENT-SIDE IMAGE COMPRESSION
// ═══════════════════════════════════════════════════════════════
async function compressImage(file) {
  try {
    let bmp;
    if (typeof createImageBitmap === "function") {
      bmp = await createImageBitmap(file);
    } else {
      bmp = await new Promise((resolve, reject) => {
        const img = new Image();
        img.onload  = () => resolve(img);
        img.onerror = reject;
        img.src     = URL.createObjectURL(file);
      });
    }

    const srcW = bmp.width  || bmp.naturalWidth;
    const srcH = bmp.height || bmp.naturalHeight;

    const scale = Math.min(1, COMPRESS.maxEdge / Math.max(srcW, srcH));
    const dstW  = Math.round(srcW * scale);
    const dstH  = Math.round(srcH * scale);

    let canvas, ctx;
    if (typeof OffscreenCanvas !== "undefined") {
      canvas = new OffscreenCanvas(dstW, dstH);
      ctx    = canvas.getContext("2d");
    } else {
      canvas        = document.createElement("canvas");
      canvas.width  = dstW;
      canvas.height = dstH;
      ctx           = canvas.getContext("2d");
    }

    ctx.drawImage(bmp, 0, 0, dstW, dstH);
    if (typeof bmp.close === "function") bmp.close();

    let blob;
    if (canvas.convertToBlob) {
      blob = await canvas.convertToBlob({ type: COMPRESS.type, quality: COMPRESS.quality });
    } else {
      blob = await new Promise(r => canvas.toBlob(r, COMPRESS.type, COMPRESS.quality));
    }

    const b64 = await blobToDataURL(blob);
    return { b64, origSize: file.size, newSize: blob.size, width: dstW, height: dstH };

  } catch (e) {
    console.warn("Compression failed, falling back to raw upload:", e);
    const b64 = await readAsBase64(file);
    return { b64, origSize: file.size, newSize: file.size, width: 0, height: 0 };
  }
}

function blobToDataURL(blob) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload  = () => resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(blob);
  });
}

// ═══════════════════════════════════════════════════════════════
//  FILE INPUT HANDLING
// ═══════════════════════════════════════════════════════════════
let _targetSlot = undefined;

function triggerSlotUpload(idx) {
  _targetSlot = idx;
  DOM.fileInput.click();
}

DOM.fileInput.addEventListener("change", async function () {
  const files = Array.from(this.files);
  if (!files.length) return;

  const n = getNeeded();

  if (_targetSlot !== undefined) {
    const photo = await loadPhoto(files[0]);
    State.photos[_targetSlot] = { name: files[0].name, ...photo };
    _targetSlot = undefined;
  } else {
    const slots = [];
    const filesToLoad = [];
    for (const file of files) {
      const emptyIdx = State.photos.findIndex((p, i) => i < n && !p);
      if (emptyIdx === -1) break;
      State.photos[emptyIdx] = null;
      slots.push({ idx: emptyIdx, file });
      filesToLoad.push(file);
    }

    if (slots.length > 0) {
      const compressed = await Promise.all(filesToLoad.map(f => compressImage(f)));
      const b64List    = compressed.map(c => c.b64);
      const rois       = await fetchSaliencyBatch(b64List);

      slots.forEach(({ idx, file }, i) => {
        const { b64, newSize } = compressed[i];
        const { cx, cy }      = rois[i] || { cx: 0.5, cy: 0.45 };
        State.photos[idx] = { name: file.name, b64, cx, cy, size: newSize };
      });
    }
  }

  refreshUploadUI();
  this.value = "";
});

async function loadPhoto(file) {
  const { b64, origSize, newSize } = await compressImage(file);
  if (origSize && newSize) {
    const saved = ((1 - newSize / origSize) * 100).toFixed(0);
    console.log(`📷 ${file.name}: ${(origSize/1024).toFixed(0)}KB → ${(newSize/1024).toFixed(0)}KB  (-${saved}%)`);
  }
  const { cx, cy } = await fetchSaliency(b64);
  return { b64, cx, cy, size: newSize };
}

function readAsBase64(file) {
  return new Promise(resolve => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.readAsDataURL(file);
  });
}

async function fetchSaliency(b64) {
  try {
    const resp = await fetch("/api/saliency", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ image: b64 }),
    });
    if (!resp.ok) throw new Error();
    return await resp.json();
  } catch {
    return { cx: 0.5, cy: 0.45 };
  }
}

async function fetchSaliencyBatch(b64List) {
  try {
    const resp = await fetch("/api/saliency_batch", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ images: b64List }),
    });
    if (!resp.ok) throw new Error();
    const data = await resp.json();
    return data.results;
  } catch {
    return Promise.all(b64List.map(b64 => fetchSaliency(b64)));
  }
}

// ═══════════════════════════════════════════════════════════════
//  DRAG-AND-DROP
// ═══════════════════════════════════════════════════════════════
DOM.dropzone.addEventListener("click", () => {
  _targetSlot = undefined;
  DOM.fileInput.click();
});

DOM.dropzone.addEventListener("dragover",  e => { e.preventDefault(); DOM.dropzone.classList.add("drag-over"); });
DOM.dropzone.addEventListener("dragleave", ()  => DOM.dropzone.classList.remove("drag-over"));
DOM.dropzone.addEventListener("drop", e => {
  e.preventDefault();
  DOM.dropzone.classList.remove("drag-over");
  const dt = e.dataTransfer;
  if (!dt.files.length) return;
  _targetSlot = undefined;
  (async () => {
    const n     = getNeeded();
    const files = Array.from(dt.files);
    const slots = [];
    const filesToLoad = [];

    for (const file of files) {
      const emptyIdx = State.photos.findIndex((p, i) => i < n && !p);
      if (emptyIdx === -1) break;
      State.photos[emptyIdx] = null;
      slots.push({ idx: emptyIdx, file });
      filesToLoad.push(file);
    }

    if (slots.length > 0) {
      const compressed = await Promise.all(filesToLoad.map(f => compressImage(f)));
      const rois       = await fetchSaliencyBatch(compressed.map(c => c.b64));
      slots.forEach(({ idx, file }, i) => {
        const { b64, newSize } = compressed[i];
        const { cx, cy }      = rois[i] || { cx: 0.5, cy: 0.45 };
        State.photos[idx] = { name: file.name, b64, cx, cy, size: newSize };
      });
    }

    refreshUploadUI();
  })();
});

// ═══════════════════════════════════════════════════════════════
//  GENERATE COLLAGE
//  FIX: preview and download both use data.collage (PIL output)
//       No re-rendering, no second fetch — pixel-identical
// ═══════════════════════════════════════════════════════════════
async function generateCollage() {
  goStep(3);
  DOM.procPanel.classList.add("active");
  DOM.resultPanel.classList.remove("active");
  DOM.logConsole.innerHTML = "";

  const log = (cls, msg) => {
    const ts = new Date().toLocaleTimeString("en-IN", { hour12: false });
    DOM.logConsole.innerHTML +=
      `<div><span class="log-ts">[${ts}]</span>&nbsp;<span class="${cls}">${msg}</span></div>`;
    DOM.logConsole.scrollTop = DOM.logConsole.scrollHeight;
  };

  const needed = getNeeded();
  const images = State.photos.filter(Boolean).slice(0, needed).map(p => p.b64);

  const totalBytes = State.photos.filter(Boolean).slice(0, needed)
                       .reduce((s, p) => s + (p.size || 0), 0);
  log("log-warn", `Template: ${State.templatesMeta[State.selectedTemplate]?.name}`);
  log("log-warn", `Uploading ${images.length} image${images.length !== 1 ? "s" : ""} (~${(totalBytes / 1024 / 1024).toFixed(2)} MB)…`);

  const tFront = performance.now();

  try {
    const resp = await fetch("/api/generate", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        template: State.selectedTemplate,
        images,
        quality: 95,   // raised from 92 — better download quality
      }),
    });

    const data = await resp.json();
    const totalFront = ((performance.now() - tFront) / 1000).toFixed(2);

    if (!resp.ok) {
      log("log-err", `Error: ${data.error}`);
      return;
    }

    log("log-ok", `✓  ${data.slots} images processed in ${data.time_s}s (AI backend)`);
    log("log-ok", `✓  Total round-trip: ${totalFront}s`);
    data.rois.forEach(r => {
      log("log-ok", `   Slot ${r.slot}: ROI cx=${r.cx}% cy=${r.cy}%  (${r.time_ms}ms)`);
    });
    log("log-ok", `✓  Canvas: ${data.canvas}`);
    log("log-ok", `✓  Done — collage ready!`);

    // ── SINGLE SOURCE OF TRUTH ──────────────────────────────────
    // Store the PIL-generated base64 once.
    // Preview img AND download both use this exact same string.
    // Nothing is re-rendered, re-fetched, or re-encoded.
    State.collageDataUrl = data.collage;
    DOM.collageImg.src   = data.collage;   // preview = PIL output

    DOM.statTemplate.textContent = data.template;
    DOM.statPhotos.textContent   = `${data.slots} / ${data.slots}`;
    DOM.statCanvas.textContent   = data.canvas;
    DOM.statSlots.textContent    = data.slots;
    DOM.timingValue.textContent  = data.time_s;

    DOM.roiRow.innerHTML = data.rois.map(r =>
      `<div class="roi-pill">
         <span class="slot-n">Slot ${r.slot}</span>
         <span class="coords">cx:${r.cx}% cy:${r.cy}%</span>
         <span class="ms">${r.time_ms}ms</span>
       </div>`
    ).join("");

    setTimeout(() => {
      DOM.procPanel.classList.remove("active");
      DOM.resultPanel.classList.add("active");
      showToast("🎉", "Your collage is ready!");
    }, 500);

  } catch (err) {
    log("log-err", `Network error: ${err.message}`);
  }
}

// ═══════════════════════════════════════════════════════════════
//  DOWNLOAD
//  FIX: directly uses State.collageDataUrl (the PIL base64)
//       No canvas re-draw, no second fetch — byte-for-byte same as preview
// ═══════════════════════════════════════════════════════════════
function downloadCollage() {
  if (!State.collageDataUrl) {
    showToast("❗", "No collage to download yet");
    return;
  }
  const a    = document.createElement("a");
  a.href     = State.collageDataUrl;                              // same base64 as preview
  a.download = `jiopics_${State.selectedTemplate}_${Date.now()}.jpg`;
  document.body.appendChild(a);   // required for Firefox
  a.click();
  document.body.removeChild(a);
  showToast("⬇️", "Download started!");
}

// ═══════════════════════════════════════════════════════════════
//  RESTART
// ═══════════════════════════════════════════════════════════════
function restart() {
  State.selectedTemplate = null;
  State.photos           = [];
  State.collageDataUrl   = null;
  document.querySelectorAll(".tmpl-card").forEach(c => c.classList.remove("selected"));
  DOM.btnNext1.disabled = true;
  DOM.photoSlots.innerHTML = "";
  DOM.resultPanel.classList.remove("active");
  DOM.procPanel.classList.remove("active");
  goStep(1);
}

// ═══════════════════════════════════════════════════════════════
//  TOAST
// ═══════════════════════════════════════════════════════════════
let _toastTimer;
function showToast(icon, msg) {
  DOM.toastIcon.textContent = icon;
  DOM.toastMsg.textContent  = msg;
  DOM.toast.classList.add("visible");
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => DOM.toast.classList.remove("visible"), 3000);
}

// ═══════════════════════════════════════════════════════════════
//  EVENT BINDINGS
// ═══════════════════════════════════════════════════════════════
function bindEvents() {
  document.querySelectorAll(".step-tab").forEach((tab, i) => {
    tab.addEventListener("click", () => goStep(i + 1));
  });
}

// ── Boot ──
document.addEventListener("DOMContentLoaded", init);