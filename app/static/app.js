"use strict";
const $ = id => document.getElementById(id);
const colors = ["#8b9eff", "#54c5bb", "#e3b46b", "#c393e6", "#df8294", "#79b7e8", "#a6c878", "#da9d73", "#80ccc7", "#b6a8ec"];
let neighbors = [], sheets = {};
let points = [], imageCount = 0, method = "kmeans_cluster", selected = 0;
const camera = {eye: {x: 1.5, y: 1.5, z: 1.05}};
const methodName = () => method === "kmeans_cluster" ? "KMeans" : "DBSCAN";
const colorFor = cluster => cluster === -1 ? "#999999" : colors[((cluster % colors.length) + colors.length) % colors.length];

function inspect(value) {
  const id = Number(value);
  if (String(value).trim() === "" || !Number.isInteger(id) || id < 0 || id >= imageCount) {
    $("input-error").textContent = `Enter an image ID from 0 to ${imageCount - 1}.`;
    return;
  }
  selected = id;
  renderNeighbors(id);
  $("input-error").textContent = "";
  $("image-id").value = id;
  $("previous").disabled = id === 0;
  $("next").disabled = id === imageCount - 1;
  const cluster = points.find(p => p.image_id === id)?.[method];
  $("preview").src = `/api/images/${id}`;
  $("preview").alt = `Dataset image ${id}`;
  $("preview").hidden = false;
  $("image-name").textContent = `Image ${String(id).padStart(4, "0")}`;
  $("cluster-badge").textContent = cluster === undefined ? "Not projected" : cluster === -1 ? "Noise / −1" : `Cluster ${cluster}`;
  $("cluster-badge").style.setProperty("--cluster-color", colorFor(cluster ?? -1));
}

function draw() {
  const clusters = [...new Set(points.map(p => p[method]))].sort((a, b) => a - b);
  $("cluster-count").textContent = clusters.filter(c => c !== -1).length;
  const noise = points.filter(p => p[method] === -1).length;
  $("cluster-description").textContent = `${methodName()} grouping${noise ? ` · ${noise} noise points` : ""}`;
  const traces = clusters.map(cluster => {
    const group = points.filter(p => p[method] === cluster);
    return {type: "scatter3d", mode: "markers", name: cluster === -1 ? "Noise" : `Cluster ${cluster}`,
      x: group.map(p => p.x), y: group.map(p => p.y), z: group.map(p => p.z),
      customdata: group.map(p => p.image_id), marker: {size: 3.5, opacity: .88, color: colorFor(cluster)},
      hovertemplate: "Image %{customdata}<br>" + (cluster === -1 ? "Noise" : `Cluster ${cluster}`) + "<extra></extra>"};
  });
  const axis = title => ({title: {text: title, font: {size: 10, color: "#888"}}, gridcolor: "#252525", zerolinecolor: "#353535", showbackground: false, tickfont: {size: 9, color: "#888"}, showspikes: false});
  return Plotly.react($("plot"), traces, {paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    margin: {l: 0, r: 0, t: 30, b: 0}, font: {family: "Inter, Segoe UI, sans-serif", color: "#aaa"},
    scene: {xaxis: axis("X"), yaxis: axis("Y"), zaxis: axis("Z"), camera: structuredClone(camera), bgcolor: "rgba(0,0,0,0)"},
    legend: {orientation: "h", x: .5, xanchor: "center", y: 0, font: {size: 9}, itemsizing: "constant"},
    uirevision: "preserve-camera", hoverlabel: {bgcolor: "#181818", bordercolor: "#444", font: {color: "#eee"}}},
    {responsive: true, displayModeBar: false, scrollZoom: true, displaylogo: false});
}

function imageTile(id, caption) {
  const button = document.createElement("button");
  button.className = "image-tile";
  button.type = "button";
  button.setAttribute("aria-label", `Inspect image ${id}${caption ? `, cosine similarity ${caption}` : ""}`);
  const image = document.createElement("img");
  image.src = `/api/images/${id}`;
  image.alt = `Image ${id}`;
  image.loading = "lazy";
  const label = document.createElement("span");
  label.textContent = `#${id}${caption ? ` · ${caption}` : ""}`;
  button.append(image, label);
  button.addEventListener("click", () => {
    inspect(id);
    $("image-id").focus({preventScroll: true});
    $("image-id").scrollIntoView({block: "center", behavior: "auto"});
  });
  return button;
}

function renderNeighbors(id) {
  const matches = neighbors[id] || [];
  $("similar-images").replaceChildren(...matches.map(match => imageTile(match.image_id, match.score.toFixed(2))));
  if (!matches.length) $("similar-images").textContent = "No comparable embeddings available.";
}

function renderSheets() {
  const sheet = sheets[method];
  $("contact-sheets").replaceChildren();
  if (!sheet) return;
  for (const group of sheet.groups) {
    const card = document.createElement("article");
    card.className = "contact-card";
    const title = document.createElement("h3");
    title.textContent = `Cluster ${group.cluster}`;
    title.style.color = colorFor(group.cluster);
    const count = document.createElement("span");
    count.textContent = `${group.count} images`;
    const header = document.createElement("div");
    header.className = "contact-header";
    header.append(title, count);
    const grid = document.createElement("div");
    grid.className = "thumbnail-grid";
    grid.append(...group.images.map(id => imageTile(id)));
    card.append(header, grid);
    $("contact-sheets").append(card);
  }
  $("contact-note").textContent = sheet.groups.length
    ? `${methodName()} · Representatives are nearest the cluster mean, not predicted labels.${sheet.noiseCount ? ` ${sheet.noiseCount} noise images are excluded.` : ""}`
    : `No clusters to summarize. ${methodName()} marks ${sheet.noiseCount} images as noise in the saved results.`;
}

async function start() {
  try {
    const response = await fetch("/api/data");
    if (!response.ok) throw new Error("Could not load data");
    const data = await response.json();
    points = data.points; imageCount = data.imageCount;
    neighbors = data.neighbors; sheets = data.sheets;
    renderSheets();
    if (!points.length || !imageCount) throw new Error("Empty collection");
    $("image-count").textContent = imageCount.toLocaleString();
    $("point-count").textContent = `${points.length.toLocaleString()} points`;
    $("id-range").textContent = `0 – ${imageCount - 1}`;
    $("image-id").max = imageCount - 1;
    await draw();
    $("plot-status").hidden = true;
    $("image-id").disabled = false;
    $("reset-view").disabled = false;
    document.querySelectorAll("[data-method]").forEach(button => {
      button.disabled = false;
      button.addEventListener("click", async () => {
        method = button.dataset.method;
        document.querySelectorAll("[data-method]").forEach(b => b.setAttribute("aria-pressed", String(b === button)));
        $("method-note").textContent = method === "kmeans_cluster" ? "Groups images into a fixed number of clusters based on similar features." : "Finds dense groups of similar images. Unassigned images are marked as noise.";
        inspect(selected);
        renderSheets();
        try { await draw(); } catch { $("plot-status").textContent = "Unable to render the plot. Please reload to try again."; $("plot-status").hidden = false; }
      });
    });
    inspect(0);
    $("plot").on("plotly_click", event => inspect(event.points[0].customdata));
    $("image-id").addEventListener("input", event => inspect(event.target.value));
    $("previous").addEventListener("click", () => inspect(selected - 1));
    $("next").addEventListener("click", () => inspect(selected + 1));
    $("reset-view").addEventListener("click", () => Plotly.relayout($("plot"), {"scene.camera": structuredClone(camera)}));
    $("preview").addEventListener("error", () => { $("preview").hidden = true; $("input-error").textContent = "Could not load this image. Select another image to retry."; });
  } catch (error) {
    $("plot-status").textContent = "Unable to load the explorer. Check that the app is running, then reload this page.";
    console.error(error);
  }
}
start();
let uploadFile = null, uploadUrl = null, uploadRequest = null, uploadVersion = 0;

/* ── Liquid glass indicator + morph transition helpers ── */
function updateIndicator(tabButton) {
  const list = tabButton.closest('.tab-list');
  if (!list) return;
  const listRect = list.getBoundingClientRect();
  const btnRect = tabButton.getBoundingClientRect();
  list.style.setProperty('--tab-indicator-left', `${btnRect.left - listRect.left}px`);
  list.style.setProperty('--tab-indicator-width', `${btnRect.width}px`);
}

let _currentTab = 'overview';

function selectTab(name, focus = false) {
  if (name === _currentTab) return;
  const outgoing = $(`${_currentTab}-panel`);
  const incoming = $(`${name}-panel`);

  /* Morph‑out the old panel */
  outgoing.classList.add('tab-exit');
  outgoing.addEventListener('animationend', function handler() {
    outgoing.removeEventListener('animationend', handler);
    outgoing.classList.remove('tab-exit');
    outgoing.hidden = true;
  }, { once: true });

  /* Update tab states */
  for (const view of ['overview', 'playground']) {
    const active = view === name;
    $(`${view}-tab`).classList.toggle('active', active);
    $(`${view}-tab`).setAttribute('aria-selected', String(active));
    $(`${view}-tab`).tabIndex = active ? 0 : -1;
  }

  /* Slide the liquid glass indicator */
  updateIndicator($(`${name}-tab`));

  /* Morph‑in the new panel */
  incoming.hidden = false;
  incoming.style.animation = 'none';
  incoming.offsetHeight; /* force reflow */
  incoming.style.animation = '';

  _currentTab = name;
  if (focus) $(`${name}-tab`).focus();
  if (name === 'overview' && $('plot').data && window.Plotly) requestAnimationFrame(() => Plotly.Plots.resize($('plot')));
}

/* Bind tab buttons */
for (const view of ['overview', 'playground']) {
  $(`${view}-tab`).addEventListener('click', () => selectTab(view));
  $(`${view}-tab`).addEventListener('keydown', event => {
    if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      selectTab(event.key === 'Home' ? 'overview' : event.key === 'End' ? 'playground' : view === 'overview' ? 'playground' : 'overview', true);
    }
  });
}

/* Set initial indicator position once layout is ready */
requestAnimationFrame(() => updateIndicator($('overview-tab')));
window.addEventListener('resize', () => updateIndicator($(`${_currentTab}-tab`)));

function uploadMessage(text, error = false) {
  $('upload-status').textContent = text;
  $('upload-status').classList.toggle('error', error);
}   

function resetUpload() {
  uploadVersion++;
  uploadRequest?.abort();
  uploadRequest = null;
  uploadFile = null;
  if (uploadUrl) URL.revokeObjectURL(uploadUrl);
  uploadUrl = null;
  $('upload-preview').removeAttribute('src');
  $('upload-file').value = '';
  $('upload-preview-wrap').hidden = true;
  $('upload-result').hidden = true;
  $('upload-empty').hidden = false;
  $('upload-matches').replaceChildren();
  $('upload-representatives').replaceChildren();
  $('analyze-upload').disabled = true;
  $('analyze-upload').textContent = 'Find my cluster ↗';
  $('clear-upload').disabled = true;
  uploadMessage('Choose an image to get started.');
}

async function chooseUpload(file) {
  resetUpload();
  if (!file) return;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
    uploadMessage('Choose a JPEG, PNG or WebP image.', true); return;
  }
  if (!file.size || file.size > 10 * 1024 * 1024) {
    uploadMessage('Choose an image smaller than 10 MB.', true); return;
  }
  const version = uploadVersion;
  uploadUrl = URL.createObjectURL(file);
  $('upload-preview').src = uploadUrl;
  try {
    await $('upload-preview').decode();
    if (version !== uploadVersion) return;
    if ($('upload-preview').naturalWidth * $('upload-preview').naturalHeight > 20_000_000) throw new Error('too large');
    uploadFile = file;
    $('upload-preview-wrap').hidden = false;
    $('upload-name').textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
    $('analyze-upload').disabled = false;
    $('clear-upload').disabled = false;
    uploadMessage('Ready when you are.');
  } catch {
    if (version !== uploadVersion) return;
    resetUpload();
    uploadMessage('Could not read this image. Choose a valid image with no more than 20 million pixels.', true);
  }
}

function resultTile(id, score) {
  const tile = document.createElement('div');
  tile.className = 'image-tile';
  const img = document.createElement('img');
  img.src = `/api/images/${id}`;
  img.alt = `Dataset image ${id}`;
  const caption = document.createElement('span');
  caption.textContent = `#${id}${score === undefined ? '' : ` · ${score.toFixed(2)}`}`;
  tile.append(img, caption);
  return tile;
}

$('upload-file').addEventListener('change', event => chooseUpload(event.target.files[0]));
$('clear-upload').addEventListener('click', resetUpload);
for (const name of ['dragenter', 'dragover']) $('drop-zone').addEventListener(name, event => {
  event.preventDefault(); $('drop-zone').classList.add('dragging');
});
for (const name of ['dragleave', 'drop']) $('drop-zone').addEventListener(name, event => {
  event.preventDefault(); $('drop-zone').classList.remove('dragging');
});
$('drop-zone').addEventListener('drop', event => chooseUpload(event.dataTransfer.files[0]));
$('analyze-upload').addEventListener('click', async () => {
  if (!uploadFile || uploadRequest) return;
  const version = uploadVersion;
  uploadRequest = new AbortController();
  $('analyze-upload').disabled = true;
  $('analyze-upload').textContent = 'Finding connections…';
  $('upload-result').hidden = true;
  $('upload-empty').hidden = false;
  uploadMessage('Extracting visual features and finding your cluster. The first analysis may take a little longer.');
  try {
    const response = await fetch('/api/playground/predict', {
      method: 'POST', headers: {'Content-Type': uploadFile.type}, body: uploadFile, signal: uploadRequest.signal
    });
    const result = await response.json();
    if (version !== uploadVersion) return;
    if (!response.ok) throw new Error(result.error || 'Analysis failed. Please try again.');
    $('upload-cluster').textContent = `Cluster ${result.cluster}`;
    $('upload-cluster').style.color = colorFor(result.cluster);
    $('upload-cluster-size').textContent = `${result.clusterSize} dataset images`;
    $('upload-matches').replaceChildren(...result.neighbors.map(match => resultTile(match.image_id, match.score)));
    $('upload-representatives').replaceChildren(...result.representatives.map(id => resultTile(id)));
    $('upload-empty').hidden = true;
    $('upload-result').hidden = false;
    uploadMessage(`Analysis complete · ${result.width} × ${result.height} pixels. Try another image whenever you like.`);
  } catch (error) {
    if (version === uploadVersion && error.name !== 'AbortError') uploadMessage(error.message || 'Could not reach the server. Please try again.', true);
  } finally {
    if (version === uploadVersion) {
      uploadRequest = null;
      $('analyze-upload').disabled = !uploadFile;
      $('analyze-upload').textContent = 'Find my cluster ↗';
    }
  }
});
