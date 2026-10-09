"use strict";
const $ = id => document.getElementById(id);
const colors = ["#8b9eff", "#54c5bb", "#e3b46b", "#c393e6", "#df8294", "#79b7e8", "#a6c878", "#da9d73", "#80ccc7", "#b6a8ec"];
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

async function start() {
  try {
    const response = await fetch("/api/data");
    if (!response.ok) throw new Error("Could not load data");
    const data = await response.json();
    points = data.points; imageCount = data.imageCount;
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
