// The page of `spintrack run` (a preview of the run) and of `spintrack gui` (the same
// view, with controls): it polls the server's state and draws the frame with the ball,
// the trail and the axes over it, the tracking window, the path, the map and traces.

const token = new URLSearchParams(location.search).get("token") || "";
export const SVG = "http://www.w3.org/2000/svg";
export const $ = (id) => document.getElementById(id);

export function api(path, query = {}) {
  return `api/${path}?${new URLSearchParams({ token, ...query })}`;
}

export async function post(path, body = {}) {
  const response = await fetch(api(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.detail || response.statusText);
    error.status = response.status;
    throw error;
  }
  return data;
}

export function svg(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

const points = (pts) => pts.map(([x, y]) => `${x},${y}`).join(" ");
const TRACE_SECONDS = 10;
const KEEP = 8192; // per-frame numbers kept here
const TRAIL_BANDS = 6;

// What the gui module hooks into: called with each state, and the overlay's layers.
export const page = {
  state: null,
  listeners: [],
  overlayTop: null, // an <g> the gui draws its handles in, above the run's drawing
};

const local = {
  run: null,
  since: -1,
  logSince: -1,
  series: null,
  alive: true,
  gui: false,
  last: null,
};

function resetSeries() {
  local.series = { frame: [], ts: [], forward: [], side: [], turn: [], cost: [] };
  local.since = -1;
}
resetSeries();

// ----- images: fetched as blobs, so a frame is never loaded twice -----
const loading = new Set();
async function loadImage(img, kind, version) {
  if (img.dataset.v === String(version) || loading.has(kind)) return;
  loading.add(kind);
  try {
    const response = await fetch(api(`image/${kind}`, { v: version }));
    if (!response.ok) return;
    const url = URL.createObjectURL(await response.blob());
    const old = img.src;
    img.src = url;
    img.dataset.v = String(version);
    if (old.startsWith("blob:")) URL.revokeObjectURL(old);
  } catch {
    // the next state retries
  } finally {
    loading.delete(kind);
  }
}

// ----- header -----
const basename = (p) => (p ? String(p).split(/[\\/]/).pop() : "");
function clock(seconds) {
  const s = Math.round(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

function drawHeader(s) {
  const v = s.video || {};
  const count = v.count > 1 ? ` (${v.index + 1}/${v.count})` : "";
  $("video").textContent = basename(v.name) + count;
  $("video").title = v.name || "";
  const chip = $("state");
  const label = s.gui?.paused && s.state === "tracking" ? "paused" : s.state;
  chip.textContent = label;
  chip.className = `chip ${label}`;
  $("message").textContent = s.message || "";
  const p = s.progress;
  let stats = "";
  let fraction = 0;
  if (p) {
    const parts = [];
    if (p.total) {
      fraction = Math.min(p.frames / p.total, 1);
      parts.push(`${p.frames}/${p.total} frames`);
      if (s.state === "tracking" && p.fps > 0 && s.mode === "run") {
        parts.push(`${clock(Math.max(p.total - p.frames, 0) / p.fps)} left`);
      }
    } else {
      parts.push(`${p.frames} frames`);
    }
    parts.push(`${p.fps.toFixed(0)} fps`);
    parts.push(`${p.dropped} dropped`);
    stats = parts.join(" \u00b7 ");
  }
  $("stats").textContent = stats;
  $("bar").style.width = `${100 * (s.state === "done" ? 1 : fraction)}%`;
  const running = s.state === "tracking" || s.state === "preparing";
  $("stop").hidden = !(running && (s.mode === "run" || s.gui?.full_run));
  document.title = `${basename(v.name) || "spintrack"} \u00b7 ${label}`;
}

// ----- the frame and what is drawn over it -----
const overlay = $("overlay");
const runLayer = svg("g", {}, overlay);
page.overlayTop = svg("g", {}, overlay);

function trailColor(band) {
  // Oldest dark blue to newest cyan, as the debug video draws it.
  const t = band / (TRAIL_BANDS - 1);
  const mix = (a, b) => Math.round(a + (b - a) * t);
  return `rgb(${mix(0, 46)},${mix(80, 230)},${mix(143, 255)})`;
}

function drawOverlay(s) {
  const o = s.capture?.overlay;
  runLayer.replaceChildren();
  if (!o) return;
  for (const poly of o.ignore) svg("polygon", { class: "ignore", points: points(poly) }, runLayer);
  svg("polygon", { class: "outline", points: points(o.outline) }, runLayer);
  // The trail, in runs of points the camera sees, colored by age.
  const n = o.trail.length;
  for (let band = 0; band < TRAIL_BANDS; band++) {
    const lo = Math.floor((band * n) / TRAIL_BANDS);
    const hi = Math.min(n, Math.floor(((band + 1) * n) / TRAIL_BANDS) + 1);
    let run = [];
    const flush = () => {
      if (run.length > 1) {
        svg("polyline", { class: "trail", stroke: trailColor(band), points: points(run) }, runLayer);
      }
      run = [];
    };
    for (let i = lo; i < hi; i++) {
      if (o.trail[i]) run.push(o.trail[i]);
      else flush();
    }
    flush();
  }
  const contact = o.trail[n - 1];
  if (contact) svg("circle", { class: "contact", cx: contact[0], cy: contact[1], r: 3 }, runLayer);
  if ($("show-axes").checked) drawAxes(o);
}

// The ball's axes from its center, and the animal's from where it stands on the ball.
function drawAxes(o) {
  const [cx, cy] = o.center;
  const [ox, oy] = o.axes.contact;
  const colors = ["var(--x)", "var(--y)", "var(--z)"];
  if (o.axes.ball) {
    o.axes.ball.forEach(([x, y], i) => {
      svg("line", { class: "axis ball", x1: cx, y1: cy, x2: x, y2: y, stroke: colors[i], "stroke-opacity": 0.55 }, runLayer);
    });
  }
  o.axes.animal.forEach(([x, y], i) => {
    svg("line", { class: "axis", x1: ox, y1: oy, x2: x, y2: y, stroke: colors[i] }, runLayer);
    const text = svg("text", { x: x + 4, y: y + 4, fill: colors[i] }, runLayer);
    text.textContent = "xyz"[i];
  });
}

function drawFrame(s) {
  const v = s.video || {};
  if (v.width && v.height) {
    $("stage").style.aspectRatio = `${v.width} / ${v.height}`;
    overlay.setAttribute("viewBox", `0 0 ${v.width} ${v.height}`);
  }
  const images = s.capture?.images || {};
  const targets = { frame: $("frame"), window: $("window"), map: $("map"), lighting: document.querySelector(".tile.lighting img") };
  for (const [kind, version] of Object.entries(images)) {
    if (targets[kind]) loadImage(targets[kind], kind, version);
  }
  for (const [kind, img] of Object.entries(targets)) {
    if (img && !(kind in images) && img.src) {
      img.removeAttribute("src");
      delete img.dataset.v;
    }
  }
  $("placeholder").hidden = "frame" in images;
  $("placeholder").textContent = s.state === "preparing" ? s.message || "preparing" : "waiting for the first frame";
  const c = s.capture;
  if (c?.overlay) $("coverage").textContent = `${Math.round(100 * c.overlay.coverage)}% seen`;
  const last = s.last;
  $("frame-info").textContent = last
    ? `frame ${last.frame} \u00b7 ${last.ok ? `${last.source}, ${last.iterations} iterations` : "dropped"}`
    : "";
}

// ----- the map's net: labels per face, the lighting in a free tile -----
let netBuilt = false;
function buildNet(net) {
  if (netBuilt || !net) return;
  netBuilt = true;
  const [rows, cols] = net.shape;
  const box = $("net");
  const place = (div, r, c) => {
    div.style.left = `${(100 * c) / cols}%`;
    div.style.top = `${(100 * r) / rows}%`;
    div.style.width = `${100 / cols}%`;
    div.style.height = `${100 / rows}%`;
    box.appendChild(div);
  };
  const used = new Set();
  for (const [r, c, name] of net.labels) {
    const div = document.createElement("div");
    div.className = "tile";
    div.textContent = name;
    place(div, r, c);
    used.add(`${r},${c}`);
  }
  // The lighting field goes where the debug video puts it: the top-right free tile.
  for (let c = cols - 1; c >= 0; c--) {
    if (!used.has(`0,${c}`)) {
      const div = document.createElement("div");
      div.className = "tile lighting";
      div.title = "the lighting the tracker divides out (dark: shadow)";
      div.appendChild(document.createElement("img"));
      place(div, 0, c);
      break;
    }
  }
}

// ----- the path -----
function drawPath(s) {
  const svgPath = $("path");
  svgPath.replaceChildren();
  const pts = s.path?.points || [];
  if (pts.length < 2) return;
  // x forward is up, y left is left.
  let [x0, x1, y0, y1] = [Infinity, -Infinity, Infinity, -Infinity];
  for (const [x, y] of pts) {
    x0 = Math.min(x0, x); x1 = Math.max(x1, x);
    y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  }
  const span = Math.max(x1 - x0, y1 - y0, 0.2);
  const pad = 0.08 * span;
  const cx = -(y0 + y1) / 2;
  const cy = -(x0 + x1) / 2;
  const half = span / 2 + pad;
  svgPath.setAttribute("viewBox", `${cx - half} ${cy - half} ${2 * half} ${2 * half}`);
  svg("polyline", { points: pts.map(([x, y]) => `${-y},${-x}`).join(" ") }, svgPath);
  const [hx, hy] = pts[pts.length - 1];
  const r = half / 40;
  svg("circle", { class: "head", cx: -hy, cy: -hx, r }, svgPath);
  const heading = s.last?.heading;
  if (heading != null) {
    const len = half / 8;
    svg("line", {
      class: "heading",
      x1: -hy, y1: -hx,
      x2: -hy - len * Math.sin(heading), y2: -hx - len * Math.cos(heading),
    }, svgPath);
  }
  $("path-scale").textContent = `${span.toPrecision(2)} ball radii across`;
}

// ----- traces -----
function takeSeries(s) {
  const incoming = s.series;
  if (!incoming || incoming.end === undefined) return;
  const keys = Object.keys(local.series);
  const skip = Math.max(local.since + 1 - incoming.start, 0);
  for (const k of keys) {
    const values = incoming[k].slice(skip);
    local.series[k].push(...values);
    const extra = local.series[k].length - KEEP;
    if (extra > 0) local.series[k].splice(0, extra);
  }
  local.since = incoming.end - 1;
}

const nice = (v) => (v >= 100 ? v.toFixed(0) : v.toPrecision(2));

function median(values) {
  const v = values.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  return v.length ? v[Math.floor(v.length / 2)] : NaN;
}

function drawTraces() {
  const canvas = $("traces");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement);
  const color = (name) => css.getPropertyValue(name).trim();
  const S = local.series;
  const n = S.ts.length;
  const rows = [
    { label: "forward, side (rad/s)", keys: [["forward", "--forward"], ["side", "--side"]], scale: 1 },
    { label: "turn (deg/s)", keys: [["turn", "--turn"]], scale: 180 / Math.PI },
    { label: "solver cost", keys: [["cost", "--cost"]], scale: 1, positive: true },
  ];
  const left = 8;
  const right = w - 8;
  const top = 6;
  const rowH = (h - top - 18) / rows.length;
  g.font = "11px system-ui, sans-serif";
  if (n < 2) {
    g.fillStyle = color("--muted");
    g.fillText("traces appear once frames are tracked", left + 4, top + 14);
    return;
  }
  const tEnd = S.ts[n - 1];
  let first = n - 1;
  while (first > 0 && tEnd - S.ts[first - 1] <= TRACE_SECONDS * 1000) first--;
  const dts = [];
  for (let i = Math.max(first, 1); i < n; i++) dts.push(S.ts[i] - S.ts[i - 1]);
  const dt = median(dts) / 1000 || 0.01; // s per frame
  const xOf = (t) => right - ((tEnd - t) / (TRACE_SECONDS * 1000)) * (right - left);
  rows.forEach((row, r) => {
    const y0 = top + r * rowH;
    let peak = 0;
    for (const [key] of row.keys) {
      for (let i = first; i < n; i++) {
        const v = S[key][i];
        if (v != null) peak = Math.max(peak, Math.abs((v * row.scale) / (row.positive ? 1 : dt)));
      }
    }
    peak = peak || 1;
    const mid = row.positive ? y0 + rowH - 4 : y0 + rowH / 2;
    const amp = row.positive ? rowH - 18 : rowH / 2 - 10;
    g.strokeStyle = color("--line");
    g.beginPath();
    g.moveTo(left, mid);
    g.lineTo(right, mid);
    g.stroke();
    for (const [key, c] of row.keys) {
      g.strokeStyle = color(c);
      g.lineWidth = 1.25;
      g.beginPath();
      let pen = false;
      for (let i = first; i < n; i++) {
        const v = S[key][i];
        if (v == null) { pen = false; continue; }
        const value = (v * row.scale) / (row.positive ? 1 : dt);
        const x = xOf(S.ts[i]);
        const y = mid - (value / peak) * amp;
        if (pen) g.lineTo(x, y); else g.moveTo(x, y);
        pen = true;
      }
      g.stroke();
    }
    // Dropped frames: red ticks under the cost.
    if (row.positive) {
      g.strokeStyle = color("--bad");
      g.beginPath();
      for (let i = first; i < n; i++) {
        if (S.cost[i] == null) {
          const x = xOf(S.ts[i]);
          g.moveTo(x, y0 + rowH - 4);
          g.lineTo(x, y0 + rowH - 10);
        }
      }
      g.stroke();
    }
    g.fillStyle = color("--muted");
    const title = `${row.label}  ${row.positive ? "" : "\u00b1"}${nice(peak)}`;
    g.fillText(title, left + 4, y0 + 12);
    if (!row.positive) {
      let x = left + 4 + g.measureText(title).width + 12;
      for (const [key, c] of row.keys) {
        if (row.keys.length < 2) break;
        g.fillStyle = color(c);
        g.fillText(key, x, y0 + 12);
        x += g.measureText(key).width + 10;
      }
    }
  });
  g.fillStyle = color("--muted");
  g.fillText(`last ${TRACE_SECONDS} s`, right - 50, h - 4);
}

// ----- log and summary -----
function takeLog(s) {
  if (!s.log?.length) return;
  const box = $("log");
  const atEnd = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
  for (const [i, line] of s.log) {
    if (i <= local.logSince) continue;
    box.append(`${line}\n`);
    local.logSince = i;
  }
  if (atEnd) box.scrollTop = box.scrollHeight;
}

function drawSummary(s) {
  const box = $("summary");
  box.hidden = !s.summary;
  if (s.summary) box.querySelector("pre").textContent = s.summary;
}

// ----- the loop -----
function banner(text) {
  $("banner").hidden = !text;
  $("banner").textContent = text || "";
}

function update(s) {
  if (s.run !== local.run) {
    local.run = s.run;
    resetSeries();
    // Versions count again from 1: keep the pictures, but take the next ones.
    for (const img of document.querySelectorAll("img")) delete img.dataset.v;
  }
  page.state = s;
  buildNet(s.net);
  takeSeries(s);
  takeLog(s);
  drawHeader(s);
  drawFrame(s);
  drawOverlay(s);
  drawPath(s);
  drawTraces();
  drawSummary(s);
  if (s.mode === "gui" && !local.gui) {
    local.gui = true;
    import("./gui.js").then((m) => m.init()).catch((e) => banner(`gui: ${e.message}`));
  }
  for (const listener of page.listeners) listener(s);
}

async function poll() {
  let delay = document.hidden ? 1000 : 150;
  try {
    const response = await fetch(api("state", { since: local.since, log_since: local.logSince }));
    if (response.status === 403) {
      banner("This link lacks its token: open the full link the terminal printed.");
      return;
    }
    update(await response.json());
    if (!local.alive) banner("");
    local.alive = true;
  } catch {
    if (local.alive) {
      banner(
        page.state?.mode === "run"
          ? "The run has ended: this page is no longer live."
          : "The gui has stopped: this page is no longer live.",
      );
    }
    local.alive = false;
    delay = 3000;
  }
  setTimeout(poll, delay);
}

$("stop").addEventListener("click", async () => {
  if (!confirm("Stop the run? The records so far are written.")) return;
  await post("stop").catch((e) => banner(e.message));
});
$("show-axes").addEventListener("change", () => page.state && drawOverlay(page.state));
window.addEventListener("resize", drawTraces);
poll();
