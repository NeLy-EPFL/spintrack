// The controls of `spintrack gui`: the ball (dragged over the frame, or clicked on its
// edge), the camera position, the field of view, the ignored regions, the tracking
// parameters, and saving. Every change goes to the server, which restarts the clip.

import { $, api, page, post, svg } from "./app.js";

const RIM_POINTS = 16;
const CAMERA_PRESETS = [
  ["behind", [0, 180, 0]],
  ["in front", [0, 0, 0]],
  ["its right", [0, 90, 0]],
  ["its left", [0, -90, 0]],
  ["above", [90, 0, 180]],
];
// The tracking parameters worth a control, with what they do.
const PARAMS = [
  ["tracking.window_px", "number", "window size (px)", "side of the square window the ball is tracked in; larger tracks more texture, slower", { min: 16, max: 400, step: 4 }],
  ["tracking.norm_window", "number", "brightness normalization", "size of the local brightness normalization, as a fraction of the window", { min: 0.05, max: 1, step: 0.05 }],
  ["tracking.max_step_rad", "number", "largest step (rad)", "the largest rotation accepted in one frame", { min: 0.05, max: 1.5, step: 0.05 }],
  ["tracking.illumination", "checkbox", "separate lighting", "separate the rig's static lighting from the ball's texture"],
  ["tracking.global_search", "checkbox", "relocalize when lost", "search the whole map for the ball's orientation after a lost frame"],
  ["tracking.max_bad_frames", "number", "restart after lost frames", "lost frames in a row before tracking restarts; empty: never", { min: 0, max: 1000, step: 1, optional: true }],
  ["tracking.forget_outside_view", "checkbox", "forget unseen map", "forget map cells the window does not see"],
];

let mode = null; // null, "rim", "ignore" or "square"
let clicks = [];
let drag = null; // {kind: "center"|"edge", ...} while a ball handle is held
let localBall = null; // the circle being dragged
const pending = new Map(); // control key -> timer
let last = null; // the last gui state

const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) node.setAttribute(k, v === true ? "" : v);
  }
  for (const child of children) if (child != null) node.append(child);
  return node;
};

const get = (cfg, key) => key.split(".").reduce((o, k) => (o == null ? o : o[k]), cfg);
const round = (v, d = 1) => Math.round(v * 10 ** d) / 10 ** d;

function say(text, bad = false) {
  const box = $("gui-message");
  box.textContent = text || "";
  box.className = bad ? "alert bad" : "alert";
  box.hidden = !text;
}

async function send(changes) {
  try {
    await post("config", { changes });
    say("");
  } catch (e) {
    say(e.message, true);
  }
}

function later(key, fn, ms = 250) {
  clearTimeout(pending.get(key));
  pending.set(key, setTimeout(() => { pending.delete(key); fn(); }, ms));
}

// ----- building the panel -----
function card(title, ...children) {
  return el("section", { class: "card" }, el("h3", {}, title), ...children);
}

function buildBall() {
  return card(
    "Ball",
    el("p", { id: "ball-note", class: "note" }),
    el("p", { class: "hint" }, "Drag the circle's center to move it and its edge to resize it, or click points on the ball's edge."),
    el("div", { class: "row" },
      el("button", { id: "ball-detect", onclick: () => post("detect").catch((e) => say(e.message, true)) }, "Find it again"),
      el("button", { id: "ball-rim", onclick: () => start("rim") }, "Click its edge"),
    ),
  );
}

function slider(key, label, min, max, index) {
  const range = el("input", { type: "range", min, max, step: 0.5, "data-angle": index });
  const number = el("input", { type: "number", min, max, step: 0.5, class: "num", "data-angle": index });
  const onInput = (source, other) => () => {
    other.value = source.value;
    later("camera", sendAngles, 150);
  };
  range.addEventListener("input", onInput(range, number));
  number.addEventListener("change", onInput(number, range));
  return el("label", { class: "slider" }, el("span", {}, label), range, number);
}

function angles() {
  const values = [0, 0, 0];
  for (const input of document.querySelectorAll("input[type=number][data-angle]")) {
    values[Number(input.dataset.angle)] = Number(input.value);
  }
  return values;
}

function sendAngles() {
  send({ "camera.position_deg": angles() });
}

function buildCamera() {
  const presets = el("div", { class: "row wrap" }, el("span", { class: "muted" }, "the camera is"));
  // A side keeps the elevation and twist the sliders hold; above sets all three.
  for (const [name, value] of CAMERA_PRESETS) {
    const position = () => (value[0] === 90 ? value : [angles()[0], value[1], angles()[2]]);
    presets.append(el("button", { onclick: () => send({ "camera.position_deg": position() }) }, name));
  }
  const plane = el("select", { id: "square-plane", onchange: updateToolbar },
    el("option", { value: "xy" }, "xy: flat, under the fly"),
    el("option", { value: "yz" }, "yz: upright, across the fly"),
    el("option", { value: "xz" }, "xz: upright, along the fly"),
  );
  return card(
    "Camera position",
    el("p", { id: "camera-ask", class: "alert bad", hidden: true },
      "Where is the camera? A video cannot tell the fly's front from its back: pick the side it films from. Until then the clip is tracked as if from behind."),
    el("p", { id: "camera-note", class: "note" }),
    presets,
    el("div", { id: "camera-angles" },
      slider("elevation", "elevation", -90, 90, 0),
      slider("azimuth", "azimuth", -180, 180, 1),
      slider("twist", "twist", -180, 180, 2),
    ),
    el("p", { id: "camera-rotation", class: "note", hidden: true }),
    el("p", { class: "hint" }, "Tick axes under the frame: x (red) points forward, y (green) to the fly's left, z (blue) up from the ball where the fly stands, which is where the trail starts: under the fly."),
    el("details", {},
      el("summary", {}, "from a calibration square"),
      el("p", { class: "hint" }, "Click the corners of a square aligned with the fly's axes, in FicTrac's order, which names them from the fly's point of view and so holds from any side: xy front-left, front-right, back-right, back-left; yz upper left, upper right, lower right, lower left; xz upper front, upper back, lower back, lower front. Another order can fit too, a half turn off: check the axes."),
      el("div", { class: "row" }, plane, el("button", { onclick: () => start("square") }, "Mark corners")),
    ),
  );
}

function buildLens() {
  const vfov = el("input", { type: "number", id: "vfov", min: 0.1, max: 179, step: 0.1, class: "num wide" });
  vfov.addEventListener("change", () => vfov.value && send({ "camera.vfov_deg": Number(vfov.value) }));
  const fisheye = el("input", { type: "checkbox", id: "fisheye" });
  fisheye.addEventListener("change", () => send({ "camera.fisheye": fisheye.checked }));
  return card(
    "Field of view",
    el("p", { id: "vfov-note", class: "note" }),
    el("div", { class: "row" },
      el("label", { class: "inline" }, "vertical (deg)", vfov),
      el("button", { id: "vfov-fit", onclick: () => post("fit-vfov").catch((e) => say(e.message, true)) }, "Fit"),
    ),
    el("label", { class: "inline" }, fisheye, "fisheye lens (equidistant)"),
  );
}

function buildIgnore() {
  return card(
    "Ignored regions",
    el("p", { class: "hint" }, "Outline what moves over the ball but is not the ball: the fly's legs, the tether, the holder's edge."),
    el("ul", { id: "ignore-list", class: "list" }),
    el("button", { onclick: () => start("ignore") }, "Add a region"),
  );
}

function buildParams() {
  const rows = PARAMS.map(([key, type, label, help, opts = {}]) => {
    const input = el("input", { type, "data-key": key, class: type === "number" ? "num wide" : "" });
    if (type === "number") {
      for (const k of ["min", "max", "step"]) input.setAttribute(k, opts[k]);
      input.addEventListener("change", () => {
        const value = input.value === "" ? (opts.optional ? null : undefined) : Number(input.value);
        if (value !== undefined) send({ [key]: value });
      });
    } else {
      input.addEventListener("change", () => send({ [key]: input.checked }));
    }
    const reset = el("button", { class: "reset", title: "back to the default", onclick: () => send({ [key]: get(last.defaults, key) }) }, "default");
    return el("div", { class: "param", title: help }, el("label", {}, type === "checkbox" ? input : null, el("span", {}, label)), type === "number" ? input : el("span"), reset);
  });
  return card("Tracking", ...rows);
}

function buildSave() {
  const path = el("input", { type: "text", id: "save-path", class: "path" });
  return card(
    "Save",
    el("div", { class: "row" }, path, el("button", { class: "primary", onclick: save }, "Save")),
    el("p", { id: "save-note", class: "note" }),
    el("code", { id: "command", class: "command" }),
    el("div", { class: "row" },
      el("button", { id: "track-all", onclick: trackAll }, "Track the whole video"),
    ),
    el("div", { id: "full-run", class: "note", hidden: true }),
  );
}

async function save() {
  try {
    const { path } = await post("save", { path: $("save-path").value || null });
    say(`saved ${path}`);
  } catch (e) {
    say(e.message, true);
  }
}

async function trackAll() {
  try {
    await post("track", { overwrite: false });
  } catch (e) {
    if (e.status === 409 && e.message.includes("earlier run") && confirm(`${e.message}. Replace it?`)) {
      await post("track", { overwrite: true }).catch((err) => say(err.message, true));
    } else {
      say(e.message, true);
    }
  }
}

// ----- the transport: the clip, the speed, pause and step -----
function buildTransport(box) {
  const pause = el("button", { id: "pause", onclick: () => post("play", { paused: !last.paused }) }, "Pause");
  const step = el("button", { id: "step", title: "one frame (.)", onclick: () => post("play", { step: true }) }, "Step");
  const restart = el("button", { title: "track the clip again from its start", onclick: () => post("play", { clip: last.clip }) }, "Restart");
  const startInput = el("input", { type: "range", id: "clip-start", min: 0, step: 1 });
  const length = el("select", { id: "clip-length" });
  for (const s of [2, 5, 10, 30, 60]) length.append(el("option", { value: s }, `${s} s`));
  length.append(el("option", { value: "all" }, "whole video"));
  const speed = el("select", { id: "speed" });
  for (const [v, label] of [[0.25, "0.25x"], [0.5, "0.5x"], [1, "1x"], [2, "2x"], [4, "4x"], [0, "max"]]) {
    speed.append(el("option", { value: v }, label));
  }
  const sendClip = () => {
    const first = Number(startInput.value);
    const n = length.value === "all" ? last.frames : Math.round(Number(length.value) * last.fps);
    post("play", { clip: [first, Math.min(first + n, last.frames)] });
  };
  startInput.addEventListener("input", () => {
    $("clip-label").textContent = clipLabel(Number(startInput.value));
    later("clip", sendClip, 300);
  });
  length.addEventListener("change", sendClip);
  speed.addEventListener("change", () => post("play", { speed: Number(speed.value) }));
  box.append(
    el("div", { class: "row" }, pause, step, restart,
      el("span", { class: "muted" }, "clip from"), startInput, el("span", { id: "clip-label", class: "muted mono" }),
      length, el("span", { class: "muted" }, "at"), speed),
  );
}

function clipLabel(frame) {
  const s = frame / (last?.fps || 1);
  return `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")} (frame ${frame})`;
}

// ----- drawing and dragging over the frame -----
const overlay = $("overlay");

function toSource(event) {
  const pt = overlay.createSVGPoint();
  pt.x = event.clientX;
  pt.y = event.clientY;
  return pt.matrixTransform(overlay.getScreenCTM().inverse());
}

function screenScale() {
  const box = overlay.viewBox.baseVal;
  return box && box.width ? overlay.clientWidth / box.width : 1;
}

function drawTools() {
  const layer = page.overlayTop;
  layer.replaceChildren();
  if (!last) return;
  const k = 1 / screenScale();
  const ball = localBall || last.ball || defaultBall();
  if (ball && !mode) {
    const [cx, cy, r] = ball;
    svg("circle", { class: "tool-circle", cx, cy, r }, layer);
    const edge = svg("circle", { class: "handle edge", cx: cx + r, cy, r: 7 * k }, layer);
    const center = svg("circle", { class: "handle", cx, cy, r: 7 * k }, layer);
    edge.addEventListener("pointerdown", (e) => grab(e, "edge"));
    center.addEventListener("pointerdown", (e) => grab(e, "center"));
    if (!last.ball) {
      const text = svg("text", { x: cx, y: cy - 12 * k, class: "tool-text", "text-anchor": "middle" }, layer);
      text.textContent = "drag onto the ball";
    }
  }
  if (mode) {
    const closed = mode !== "rim";
    if (clicks.length > 1) {
      svg(closed && clicks.length > 2 ? "polygon" : "polyline", {
        class: "tool-shape",
        points: clicks.map(([x, y]) => `${x},${y}`).join(" "),
      }, layer);
    }
    clicks.forEach(([x, y], i) => {
      svg("circle", { class: "tool-point", cx: x, cy: y, r: 4 * k }, layer);
      if (mode === "square") {
        const t = svg("text", { x: x + 6 * k, y: y - 6 * k, class: "tool-text" }, layer);
        t.textContent = String(i + 1);
      }
    });
  }
}

function defaultBall() {
  if (!last?.size || last.busy) return null;
  const [w, h] = last.size;
  return [w / 2, h / 2, 0.3 * Math.min(w, h)];
}

function grab(event, kind) {
  if (last.busy) return;
  event.preventDefault();
  event.stopPropagation();
  overlay.setPointerCapture(event.pointerId);
  localBall = [...(last.ball || defaultBall())];
  const p = toSource(event);
  drag = { kind, dx: p.x - localBall[0], dy: p.y - localBall[1] };
}

overlay.addEventListener("pointermove", (event) => {
  if (!drag) return;
  const p = toSource(event);
  if (drag.kind === "center") {
    localBall[0] = p.x - drag.dx;
    localBall[1] = p.y - drag.dy;
  } else {
    localBall[2] = Math.max(5, Math.hypot(p.x - localBall[0], p.y - localBall[1]));
  }
  drawTools();
});

overlay.addEventListener("pointerup", () => {
  if (!drag) return;
  drag = null;
  const [cx, cy, r] = localBall;
  const rim = Array.from({ length: RIM_POINTS }, (_, i) => {
    const a = (2 * Math.PI * i) / RIM_POINTS;
    return [round(cx + r * Math.cos(a)), round(cy + r * Math.sin(a))];
  });
  send({ "ball.rim": rim }).finally(() => { localBall = null; });
});

overlay.addEventListener("click", (event) => {
  if (!mode) return;
  const p = toSource(event);
  clicks.push([round(p.x), round(p.y)]);
  if (mode === "square" && clicks.length === 4) finish();
  else drawTools();
  updateToolbar();
});

overlay.addEventListener("dblclick", () => mode === "ignore" && finish());

// ----- click modes: the rim, an ignored region, a calibration square -----
const HINTS = {
  rim: "Click three or more points on the ball's edge, then Done.",
  ignore: "Click the corners of the region, then Done (or double-click).",
  get square() {
    return `Click the square's corners, as the fly sees them: ${SQUARE_ORDER[$("square-plane").value]}.`;
  },
};
// FicTrac's click order in the fly's terms, which unlike the image's hold from any side.
const SQUARE_ORDER = {
  xy: "front-left, front-right, back-right, back-left",
  yz: "upper left, upper right, lower right, lower left",
  xz: "upper front, upper back, lower back, lower front",
};

function start(next) {
  mode = next;
  clicks = [];
  updateToolbar();
  drawTools();
}

function updateToolbar() {
  const bar = $("toolbar");
  bar.hidden = !mode;
  if (!mode) return;
  $("toolbar-text").textContent = `${HINTS[mode]} (${clicks.length} so far)`;
  $("toolbar-done").hidden = mode === "square";
  $("toolbar-done").disabled = clicks.length < 3;
}

async function finish() {
  const done = mode;
  const pts = clicks;
  mode = null;
  clicks = [];
  updateToolbar();
  drawTools();
  if (done === "rim" && pts.length >= 3) await send({ "ball.rim": pts });
  if (done === "ignore" && pts.length >= 3) {
    const polygons = [...(last.config.mask.ignore || []), pts.map(([x, y]) => [Math.round(x), Math.round(y)])];
    await send({ "mask.ignore": polygons });
  }
  if (done === "square" && pts.length === 4) {
    await post("square", { corners: pts, plane: $("square-plane").value }).catch((e) => say(e.message, true));
  }
}

function cancel() {
  mode = null;
  clicks = [];
  updateToolbar();
  drawTools();
}

// ----- keeping the panel in step with the server -----
function setValue(input, value) {
  if (document.activeElement === input) return;
  if (input.type === "checkbox") input.checked = Boolean(value);
  else input.value = value ?? "";
}

function sync(s) {
  const g = s.gui;
  if (!g) return;
  last = g;
  const cfg = g.config;
  const busy = Boolean(g.busy);
  const fullRun = g.full_run && g.full_run.state === "tracking";
  $("controls").classList.toggle("disabled", busy || fullRun);
  if (g.error) say(g.error, true);
  // Ball
  $("ball-note").textContent = g.ball
    ? `${g.notes.ball || "from the config"}: center (${round(g.ball[0])}, ${round(g.ball[1])}), radius ${round(g.ball[2])} px`
    : busy ? "looking for it" : "not found yet: drag the circle onto it";
  // Camera
  const position = cfg.camera.position_deg;
  const rotation = cfg.camera.rotation;
  $("camera-note").textContent = g.notes.camera || (position ? "from the config" : "");
  $("camera-rotation").hidden = !rotation;
  if (rotation) $("camera-rotation").textContent = `set by a calibration square: rotation [${rotation.map((v) => v.toFixed(4)).join(", ")}]; move a slider or pick a preset to use angles instead`;
  $("camera-ask").hidden = !g.camera_unset;
  const shown = position || g.provisional;
  for (const input of document.querySelectorAll("[data-angle]")) {
    if (!pending.has("camera")) setValue(input, shown ? shown[Number(input.dataset.angle)] : 0);
  }
  // Field of view
  setValue($("vfov"), cfg.camera.vfov_deg != null ? round(cfg.camera.vfov_deg, 3) : "");
  setValue($("fisheye"), cfg.camera.fisheye);
  $("vfov-note").textContent = g.notes["field of view"] || (cfg.camera.vfov_deg != null ? "from the config" : "not known yet");
  // Ignored regions
  const list = $("ignore-list");
  const polygons = cfg.mask.ignore || [];
  if (list.dataset.n !== JSON.stringify(polygons)) {
    list.dataset.n = JSON.stringify(polygons);
    list.replaceChildren(...polygons.map((poly, i) => el("li", {},
      `region ${i + 1}, ${poly.length} corners `,
      el("button", { onclick: () => send({ "mask.ignore": polygons.filter((_, j) => j !== i) }) }, "remove"),
    )));
    if (!polygons.length) list.append(el("li", { class: "muted" }, "none"));
  }
  // Tracking parameters
  for (const [key] of PARAMS) {
    const input = document.querySelector(`[data-key="${key}"]`);
    setValue(input, get(cfg, key));
    const changed = JSON.stringify(get(cfg, key)) !== JSON.stringify(get(g.defaults, key));
    input.closest(".param").classList.toggle("changed", changed);
  }
  // Save
  setValue($("save-path"), g.save_to);
  $("save-note").textContent = g.saved ? "saved" : "unsaved changes";
  $("save-note").className = g.saved ? "note ok" : "note warn";
  $("command").textContent = g.command;
  const run = g.full_run;
  $("full-run").hidden = !run;
  if (run) {
    $("full-run").replaceChildren(
      `whole video: ${run.state}${run.folder ? `, outputs in ${run.folder}` : ""} `,
      g.hold ? el("button", { onclick: () => post("play", { resume: true }) }, "Back to tuning") : "",
    );
  }
  $("track-all").disabled = busy || fullRun || g.camera_unset;
  $("track-all").title = g.camera_unset ? "say where the camera sits first" : "";
  // Transport
  $("transport").hidden = fullRun || g.hold;
  $("pause").textContent = g.paused ? "Play" : "Pause";
  $("pause").classList.toggle("active", g.paused);
  $("step").disabled = !g.paused;
  const startInput = $("clip-start");
  startInput.max = Math.max(g.frames - 1, 0);
  if (!pending.has("clip")) {
    setValue(startInput, g.clip[0]);
    $("clip-label").textContent = clipLabel(g.clip[0]);
  }
  const span = (g.clip[1] - g.clip[0]) / g.fps;
  const length = $("clip-length");
  if (document.activeElement !== length) {
    length.value = g.clip[1] - g.clip[0] >= g.frames ? "all" : String([2, 5, 10, 30, 60].find((v) => Math.abs(v - span) < 0.5) ?? 10);
  }
  setValue($("speed"), String(g.speed));
  if (!drag) drawTools();
}

export function init() {
  $("main").classList.add("with-controls");
  const box = $("controls");
  box.hidden = false;
  box.append(
    el("div", { id: "gui-message", class: "alert", hidden: true }),
    buildBall(),
    buildCamera(),
    buildLens(),
    buildIgnore(),
    buildParams(),
    buildSave(),
    el("button", { class: "quit", onclick: () => confirm("Quit the gui?") && post("quit") }, "Quit"),
  );
  const toolbar = el("div", { id: "toolbar", class: "toolbar", hidden: true },
    el("span", { id: "toolbar-text" }),
    el("button", { id: "toolbar-done", class: "primary", onclick: finish }, "Done"),
    el("button", { onclick: cancel }, "Cancel"),
  );
  $("stage").append(toolbar);
  $("transport").hidden = false;
  buildTransport($("transport"));
  overlay.classList.add("editable");
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, select, textarea")) return;
    if (e.key === " ") { e.preventDefault(); post("play", { paused: !last?.paused }); }
    if (e.key === "." && last?.paused) post("play", { step: true });
    if (e.key === "Escape" && mode) cancel();
  });
  page.listeners.push(sync);
  if (page.state) sync(page.state);
}
