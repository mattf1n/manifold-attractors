// "How to characterize the activation space dynamics" slide. The paper's sphere probe of Llama 3.1 8B: a 2-D sphere through one
// natural activation x (radius |x|, spanned by x and two random tangents), injected at layer 15 and pushed through the later layers
// (assets/probe-sphere.js, assets/probe-face.js, computed on carc with the paper's code). The layer slider picks the layer; both panels
// follow it. Panels, left to right (top to bottom in portrait):
//   Globe: the input sphere, colored by how much the map stretches the surface there, with the
//   front face (the triangle facing the viewer) outlined. Drag to rotate.
//   Front face: the full-D surface projected onto the span of the globe's front face (see the panel code); its own rotation.
//   A white dot on the input sphere is shown on both and can be dragged along either. Drag elsewhere to rotate.
// Expansion (the paper's measure, maps/geometry.surface_area_map): the output/input area ratio sqrt(det J^T J) of the map from layer
// 15 to the selected layer at each point of the sphere, as log10; each triangle takes the mean of its three vertices. Colors: the
// paper's expansion colormap (matplotlib 'magma', linear in log10 area from the layer's minimum to its maximum, as in the paper's
// render/triple_area_heatmaps.py and language_area_heatmaps.py: cmap='magma', norm=Normalize(log10(area).min(), .max())).
// Nothing animates on its own: the layer plays only from the play button, and the panels redraw only when the layer changes, on a drag, or on resize.
// Edit CONFIG to tweak.
(() => {
  const IDS = { slide: '#probe-slide', globe: '#probe-globe', face: '#probe-face', slider: '#probe-layer', play: '#probe-play' };
  const CONFIG = {
    maxLayer: 20,             // the slider (and the play button) stops at this layer; the data runs to it; colors cover only reachable layers
    startLayer: null,         // layer shown when the slide opens (null: the injection layer, where nothing has stretched yet)
    normalize: true,          // front-face panel: each layer's outputs divided by its activation's norm |F(x)| (the paper scales each panel to
                              // its own surface, too); false: raw residual units
    colorRange: 'layer',      // 'layer': each layer spans the whole colormap from its own min to max log10 area (the paper: an independent log
                              // range per panel, so colors compare regions within a layer, not across layers); 'shared': one min..max over all
                              // reachable layers, so colors also compare across layers
    minColorRange: 0.05,      // 'layer': a layer whose log10 area spans less than this (the injection layer, where the map is the identity and
                              // every ratio is 1) is drawn over a range this wide around its mean, i.e. one flat mid-colormap color
    relativeToNorm: true,     // 'shared' only (a constant per layer changes nothing in 'layer'): divide the area ratio by the activation's own
                              // squared norm growth (|F(x)| / |x|)^2, i.e. the area as it appears with normalize
    // matplotlib 'magma' (the paper's expansion colormap), sampled at 33 evenly spaced points (matplotlib 3.11) and interpolated linearly.
    colormap: [[0,0,4],[3,3,18],[10,8,34],[19,13,52],[29,17,71],[41,17,90],[54,16,107],[68,15,118],[81,18,124],[93,23,127],[106,28,129],
      [118,33,129],[131,38,129],[144,42,129],[156,46,127],[170,51,125],[183,55,121],[196,60,117],[208,65,111],[220,72,105],[231,82,99],
      [239,93,94],[245,107,92],[249,121,93],[252,137,97],[253,152,105],[254,167,114],[254,182,124],[254,196,136],[254,211,149],
      [253,226,163],[252,240,178],[252,253,191]],
    faceAlpha: 1,             // opacity of the surface triangles (below 1 shows the folds behind)
    seam: 0.5,                // each triangle is grown by this many CSS pixels around its centre so neighbors overlap (no hairline seams)
    depthSlabs: 48,           // front-face panel: back-to-front painting in this many depth slices (triangles of one color within a slice share
                              // one fill; more slices = finer occlusion at folds, more fills)
    extentMargin: 1.04,       // margin around the front-face panel's farthest point
    guideColor: 'rgba(32,42,54,0.16)', guideWidth: 1,   // the globe's outline
    markColor: '#202a36',     // the activation (x, as in the paper's figure; on the globe) and the origin (+; front-face panel); both span the deck's --mark-tiny
    markStroke: 0.2,          // mark line width as a fraction of --mark-tiny
    markHalo: 'rgba(255,255,255,0.9)',   // drawn under the marks and the front face's outline (twice their width) so they show on magma's dark end and on the page
    // Starting rotation of the globe (radians). At 0, 0 the activation sits on the globe's right edge; yaw < 0 turns it toward the viewer.
    view: { yaw: -0.5, pitch: 0.3 },
    // At these layers a panel draws the probe's sample points (the paper's 4000-point lattice as tiny dots, colored like the surface
    // there, both sides of it, farther dots fainter) instead of the triangulated surface; layer 15 is the input sphere itself. Per panel:
    // globe, face (the front-face panel). [] always draws the surface.
    pointsAtLayers: { globe: [15], face: [15] },
    pointSize: 1.8,           // dot diameter in CSS pixels
    pointBackAlpha: 0.3,      // opacity of the farthest dots (the nearest are fully opaque), so the near side reads in front of the far side
    dragSpeed: 0.01,          // radians of rotation per CSS pixel dragged
    // Front-face panel: its own rotation on top of the axes matched to the globe (radians; 0, 0 = the globe's view, so at layer 15 the two
    // panels look the same), and whether it marks the origin with a +.
    faceView: { yaw: 0, pitch: 0 },
    faceOrigin: false,
    faceLine: 0, faceOutline: 0.6,   // front-face panel line widths (fractions of the mark stroke): origin-to-face-centroid line (0 = hidden), the face outline
    faceScale: 'fit',         // front-face panel: 'fit' scales each projection to fill the panel; 'fixed' keeps one scale for every face and layer (see the panel code)
    // The draggable dot (both panels): its start on the input sphere, in lattice coordinates (e1, e2, xhat; normalized): (0, 0, 1) is the
    // activation (the dot is drawn over the ✕). Size --mark-small, fill --probe-dot (CSS), outline markColor.
    // dotGrab: CSS pixels beyond its edge that still grab it; dotFarAlpha: its opacity on the globe's far side.
    dotStart: [0, 0, 1], dotGrab: 8, dotFarAlpha: 0.3,
    // The play button at the start of the slider row plays the layers once (js/slider-autoplay.js loop mode): from the current layer, or
    // from the first if it is at the last (holding the first layer for pauseStart ms), stepping through every layer at an even pace
    // (duration seconds for the whole climb, the same time per layer) and stopping at the last. Pressing it while playing pauses.
    // Adjusting the slider or dragging a panel stops it; leaving the slide stops it. Nothing plays on its own.
    autoplay: { mode: 'loop', pauseStart: 0, duration: 6, repeat: false }
  };
  const DATA = window.PROBE_SPHERE, FACE = window.PROBE_FACE;
  const slide = document.querySelector(IDS.slide), slider = document.querySelector(IDS.slider);
  const globeCanvas = document.querySelector(IDS.globe), faceCanvas = document.querySelector(IDS.face);
  if (!DATA || !FACE || !slide || !slider || !globeCanvas || !faceCanvas) return;

  const decode = (s, Type) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new Type(u.buffer); };
  const reach = DATA.layers.filter(l => l <= CONFIG.maxLayer).length;
  const layers = DATA.layers.slice(0, reach), n = DATA.points;
  const faces = decode(DATA.faces, Uint16Array), nf = faces.length / 3;
  // Input lattice (the paper's Fibonacci lattice; the activation is the pole (0, 0, 1)).
  const fib = new Float32Array(3 * n);
  for (let k = 0; k < n; k++) { const z = 1 - 2 * (k + 0.5) / n, r = Math.sqrt(Math.max(0, 1 - z * z)), ph = Math.PI * (3 - Math.sqrt(5)) * k; fib.set([r * Math.cos(ph), r * Math.sin(ph), z], 3 * k); }
  // Per layer: the face colors' log10 area (mean of the face's vertices).
  const unit = layers.map((_, i) => CONFIG.normalize ? 1 : DATA.norms[i]);
  const faceLog = layers.map((_, i) => {
    const a = decode(DATA.logArea[i], Int16Array), shift = CONFIG.relativeToNorm ? 2 * Math.log10(DATA.norms[i] / DATA.radius) : 0, out = new Float32Array(nf);
    for (let f = 0; f < nf; f++) out[f] = (a[faces[3 * f]] + a[faces[3 * f + 1]] + a[faces[3 * f + 2]]) / (3 * DATA.areaScale) - shift;
    return out;
  });
  // Color range per layer: [lo, hi] in log10 area (see CONFIG.colorRange).
  const span = a => { let lo = Infinity, hi = -Infinity; for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; } return [lo, hi]; };
  const shared = span(faceLog.flatMap(a => [...span(a)]));
  const colorRanges = faceLog.map(a => {
    if (CONFIG.colorRange === 'shared') return shared;
    const [lo, hi] = span(a), mid = (lo + hi) / 2, half = Math.max(hi - lo, CONFIG.minColorRange) / 2;
    return [mid - half, mid + half];
  });
  // Colors: a 256-entry palette interpolated from the colormap, and per layer the palette index of every face, all computed once here
  // so a layer change only swaps arrays (no per-face work on input).
  const cmap = CONFIG.colormap, last = cmap.length - 1;
  const palette = Array.from({ length: 256 }, (_, i) => {
    const u = i / 255 * last, k = Math.min(last - 1, Math.floor(u)), f = u - k;
    return `rgb(${cmap[k].map((c, j) => Math.round(c + (cmap[k + 1][j] - c) * f)).join(',')})`;
  });
  const faceBins = faceLog.map((a, i) => {
    const [lo, hi] = colorRanges[i], out = new Uint8Array(nf);
    for (let f = 0; f < nf; f++) out[f] = Math.round(Math.max(0, Math.min(1, (a[f] - lo) / (hi - lo))) * 255);
    return out;
  });
  // Per-point palette indices for the layers some panel draws as points (CONFIG.pointsAtLayers), on the same color range as the layer's faces;
  // each panel gets them only at its own layers (pointsFor).
  const pointLayers = Object.values(CONFIG.pointsAtLayers).flat();
  const pointBins = layers.map((l, i) => {
    if (!pointLayers.includes(l)) return null;
    const a = decode(DATA.logArea[i], Int16Array), shift = CONFIG.relativeToNorm ? 2 * Math.log10(DATA.norms[i] / DATA.radius) : 0;
    const [lo, hi] = colorRanges[i], out = new Uint8Array(n);
    for (let k = 0; k < n; k++) out[k] = Math.round(Math.max(0, Math.min(1, (a[k] / DATA.areaScale - shift - lo) / (hi - lo))) * 255);
    return out;
  });
  const pointsFor = name => layers.map((l, i) => ((CONFIG.pointsAtLayers[name] ?? []).includes(l) ? pointBins[i] : null));
  let current = Math.max(0, layers.indexOf(CONFIG.startLayer ?? layers[0]));

  const rotX = a => [1, 0, 0, 0, Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a)];
  const rotY = a => [Math.cos(a), 0, Math.sin(a), 0, 1, 0, -Math.sin(a), 0, Math.cos(a)];
  const mul = (A, B) => { const M = []; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) M.push(A[3*i]*B[j] + A[3*i+1]*B[3+j] + A[3*i+2]*B[6+j]); return M; };
  const apply = (M, v) => [M[0]*v[0] + M[1]*v[1] + M[2]*v[2], M[3]*v[0] + M[4]*v[1] + M[5]*v[2], M[6]*v[0] + M[7]*v[1] + M[8]*v[2]];
  let rotation = mul(rotX(CONFIG.view.pitch), rotY(CONFIG.view.yaw));   // the globe's drag rotation (it also picks the front face)
  const markSize = (canvas, dpr) => parseFloat(getComputedStyle(canvas).getPropertyValue('--mark-tiny')) * dpr || 8 * dpr;   // CSS px -> canvas px
  const cross = (ctx, x, y, r, w, plus) => {
    ctx.beginPath();
    if (plus) { ctx.moveTo(x - r, y); ctx.lineTo(x + r, y); ctx.moveTo(x, y - r); ctx.lineTo(x, y + r); }
    else { const d = r / Math.SQRT2; ctx.moveTo(x - d, y - d); ctx.lineTo(x + d, y + d); ctx.moveTo(x - d, y + d); ctx.lineTo(x + d, y - d); }
    ctx.lineCap = 'round';
    ctx.strokeStyle = CONFIG.markHalo; ctx.lineWidth = 2 * w; ctx.stroke();
    ctx.strokeStyle = CONFIG.markColor; ctx.lineWidth = w; ctx.stroke();
  };

  // One rotatable orthographic panel: points (3 per vertex, in the panel's own axes) -> screen through base (axes to screen x right,
  // y up, z toward the viewer) and the drag rotation: the shared one, or the panel's own when opts.view gives its start. Triangles are
  // filled in runs of one palette color (one path and one fill per run, no strokes); each is grown by CONFIG.seam canvas pixels around
  // its centre and wound the same way, so neighbors overlap and no hairline seams show. Layers drawn as points (opts.pointBins) fill dots
  // in runs the same way. Paint order comes from one counting sort (see draw), with buffers allocated once per panel.
  function panel(canvas, base, opts) {
    const ctx = canvas.getContext('2d'), screen = new Float32Array(3 * n), size = Math.max(nf, n), depth = new Float32Array(size);
    const order = new Uint32Array(size), key = new Int32Array(size), counts = new Uint32Array(CONFIG.depthSlabs * 256 + 1);
    let drag = null, own = opts.view ? mul(rotX(opts.view.pitch), rotY(opts.view.yaw)) : null;
    const view = () => mul(own ?? rotation, base);
    function draw() {
      const W = canvas.width, H = canvas.height, dpr = W / canvas.getBoundingClientRect().width, mark = markSize(canvas, dpr);
      const p = opts.points(), ext = typeof opts.extent === 'function' ? opts.extent() : opts.extent;   // points first: a fitted extent depends on them
      const s = (Math.min(W, H) / 2 - mark) / ext, cx = W / 2, cy = H / 2, M = view();
      const dots = opts.pointBins?.[current], bins = dots ?? faceBins[current], count = dots ? n : nf;
      // The convex globe's surface skips its far side and needs no depth order (one slice); dots (any panel) show both sides, back to front.
      const cull = opts.cull && !dots, slabs = cull ? 1 : CONFIG.depthSlabs;
      const X = v => cx + v[0] * s, Y = v => cy - v[1] * s;
      for (let k = 0; k < n; k++) { const x = p[3*k], y = p[3*k+1], z = p[3*k+2]; screen[3*k] = cx + (M[0]*x + M[1]*y + M[2]*z) * s; screen[3*k+1] = cy - (M[3]*x + M[4]*y + M[5]*z) * s; screen[3*k+2] = M[6]*x + M[7]*y + M[8]*z; }
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = CONFIG.guideColor; ctx.lineWidth = CONFIG.guideWidth * dpr; ctx.lineJoin = 'round';
      opts.guides?.(ctx, M, X, Y);
      if (dots) for (let k = 0; k < n; k++) depth[k] = screen[3 * k + 2];
      else for (let f = 0; f < nf; f++) depth[f] = screen[3 * faces[3 * f] + 2] + screen[3 * faces[3 * f + 1] + 2] + screen[3 * faces[3 * f + 2] + 2];
      // Paint order: one counting sort on (depth slab, color). Folded surfaces and point clouds go back to front in `slabs` equal depth
      // slices (painter's order between slices; within a slice, items are grouped by color, so whole color runs share one fill); the
      // globe's surface uses a single slice of its near side.
      let lo = Infinity, hi = -Infinity;
      if (!cull) for (let f = 0; f < count; f++) { if (depth[f] < lo) lo = depth[f]; if (depth[f] > hi) hi = depth[f]; }
      const slabScale = cull ? 0 : (slabs - 1e-6) / Math.max(hi - lo, 1e-9);
      counts.fill(0);
      for (let f = 0; f < count; f++) if (!cull || depth[f] > -0.05) { key[f] = Math.floor((depth[f] - lo) * slabScale || 0) * 256 + bins[f]; counts[key[f] + 1]++; } else key[f] = -1;
      for (let b = 1; b < counts.length; b++) counts[b] += counts[b - 1];
      let m = 0;
      for (let f = 0; f < count; f++) if (key[f] >= 0) { order[counts[key[f]]++] = f; m++; }
      ctx.globalAlpha = CONFIG.faceAlpha;
      let bin = -1;
      if (dots) {
        // Dots back to front; the run changes with the depth slab or the color, and farther slabs are fainter (a depth cue for the far side).
        const r = CONFIG.pointSize * dpr / 2, fade = CONFIG.pointBackAlpha;
        let run = -1;
        for (let i = 0; i < m; i++) {
          const k = order[i];
          if (key[k] !== run) {
            if (run >= 0) ctx.fill();
            run = key[k]; ctx.fillStyle = palette[bins[k]];
            ctx.globalAlpha = CONFIG.faceAlpha * (fade + (1 - fade) * Math.floor(run / 256) / Math.max(1, slabs - 1)); ctx.beginPath();
          }
          ctx.moveTo(screen[3 * k] + r, screen[3 * k + 1]); ctx.arc(screen[3 * k], screen[3 * k + 1], r, 0, 2 * Math.PI);
        }
        if (run >= 0) ctx.fill();
      } else {
        // Paths are flushed every 512 triangles even within one color: one huge path of overlapping triangles (a flat layer, all one color)
        // is far slower to fill than a few smaller ones.
        const grow = CONFIG.seam * dpr;
        let len = 0;
        for (let i = 0; i < m; i++) {
          const f = order[i];
          if (bins[f] !== bin || len === 512) { if (bin >= 0) ctx.fill(); bin = bins[f]; ctx.fillStyle = palette[bin]; ctx.beginPath(); len = 0; }
          len++;
          let a = 3 * faces[3 * f], b = 3 * faces[3 * f + 1], c = 3 * faces[3 * f + 2];
          if ((screen[b] - screen[a]) * (screen[c + 1] - screen[a + 1]) - (screen[b + 1] - screen[a + 1]) * (screen[c] - screen[a]) < 0) { const t = b; b = c; c = t; }   // same winding for all
          const gx = (screen[a] + screen[b] + screen[c]) / 3, gy = (screen[a + 1] + screen[b + 1] + screen[c + 1]) / 3;
          const out = (v, first) => { const dx = screen[v] - gx, dy = screen[v + 1] - gy, k = 1 + grow / (Math.hypot(dx, dy) || 1); first ? ctx.moveTo(gx + dx * k, gy + dy * k) : ctx.lineTo(gx + dx * k, gy + dy * k); };
          out(a, true); out(b, false); out(c, false); ctx.closePath();
        }
        if (bin >= 0) ctx.fill();
      }
      ctx.globalAlpha = 1;
      opts.marks(ctx, M, X, Y, mark, dpr);
      // The draggable dot (over the panel's marks, the activation's ✕ included): its face's corners in this panel, mixed by its
      // barycentric weights (on the globe, back onto the sphere).
      const { f, w } = probeDot, at = j => 3 * faces[3 * f + j];
      let d = [0, 1, 2].map(i => w[0] * p[at(0) + i] + w[1] * p[at(1) + i] + w[2] * p[at(2) + i]);
      if (opts.cull) { const l = Math.hypot(...d) || 1; d = d.map(v => v / l); }
      const q = apply(M, d), r = parseFloat(getComputedStyle(canvas).getPropertyValue('--mark-small')) * dpr / 2 || 6 * dpr;
      self.dot = { x: X(q), y: Y(q), r, dpr, s, cx, cy, M };
      ctx.globalAlpha = opts.cull && q[2] < 0 ? CONFIG.dotFarAlpha : 1;   // on the globe's far side: faint, so it can be followed round
      ctx.beginPath(); ctx.arc(X(q), Y(q), r, 0, 2 * Math.PI);
      ctx.fillStyle = getComputedStyle(canvas).getPropertyValue('--probe-dot').trim() || '#ffffff'; ctx.fill();
      ctx.lineWidth = 0.3 * r; ctx.strokeStyle = CONFIG.markColor; ctx.stroke();
      ctx.globalAlpha = 1;
    }
    // The surface location under a canvas point: { f, w } (face, barycentric weights), or null off the surface. The globe inverts its
    // orthographic view onto the sphere's near side (points outside the disc go to its rim); the other panels take the front-most
    // drawn triangle containing the point, from the last draw's screen positions.
    function pick(px, py) {
      const { s, cx, cy, M } = self.dot;
      if (opts.cull) {
        let x = (px - cx) / s, y = -(py - cy) / s; const r2 = x * x + y * y;
        if (r2 > 1) { const l = Math.sqrt(r2); x /= l; y /= l; }
        const z = Math.sqrt(Math.max(0, 1 - x * x - y * y));
        return sphereSpot([M[0] * x + M[3] * y + M[6] * z, M[1] * x + M[4] * y + M[7] * z, M[2] * x + M[5] * y + M[8] * z]);
      }
      let best = null, top = -Infinity;
      for (let g = 0; g < nf; g++) {
        const a = 3 * faces[3 * g], b = 3 * faces[3 * g + 1], c = 3 * faces[3 * g + 2];
        const x1 = screen[b] - screen[a], y1 = screen[b + 1] - screen[a + 1], x2 = screen[c] - screen[a], y2 = screen[c + 1] - screen[a + 1];
        const det = x1 * y2 - x2 * y1; if (Math.abs(det) < 1e-9) continue;
        const dx = px - screen[a], dy = py - screen[a + 1], w1 = (dx * y2 - x2 * dy) / det, w2 = (x1 * dy - dx * y1) / det;
        if (w1 < 0 || w2 < 0 || w1 + w2 > 1) continue;
        const z = screen[a + 2] + screen[b + 2] + screen[c + 2];
        if (z > top) { top = z; best = { f: g, w: [1 - w1 - w2, w1, w2] }; }
      }
      return best;
    }
    // Pressing the dot drags it along the surface (every panel follows); pressing anywhere else turns the panel's rotation (the globe's
    // redraws both panels and moves the front face; the front-face panel's own only itself).
    const local = e => { const b = canvas.getBoundingClientRect(), k = canvas.width / b.width; return [(e.clientX - b.left) * k, (e.clientY - b.top) * k]; };
    canvas.addEventListener('pointerdown', e => {
      e.preventDefault(); canvas.setPointerCapture(e.pointerId); canvas.style.cursor = 'grabbing';
      const [px, py] = local(e), d = self.dot;
      drag = d && Math.hypot(px - d.x, py - d.y) <= d.r + CONFIG.dotGrab * d.dpr ? { dot: true } : { x: e.clientX, y: e.clientY };
    });
    canvas.addEventListener('pointermove', e => {
      if (!drag) return;
      if (drag.dot) { const spot = pick(...local(e)); if (spot) { probeDot = spot; schedule(); } return; }
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag = { x: e.clientX, y: e.clientY };
      const turn = mul(rotX(dy * CONFIG.dragSpeed), rotY(dx * CONFIG.dragSpeed));
      if (own) { own = mul(turn, own); schedule(self); } else { rotation = mul(turn, rotation); schedule(); }
    });
    const end = e => { drag = null; canvas.style.cursor = ''; if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId); };
    canvas.addEventListener('pointerup', end); canvas.addEventListener('pointercancel', end);
    const self = { canvas, draw, view, dot: null };
    return self;
  }

  // The dot's place on the input sphere for a direction v (lattice coordinates): the face its ray crosses, with the barycentric weights
  // of the crossing (the corners' mix that points along v). Searched among the faces whose centres are nearest v.
  function sphereSpot(v) {
    const near = [];
    for (let g = 0; g < nf; g++) {
      const d = v[0] * faceDir[3 * g] + v[1] * faceDir[3 * g + 1] + v[2] * faceDir[3 * g + 2];
      if (near.length < 8 || d > near[near.length - 1][0]) { near.push([d, g]); near.sort((a, b) => b[0] - a[0]); near.length = Math.min(near.length, 8); }
    }
    let best = null, worst = -Infinity;
    for (const [, g] of near) {   // solve v = l0 a + l1 b + l2 c; the weights are l / sum(l), all >= 0 inside the face
      const [a, b, c] = [0, 1, 2].map(j => Array.from(fib.subarray(3 * faces[3 * g + j], 3 * faces[3 * g + j] + 3)));
      const det3 = (p, q, r) => p[0] * (q[1] * r[2] - q[2] * r[1]) - p[1] * (q[0] * r[2] - q[2] * r[0]) + p[2] * (q[0] * r[1] - q[1] * r[0]);
      const D = det3(a, b, c), l = [det3(v, b, c) / D, det3(a, v, c) / D, det3(a, b, v) / D], t = l[0] + l[1] + l[2];
      const w = l.map(x => x / t), lo = Math.min(...w);
      if (lo > worst) { worst = lo; best = { f: g, w }; }
      if (lo >= 0) break;
    }
    if (worst < 0) { best.w = best.w.map(x => Math.max(0, x)); const t = best.w[0] + best.w[1] + best.w[2]; best.w = best.w.map(x => x / t); }
    return best;
  }

  // A path through screen points drawn like the marks: a markHalo underlay twice as wide, then markColor.
  const strokeMark = (ctx, pts, w, closed) => {
    ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); if (closed) ctx.closePath();
    ctx.lineCap = ctx.lineJoin = 'round';
    ctx.strokeStyle = CONFIG.markHalo; ctx.lineWidth = 2 * w; ctx.stroke();
    ctx.strokeStyle = CONFIG.markColor; ctx.lineWidth = w; ctx.stroke();
  };

  // Front face: the input-sphere triangle under the centre of the globe (the face whose centroid direction is nearest the viewer's line
  // of sight). It changes as the globe is turned.
  const GLOBE_BASE = [0, 0, 1, 0, 1, 0, 1, 0, 0];
  const faceDir = new Float32Array(3 * nf);
  for (let f = 0; f < nf; f++) for (let j = 0; j < 3; j++) faceDir[3 * f + j] = fib[3 * faces[3 * f] + j] + fib[3 * faces[3 * f + 1] + j] + fib[3 * faces[3 * f + 2] + j];
  // The dot's place: a face of the input mesh and barycentric weights, so it sits on every panel's surface (see sphereSpot).
  let probeDot = sphereSpot(CONFIG.dotStart.map(v => v / Math.hypot(...CONFIG.dotStart)));
  const activationSpot = sphereSpot([0, 0, 1]);   // the activation (lattice pole), placed the same way on the front-face panel
  function frontFace() {
    const M = mul(rotation, GLOBE_BASE);   // row 3: each lattice axis's component toward the viewer
    let best = 0, top = -Infinity;
    for (let f = 0; f < nf; f++) { const d = M[6] * faceDir[3 * f] + M[7] * faceDir[3 * f + 1] + M[8] * faceDir[3 * f + 2]; if (d > top) { top = d; best = f; } }
    return best;
  }

  // Smallest sphere around a point set (n points, 3 per point), exactly: the incremental (Welzl-style) algorithm, points taken in a fixed
  // shuffled order (expected linear time). The smallest sphere is unique, so the same set in two orientations gets the same sphere, turned:
  // at layer 15 the front-face panel's points are the input sphere turned, so both panels draw it the same size at the same place.
  const ORDER = (() => { const o = Uint32Array.from({ length: n }, (_, k) => k); let s = 12345; for (let k = n - 1; k > 0; k--) { s = (s * 1103515245 + 12345) >>> 0; const j = s % (k + 1); [o[k], o[j]] = [o[j], o[k]]; } return o; })();
  function enclosing(p) {
    const P = k => [p[3 * k], p[3 * k + 1], p[3 * k + 2]];
    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], d3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const crs = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const two = (a, b) => { const c = a.map((v, i) => (v + b[i]) / 2); return [c, d3(sub(a, c), sub(a, c))]; };
    const three = (a, b, c) => {   // circumcircle of a triangle, in its plane
      const ab = sub(b, a), ac = sub(c, a), nn = crs(ab, ac), q = d3(nn, nn);
      if (q < 1e-18) { const s = [two(a, b), two(a, c), two(b, c)]; return s.reduce((x, y) => (y[1] > x[1] ? y : x)); }   // collinear: widest pair
      const t = crs(nn, ab).map((v, i) => v * d3(ac, ac)), u = crs(ac, nn).map((v, i) => v * d3(ab, ab));
      const o = a.map((v, i) => v + (t[i] + u[i]) / (2 * q)); return [o, d3(sub(a, o), sub(a, o))];
    };
    const four = (a, b, c, e) => {   // circumsphere of a tetrahedron
      const u = sub(b, a), v = sub(c, a), w = sub(e, a), det = 2 * d3(u, crs(v, w));
      if (Math.abs(det) < 1e-18) return three(a, b, c);
      const x = crs(v, w).map(t => t * d3(u, u)), y = crs(w, u).map(t => t * d3(v, v)), z = crs(u, v).map(t => t * d3(w, w));
      const o = a.map((t, i) => t + (x[i] + y[i] + z[i]) / det); return [o, d3(sub(a, o), sub(a, o))];
    };
    const out = (q, [c, r2]) => d3(sub(q, c), sub(q, c)) > r2 * (1 + 1e-10) + 1e-14;
    let ball = [P(ORDER[0]), 0];
    for (let i = 1; i < n; i++) {
      const a = P(ORDER[i]); if (!out(a, ball)) continue;
      ball = [a, 0];
      for (let j = 0; j < i; j++) {
        const b = P(ORDER[j]); if (!out(b, ball)) continue;
        ball = two(a, b);
        for (let k = 0; k < j; k++) {
          const c = P(ORDER[k]); if (!out(c, ball)) continue;
          ball = three(a, b, c);
          for (let l = 0; l < k; l++) { const e = P(ORDER[l]); if (out(e, ball)) ball = four(a, b, c, e); }
        }
      }
    }
    return { mid: ball[0], radius: Math.sqrt(ball[1]) };
  }

  // Globe: lattice axes (e1, e2, xhat) -> screen (xhat right, e2 up, e1 toward the viewer), scaled by the same rule as the front-face
  // panel's 'fit' (its enclosing sphere times extentMargin), so the two panels' layer-15 spheres are the same size.
  const globeExtent = CONFIG.extentMargin * enclosing(fib).radius;
  const globePanel = panel(globeCanvas, GLOBE_BASE, {
    extent: globeExtent, points: () => fib, cull: true,
    pointBins: pointsFor('globe'),
    marks(ctx, M, X, Y, mark, dpr) {
      ctx.strokeStyle = CONFIG.guideColor; ctx.lineWidth = CONFIG.guideWidth * dpr;
      ctx.beginPath(); ctx.arc(X([0, 0]), Y([0, 0]), X([1, 0]) - X([0, 0]), 0, 2 * Math.PI); ctx.stroke();
      const pole = apply(M, [0, 0, 1]);
      if (pole[2] > 0) cross(ctx, X(pole), Y(pole), mark / 2, mark * CONFIG.markStroke, false);   // the activation (pole), when on the near side
      if (FACE) {   // the front face, outlined
        const f = frontFace(), corner = j => { const v = apply(M, Array.from(fib.subarray(3 * faces[3 * f + j], 3 * faces[3 * f + j] + 3))); return [X(v), Y(v)]; };
        strokeMark(ctx, [corner(0), corner(1), corner(2)], mark * CONFIG.markStroke * CONFIG.faceOutline, true);
      }
    }
  });

  // Front-face panel (window.PROBE_FACE, assets/probe-face.js): the whole output surface at the layer, in full D, projected orthogonally onto the 3-D span
  // of the front face's three output points (the span includes the origin), with the segment from the origin to the face's output
  // centroid and the face itself outlined. Axes inside that span: chosen to match the globe's view. Each layer's map gets a linear fit,
  // L = the least-squares linear map from input-sphere directions to output points (exact at layer 15, where the map is the identity);
  // the globe's screen axes (right, up, toward the viewer) sent through L and projected into the span give three vectors, and the panel's
  // axes are the rotation nearest to them (polar decomposition). So at layer 15 the panel shows the sphere exactly as the globe does (same
  // dots, x and dot at the same places), and later its view follows the globe's as closely as the face's span allows. It has its own
  // rotation on top (CONFIG.faceView, 0 = matched); dragging it does not move the front face; turning the globe changes the face and axes.
  let facePanel = null;
  if (FACE && faceCanvas) {
    const coords = layers.map((l, i) => {   // per layer: rank r and the point coordinates (n x r) in the layer's orthonormal basis
      const L = FACE.layers.find(e => e.layer === l), q = decode(L.coords, Int16Array), r = L.rank, out = new Float32Array(q.length);
      for (let j = 0; j < q.length; j++) out[j] = q[j] * L.scale[j % r] * unit[i];
      return { r: L.rank, c: out };
    });
    // Panel extent and centre (CONFIG.faceScale): 'fit' = centred on the projected surface itself (the middle of its bounding box), scaled so
    // its farthest point from there fills the panel, so it reads about as large as the globe (rescales when the face or layer changes; the
    // origin's + may then sit off-centre or off the panel); 'fixed' = centred on the origin, scaled to the farthest point in full D over the
    // reachable layers (never clips, but most faces' projections then fill only a small part).
    const fixedExtent = CONFIG.extentMargin * Math.max(...coords.map(({ r, c }) => {
      let m = 0; for (let k = 0; k < n; k++) { let t = 0; for (let j = 0; j < r; j++) t += c[k * r + j] ** 2; m = Math.max(m, t); } return Math.sqrt(m);
    }));
    const pts = new Float32Array(3 * n), cache = { key: '', corners: null, origin: [0, 0, 0], centroid: [0, 0, 0], fit: 1 };
    const dot = (a, b) => a.reduce((t, x, j) => t + x * b[j], 0);
    // Linear fit per layer: L (r x 3) = (sum_k C_k fib_k^T) (sum_k fib_k fib_k^T)^-1.
    const inv3 = m => { const [a, b, c, d, e, f, g, h, i] = m, A = e * i - f * h, B = f * g - d * i, Cc = d * h - e * g, det = a * A + b * B + c * Cc;
      return [A, c * h - b * i, b * f - c * e, B, a * i - c * g, c * d - a * f, Cc, b * g - a * h, a * e - b * d].map(v => v / det); };
    const G = new Array(9).fill(0);
    for (let k = 0; k < n; k++) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) G[3 * a + b] += fib[3 * k + a] * fib[3 * k + b];
    const Gi = inv3(G);
    const fits = coords.map(({ r, c }) => {
      const T = new Float64Array(3 * r), L = new Float64Array(3 * r);
      for (let k = 0; k < n; k++) for (let j = 0; j < r; j++) for (let a = 0; a < 3; a++) T[3 * j + a] += c[k * r + j] * fib[3 * k + a];
      for (let j = 0; j < r; j++) for (let a = 0; a < 3; a++) L[3 * j + a] = T[3 * j] * Gi[a] + T[3 * j + 1] * Gi[3 + a] + T[3 * j + 2] * Gi[6 + a];
      return L;
    });
    function project() {   // recompute the projection when the layer, the front face or the globe's rotation changed
      const f = frontFace(), key = current + ':' + f + ':' + CONFIG.faceScale + ':' + rotation.join(',');
      if (key === cache.key) return;
      cache.key = key;
      const { r, c } = coords[current], L = fits[current], row = v => Array.from(c.subarray(v * r, v * r + r));
      const [A, B, C] = [0, 1, 2].map(j => row(faces[3 * f + j]));
      // An orthonormal basis of the face's span (Gram-Schmidt of its three output points).
      const Q = [];
      for (const v of [A, B, C]) { let w = v.slice(); for (const q of Q) { const t = dot(w, q); w = w.map((x, j) => x - t * q[j]); } const l = Math.hypot(...w) || 1; Q.push(w.map(x => x / l)); }
      // The globe's screen axes (rows of its view matrix, in lattice coordinates) through L, in span coordinates: S (3 x 3, column = axis).
      const M = mul(rotation, GLOBE_BASE), S = new Array(9);
      for (let a = 0; a < 3; a++) {
        const t = new Array(r);
        for (let j = 0; j < r; j++) t[j] = L[3 * j] * M[3 * a] + L[3 * j + 1] * M[3 * a + 1] + L[3 * j + 2] * M[3 * a + 2];
        for (let i = 0; i < 3; i++) S[3 * i + a] = dot(Q[i], t);
      }
      // Nearest rotation to S (polar decomposition by Newton's iteration R <- (R + R^-T) / 2).
      let R = S.slice();
      for (let it = 0; it < 20; it++) { const I = inv3(R); R = R.map((v, idx) => (v + I[3 * (idx % 3) + Math.floor(idx / 3)]) / 2); }
      const axes = [0, 1, 2].map(a => Q[0].map((_, j) => Q[0][j] * R[a] + Q[1][j] * R[3 + a] + Q[2][j] * R[6 + a]));   // panel x, y, z in output coordinates
      for (let k = 0; k < n; k++) {
        let p1 = 0, p2 = 0, p3 = 0;
        for (let j = 0; j < r; j++) { const v = c[k * r + j]; p1 += v * axes[0][j]; p2 += v * axes[1][j]; p3 += v * axes[2][j]; }
        pts[3 * k] = p1; pts[3 * k + 1] = p2; pts[3 * k + 2] = p3;
      }
      // Centre ('fit'): the centre of the smallest sphere around the projected surface (enclosing), so any rotation fits and the surface
      // fills the panel; 'fixed': the origin. The points, the origin and the centroid are stored relative to it.
      const fit = CONFIG.faceScale === 'fixed' ? { mid: [0, 0, 0], radius: 0 } : enclosing(pts), mid = fit.mid;
      let m = 0;
      for (let k = 0; k < n; k++) { for (let i = 0; i < 3; i++) pts[3 * k + i] -= mid[i]; m = Math.max(m, Math.hypot(pts[3 * k], pts[3 * k + 1], pts[3 * k + 2])); }
      const centroid = A.map((x, j) => (x + B[j] + C[j]) / 3);
      cache.corners = [0, 1, 2].map(j => { const v = faces[3 * f + j]; return [pts[3 * v], pts[3 * v + 1], pts[3 * v + 2]]; });
      cache.origin = mid.map(v => -v);
      cache.centroid = axes.map((a, i) => dot(centroid, a) - mid[i]);
      cache.fit = CONFIG.extentMargin * m;
    }
    facePanel = panel(faceCanvas, [1, 0, 0, 0, 1, 0, 0, 0, 1], {
      extent: () => (CONFIG.faceScale === 'fixed' ? fixedExtent : cache.fit), view: CONFIG.faceView,
      pointBins: pointsFor('face'),
      points: () => { project(); return pts; },
      marks(ctx, M, X, Y, mark, dpr) {
        const w = mark * CONFIG.markStroke, o = apply(M, cache.origin), g = apply(M, cache.centroid);
        if (CONFIG.faceLine > 0) strokeMark(ctx, [[X(o), Y(o)], [X(g), Y(g)]], w * CONFIG.faceLine, false);       // origin -> face centroid
        strokeMark(ctx, cache.corners.map(v => { const p = apply(M, v); return [X(p), Y(p)]; }), w * CONFIG.faceOutline, true);   // the face
        if (CONFIG.faceOrigin) cross(ctx, X(o), Y(o), mark / 2, w, true);                                          // origin (off by default)
        const a = [0, 1, 2].reduce((s, j) => { const v = faces[3 * activationSpot.f + j], wt = activationSpot.w[j]; return s.map((x, i) => x + wt * pts[3 * v + i]); }, [0, 0, 0]);
        const pa = apply(M, a); cross(ctx, X(pa), Y(pa), mark / 2, w, false);                                    // the activation's image (x, under the dot)
      }
    });
  }
  const panels = [globePanel, facePanel].filter(Boolean);

  // Redraws: batched into one animation frame, only for the panels that changed, never while the slide is hidden.
  const dirty = new Set();
  let queued = false;
  function render() { queued = false; if (slide.hidden) { dirty.clear(); return; } for (const p of dirty) if (p.canvas.width > 1) { const t = performance.now(); p.draw(); p.ms = performance.now() - t; } dirty.clear(); }
  function schedule(...which) { (which.length ? which : panels).forEach(p => dirty.add(p)); if (!queued) { queued = true; requestAnimationFrame(render); } }
  // A layer change draws both panels right away, inside the input event, so they paint in the same frame as the slider's thumb, fill
  // and readout (the autoplay sets the slider from an animation frame; a redraw scheduled from there would land one frame later).
  function drawNow() { panels.forEach(p => dirty.add(p)); render(); }
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    for (const { canvas: c } of panels) { const r = c.getBoundingClientRect(); c.width = Math.max(1, Math.round(r.width * dpr)); c.height = Math.max(1, Math.round(r.height * dpr)); }
    schedule();
  }
  const observer = new ResizeObserver(resize); panels.forEach(p => observer.observe(p.canvas));
  new MutationObserver(() => { if (!slide.hidden) resize(); }).observe(slide, { attributes: true, attributeFilter: ['hidden'] });

  // Slider: one position per reachable layer; the value shown is the layer number.
  slider.min = 0; slider.max = layers.length - 1; slider.step = 1; slider.value = current;
  // Any element with data-probe-layer (e.g. a <span> in a panel title) shows the selected layer.
  const layerLabels = slide.querySelectorAll('[data-probe-layer]');
  const sync = () => {
    slider.setAttribute('aria-valuetext', String(layers[current]));   // shown by <output data-auto>
    slider.style.setProperty('--fill', `${layers.length > 1 ? current / (layers.length - 1) * 100 : 0}%`);
    layerLabels.forEach(el => { el.textContent = String(layers[current]); });
  };
  sync();
  slider.addEventListener('input', () => { current = Number(slider.value); sync(); drawNow(); });
  slider.addEventListener('pointerup', () => slider.blur());      // so the arrow keys go back to changing slides
  slider.addEventListener('keydown', e => e.stopPropagation());   // ...but while the slider is focused they adjust it instead
  // Rotating either panel counts as using the slide, like adjusting the slider: it stops the playback.
  const autoplay = window.sliderAutoplay(slider, slide, { ...CONFIG.autoplay, pauseOn: [globeCanvas, faceCanvas], playButton: document.querySelector(IDS.play) });   // js/slider-autoplay.js

  window.probeDemoState = () => ({ config: CONFIG, layer: layers[current], layers, colorRange: colorRanges[current], globeView: globePanel.view(),
    frontFace: frontFace(), drawMs: { globe: globePanel.ms, face: facePanel?.ms }, autoplaying: autoplay.playing(),
    dot: probeDot, dots: Object.fromEntries(panels.map(p => ['#' + p.canvas.id, p.dot && { x: p.dot.x, y: p.dot.y }])) });   // for console inspection (drawMs: last draw time per panel; dots: the dot's canvas px per panel)
  if (!slide.hidden) resize();
})();
