// Slide after "How to characterize the activation space dynamics": the paper's sphere probe through THREE activations (Fig. "Sphere probe
// expansion maps for six randomly sampled sets of 3 activations", panel 3, OLMo 3 7B; assets/tri-sphere.js, computed on carc with the
// paper's code; which blocks the map runs through (DATA.blocks) and the provenance are in the asset).
//   Left: the input sphere, a globe through the three activations (marked x), in the paper's frame (pole = their normalized sum, first =
//   toward the first activation, second = toward the third), colored by the paper's expansion (log10 output/input area ratio, magma).
//   Right: its image after the last block, in the same kind of frame built from the activations' images (a1 = normalized sum of the three
//   images, a2 = toward the first image, a3 = toward the third), colored the same way; the three images marked x. Under it, the plot of
//   how fast the image moves along the dot's current segment (CONFIG.plot).
// A dot on the input sphere travels on its own from activation to activation (1 -> 2 -> 3 -> 1 ...) at constant speed; the dot on the
// right is its image, interpolated from the precomputed grid of the sphere's images. Drag the left dot to move it over the sphere; the
// right dot follows. resumeDelay ms after release it sets off again from there toward the activation it was heading to. Dragging
// anywhere else on either panel rotates both (one shared rotation: the two frames are built the same way, so the same rotation shows
// corresponding directions -- the pole and its image, toward the first activation and toward its image -- in the same place).
// Everything sleeps while the slide or tab is hidden, and the motion stops idleStop ms after the slide was shown or last touched.
// Edit CONFIG to tweak; sizes and colors of the marks are the deck's (--mark-tiny, --mark-small) and the slide's CSS variables.
(() => {
  const IDS = { slide: '#trisphere-slide', input: '#trisphere-input', output: '#trisphere-output', plot: '#trisphere-plot', play: '#trisphere-play' };
  const CONFIG = {
    // matplotlib 'magma' (the paper's expansion colormap), sampled at 33 evenly spaced points (matplotlib 3.11) and interpolated linearly;
    // linear in log10 area from the paper's minimum to its maximum for this sphere (the paper's per-panel Normalize).
    colormap: [[0,0,4],[3,3,18],[10,8,34],[19,13,52],[29,17,71],[41,17,90],[54,16,107],[68,15,118],[81,18,124],[93,23,127],[106,28,129],
      [118,33,129],[131,38,129],[144,42,129],[156,46,127],[170,51,125],[183,55,121],[196,60,117],[208,65,111],[220,72,105],[231,82,99],
      [239,93,94],[245,107,92],[249,121,93],[252,137,97],[253,152,105],[254,167,114],[254,182,124],[254,196,136],[254,211,149],
      [253,226,163],[252,240,178],[252,253,191]],
    faceAlpha: 1,             // opacity of the surface triangles (below 1 shows the folds behind)
    seam: 0.5,                // each triangle is grown by this many CSS pixels around its centre so neighbors overlap (no hairline seams)
    depthSlabs: 48,           // output panel: back-to-front painting in this many depth slices (more = finer occlusion at folds, more fills)
    extentMargin: 1.04,       // each panel's radius: its farthest point from its centre of rotation times this (any rotation fits)
    outputCenter: 'surface',  // output panel turns about (and is scaled to) 'surface': the centre of the image's bounding box, so the image fills
                              // the panel; 'origin': the origin of the output space (the image sits off-centre, far from it after many blocks)
    // Starting view, shared by both panels (the same rotation on both). center: the direction that faces the viewer at the middle of each
    // panel -- 'centroid': the centroid of the three activations, (A + B + C) / 3, i.e. the paper's chart pole (on the right, its image
    // frame's a1, the normalized sum of the three images: the same rotation); 'directions': the normalized sum of the three activations'
    // unit directions (the centre of the three marks on the globe; differs from 'centroid' only as much as the activations' norms differ).
    // Around it, toward the first activation points right and the third up. yaw, pitch (radians) then turn the view away from that.
    view: { center: 'centroid', yaw: 0, pitch: 0 },
    lockRotation: true,       // true: dragging either panel turns both; false: each panel turns on its own
    dragSpeed: 0.01,          // radians of rotation per CSS pixel dragged
    guideColor: 'rgba(32,42,54,0.16)', guideWidth: 1,   // the globe's outline
    markColor: '#202a36',     // the activations (x, size --mark-tiny) and the moving dot's outline (size --mark-small; its fill: --trisphere-dot)
    markStroke: 0.2,          // x line width as a fraction of --mark-tiny
    markHalo: 'rgba(255,255,255,0.9)',   // drawn under the marks so they show on magma's dark end as well as on the page
    farAlpha: 0.3,            // input globe: opacity of marks on the far side of the globe (so the dot can be followed all the way round)
    grabSlop: 10,             // CSS pixels around the dot that still grab it (more than its radius, for fingers)
    // The dot's own motion. path: 'chord' moves at constant speed along the straight line (chord) between consecutive activations in
    // the input space, shown at its radial projection onto the sphere (the chord itself runs inside the sphere; the projection is the
    // great-circle arc between them, traversed fastest at its middle); 'arc' moves along the same arc at constant angular speed.
    // speed: sphere radii per second along the chord ('chord') or radians per second along the arc ('arc'). hold: ms the dot rests on
    // each activation before leaving. resumeDelay: ms after the dot is released before it moves again (the deck's 10 s).
    motion: { path: 'chord', speed: 0.35, hold: 0, resumeDelay: 10000 },
    idleStop: 90000,          // ms after the slide was shown or last touched before the motion stops (sleeps); showing or touching wakes it
    maxFps: 60,               // the moving dot redraws at most this often
    // The plot under the output panel: the paper's speed profile (output distance per radian, |dF/dtheta|) along the great circle through
    // the two activations of the dot's current segment (from the one it left to the one it is heading to; the dot's path), from the
    // asset's exact speeds (not the surface grid), with a marker at the dot's place (x = angle along the circle). Between the two
    // activations the curve is drawn in full; past each of them it continues along the same circle, muted (--trisphere-curve-muted-alpha),
    // until the speed has risen again: on each side, out to the first point where it reaches riseTo times the segment's midpoint speed,
    // plus pastRise more (both in segment lengths), but no farther than maxExtent (and never past the far side of the circle). scale:
    // 'shared' -- one y scale for all three segments (over the shown ranges), so their heights compare; 'segment' -- each fills the plot.
    // The marker hides while the dot is off the three segments (dragged, or heading back from where it was released); it shows again from
    // the next activation on. Sizes and colors: the slide's CSS variables. The button at the plot's left pauses and resumes the motion.
    // yLabel: the y-axis label, upright, above the plot at its left edge (the plot's font: --size-small; '' hides it), as on the basins
    // slide (js/basins.js); axisLabelGap: between it and the plot, in label ems.
    plot: { riseTo: 1, pastRise: 0.1, maxExtent: 1.5, scale: 'shared', headroom: 1.06, yLabel: 'Density \u2193', axisLabelGap: 0.5 }
  };
  const DATA = window.TRI_SPHERE;
  const slide = document.querySelector(IDS.slide);
  const inputCanvas = document.querySelector(IDS.input), outputCanvas = document.querySelector(IDS.output), plotCanvas = document.querySelector(IDS.plot);
  const playButton = document.querySelector(IDS.play);   // optional
  if (!DATA || !slide || !inputCanvas || !outputCanvas || !plotCanvas) return;

  const decode = (s, Type) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new Type(u.buffer); };
  // Grid: rows latitude -90..90, columns longitude -180..180 (the last column repeats the first); vertex (i, j) is i * NLON + j.
  const NLAT = DATA.grid.lat.count, NLON = DATA.grid.lon.count, n = NLAT * NLON;
  const LAT0 = DATA.grid.lat.min, DLAT = (DATA.grid.lat.max - LAT0) / (NLAT - 1), LON0 = DATA.grid.lon.min, DLON = (DATA.grid.lon.max - LON0) / (NLON - 1);
  const deg = Math.PI / 180;
  // Input sphere in the paper's chart frame (pole, first, second): (cos lat cos lon, cos lat sin lon, sin lat).
  const sphere = new Float32Array(3 * n);
  for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) {
    const la = (LAT0 + i * DLAT) * deg, lo = (LON0 + j * DLON) * deg;
    sphere.set([Math.cos(la) * Math.cos(lo), Math.cos(la) * Math.sin(lo), Math.sin(la)], 3 * (i * NLON + j));
  }
  const image = (() => { const q = decode(DATA.proj, Int16Array), out = new Float32Array(q.length); for (let k = 0; k < q.length; k++) out[k] = q[k] / DATA.coordScale; return out; })();
  const logArea = decode(DATA.logArea, Int16Array);
  // Triangles: two per grid cell, leaving out the ones collapsed to a segment at the poles (where a row is a single point).
  const faceList = [];
  for (let i = 0; i < NLAT - 1; i++) for (let j = 0; j < NLON - 1; j++) {
    const a = i * NLON + j, b = a + NLON, c = b + 1, d = a + 1;
    if (i + 1 < NLAT - 1) faceList.push(a, b, c);
    if (i > 0) faceList.push(a, c, d);
  }
  const faces = new Uint32Array(faceList), nf = faces.length / 3;
  // Colors: a 256-entry palette from the colormap, and each face's palette index (the mean log10 area of its three vertices).
  const cmap = CONFIG.colormap, last = cmap.length - 1;
  const palette = Array.from({ length: 256 }, (_, i) => {
    const u = i / 255 * last, k = Math.min(last - 1, Math.floor(u)), f = u - k;
    return `rgb(${cmap[k].map((c, j) => Math.round(c + (cmap[k + 1][j] - c) * f)).join(',')})`;
  });
  const [lo, hi] = DATA.colorRange, faceBins = new Uint8Array(nf);
  for (let f = 0; f < nf; f++) {
    const v = (logArea[faces[3 * f]] + logArea[faces[3 * f + 1]] + logArea[faces[3 * f + 2]]) / (3 * DATA.areaScale);
    faceBins[f] = Math.round(Math.max(0, Math.min(1, (v - lo) / (hi - lo))) * 255);
  }
  const extentOf = p => { let m = 0; for (let k = 0; k < p.length; k += 3) m = Math.max(m, Math.hypot(p[k], p[k + 1], p[k + 2])); return CONFIG.extentMargin * m; };

  // The dot's image: bilinear interpolation of the image grid at the dot's latitude and longitude.
  function imageAt(v) {
    const la = Math.asin(Math.max(-1, Math.min(1, v[2]))) / deg, lo = Math.atan2(v[1], v[0]) / deg;
    const fi = Math.max(0, Math.min(NLAT - 1 - 1e-9, (la - LAT0) / DLAT)), fj = Math.max(0, Math.min(NLON - 1 - 1e-9, (lo - LON0) / DLON));
    const i = Math.floor(fi), j = Math.floor(fj), u = fi - i, w = fj - j, out = [0, 0, 0];
    for (const [di, dj, c] of [[0, 0, (1 - u) * (1 - w)], [1, 0, u * (1 - w)], [0, 1, (1 - u) * w], [1, 1, u * w]]) {
      const k = 3 * ((i + di) * NLON + j + dj);
      out[0] += c * image[k]; out[1] += c * image[k + 1]; out[2] += c * image[k + 2];
    }
    return out;
  }

  const rotX = a => [1, 0, 0, 0, Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a)];
  const rotY = a => [Math.cos(a), 0, Math.sin(a), 0, 1, 0, -Math.sin(a), 0, Math.cos(a)];
  const mul = (A, B) => { const M = []; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) M.push(A[3*i]*B[j] + A[3*i+1]*B[3+j] + A[3*i+2]*B[6+j]); return M; };
  const apply = (M, v) => [M[0]*v[0] + M[1]*v[1] + M[2]*v[2], M[3]*v[0] + M[4]*v[1] + M[5]*v[2], M[6]*v[0] + M[7]*v[1] + M[8]*v[2]];
  const applyT = (M, v) => [M[0]*v[0] + M[3]*v[1] + M[6]*v[2], M[1]*v[0] + M[4]*v[1] + M[7]*v[2], M[2]*v[0] + M[5]*v[1] + M[8]*v[2]];
  const norm = v => { const l = Math.hypot(...v) || 1; return v.map(x => x / l); };
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  // Both frames -> screen: first (a2) right, second (a3) up, pole (a1) toward the viewer.
  const BASE = [0, 1, 0, 0, 0, 1, 1, 0, 0];
  // Home view: the shortest rotation that brings the chosen centre (screen coordinates through BASE) to the viewer (screen +z), then the
  // yaw/pitch offsets. 'centroid' is the pole (1, 0, 0) by the paper's construction, so it needs no rotation.
  const home = (() => {
    const c = CONFIG.view.center === 'directions' ? norm(DATA.activations.map(norm).reduce((s, a) => s.map((v, k) => v + a[k]), [0, 0, 0])) : [1, 0, 0];
    const [x, y, z] = apply(BASE, c), axis = [y, -x, 0], s = Math.hypot(...axis), angle = Math.atan2(s, z);   // axis = c x (0, 0, 1)
    let R = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    if (s > 1e-12) {
      const [u, v, w] = axis.map(a => a / s), co = Math.cos(angle), si = Math.sin(angle), t = 1 - co;
      R = [t*u*u + co, t*u*v - si*w, t*u*w + si*v, t*u*v + si*w, t*v*v + co, t*v*w - si*u, t*u*w - si*v, t*v*w + si*u, t*w*w + co];
    }
    return mul(mul(rotX(CONFIG.view.pitch), rotY(CONFIG.view.yaw)), R);
  })();
  const css = (canvas, name, fallback) => parseFloat(getComputedStyle(canvas).getPropertyValue(name)) || fallback;
  const cross = (ctx, x, y, r, w) => {
    const d = r / Math.SQRT2;
    ctx.beginPath(); ctx.moveTo(x - d, y - d); ctx.lineTo(x + d, y + d); ctx.moveTo(x - d, y + d); ctx.lineTo(x + d, y - d);
    ctx.lineCap = 'round';
    ctx.strokeStyle = CONFIG.markHalo; ctx.lineWidth = 2 * w; ctx.stroke();
    ctx.strokeStyle = CONFIG.markColor; ctx.lineWidth = w; ctx.stroke();
  };
  const disc = (ctx, x, y, r, color) => {   // filled with color, outlined in markColor (reads on both ends of magma and on the page)
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = color; ctx.fill();
    ctx.lineWidth = 0.3 * r; ctx.strokeStyle = CONFIG.markColor; ctx.stroke();
  };

  // Motion state: the dot (unit vector in the input frame), the start of its current leg, and the activation it is heading to.
  const acts = DATA.activations.map(norm);
  let dotPos = acts[0].slice(), from = acts[0].slice(), target = 1, legT = 0, holdUntil = 0;
  let legFrom = 0;     // the activation the current leg started at (-1: the leg started where the dot was released, off the three segments)
  let mode = 'auto';   // 'auto' (moving), 'drag' (held), 'wait' (released, resumeDelay running)

  // One rotatable orthographic panel. Its surface is painted into an offscreen canvas only when the rotation or size changes; every
  // frame of the dot's motion just copies that and draws the marks on top.
  function panel(canvas, points, opts) {
    const ctx = canvas.getContext('2d'), cache = document.createElement('canvas'), cctx = cache.getContext('2d');
    const screen = new Float32Array(3 * n), depth = new Float32Array(nf), order = new Uint32Array(nf), key = new Int32Array(nf);
    const slabs = opts.cull ? 1 : CONFIG.depthSlabs, counts = new Uint32Array(slabs * 256 + 1), extent = opts.extent;
    const p = { canvas, rotation: home.slice(), stale: true, ms: 0, drawMs: 0 };
    p.view = () => mul(CONFIG.lockRotation ? shared.rotation : p.rotation, BASE);
    p.geometry = () => {
      const W = canvas.width, H = canvas.height, dpr = W / (canvas.getBoundingClientRect().width || 1);
      const tiny = css(canvas, '--mark-tiny', 5) * dpr, small = css(canvas, '--mark-small', 9) * dpr;
      const s = (Math.min(W, H) / 2 - small) / extent;
      return { W, H, dpr, tiny, small, s, cx: W / 2, cy: H / 2 };
    };
    function paintSurface(g, M) {
      const t0 = performance.now();
      if (cache.width !== g.W || cache.height !== g.H) { cache.width = g.W; cache.height = g.H; }
      const c = cctx;
      for (let k = 0; k < n; k++) { const x = points[3*k], y = points[3*k+1], z = points[3*k+2]; screen[3*k] = g.cx + (M[0]*x + M[1]*y + M[2]*z) * g.s; screen[3*k+1] = g.cy - (M[3]*x + M[4]*y + M[5]*z) * g.s; screen[3*k+2] = M[6]*x + M[7]*y + M[8]*z; }
      c.clearRect(0, 0, g.W, g.H);
      if (opts.cull) { c.strokeStyle = CONFIG.guideColor; c.lineWidth = CONFIG.guideWidth * g.dpr; c.beginPath(); c.arc(g.cx, g.cy, g.s, 0, 2 * Math.PI); c.stroke(); }
      for (let f = 0; f < nf; f++) depth[f] = screen[3 * faces[3 * f] + 2] + screen[3 * faces[3 * f + 1] + 2] + screen[3 * faces[3 * f + 2] + 2];
      // Paint order: one counting sort on (depth slab, color): back to front in `slabs` slices, grouped by color within a slice so a
      // whole color run shares one fill; the convex globe uses a single slice of its near side.
      let dlo = Infinity, dhi = -Infinity;
      if (!opts.cull) for (let f = 0; f < nf; f++) { if (depth[f] < dlo) dlo = depth[f]; if (depth[f] > dhi) dhi = depth[f]; }
      const slabScale = opts.cull ? 0 : (slabs - 1e-6) / Math.max(dhi - dlo, 1e-9);
      counts.fill(0);
      for (let f = 0; f < nf; f++) if (!opts.cull || depth[f] > -0.05) { key[f] = Math.floor((depth[f] - dlo) * slabScale || 0) * 256 + faceBins[f]; counts[key[f] + 1]++; } else key[f] = -1;
      for (let b = 1; b < counts.length; b++) counts[b] += counts[b - 1];
      let m = 0;
      for (let f = 0; f < nf; f++) if (key[f] >= 0) { order[counts[key[f]]++] = f; m++; }
      c.globalAlpha = CONFIG.faceAlpha;
      const grow = CONFIG.seam * g.dpr;
      let bin = -1, len = 0;
      for (let i = 0; i < m; i++) {
        const f = order[i];
        if (faceBins[f] !== bin || len === 512) { if (bin >= 0) c.fill(); bin = faceBins[f]; c.fillStyle = palette[bin]; c.beginPath(); len = 0; }
        len++;
        let a = 3 * faces[3 * f], b = 3 * faces[3 * f + 1], d = 3 * faces[3 * f + 2];
        if ((screen[b] - screen[a]) * (screen[d + 1] - screen[a + 1]) - (screen[b + 1] - screen[a + 1]) * (screen[d] - screen[a]) < 0) { const t = b; b = d; d = t; }   // same winding for all
        const gx = (screen[a] + screen[b] + screen[d]) / 3, gy = (screen[a + 1] + screen[b + 1] + screen[d + 1]) / 3;
        const out = (v, first) => { const dx = screen[v] - gx, dy = screen[v + 1] - gy, k = 1 + grow / (Math.hypot(dx, dy) || 1); first ? c.moveTo(gx + dx * k, gy + dy * k) : c.lineTo(gx + dx * k, gy + dy * k); };
        out(a, true); out(b, false); out(d, false); c.closePath();
      }
      if (bin >= 0) c.fill();
      c.globalAlpha = 1;
      p.ms = performance.now() - t0;
    }
    p.draw = () => {
      if (canvas.width <= 1) return;
      const t0 = performance.now(), g = p.geometry(), M = p.view();
      if (p.stale) { paintSurface(g, M); p.stale = false; }
      ctx.clearRect(0, 0, g.W, g.H);
      ctx.drawImage(cache, 0, 0);
      const toScreen = v => { const q = apply(M, v); return [g.cx + q[0] * g.s, g.cy - q[1] * g.s, q[2]]; };
      for (const v of opts.marks()) { const [x, y, z] = toScreen(v); ctx.globalAlpha = opts.cull && z < 0 ? CONFIG.farAlpha : 1; cross(ctx, x, y, g.tiny / 2, g.tiny * CONFIG.markStroke); }
      const [x, y, z] = toScreen(opts.dot());
      ctx.globalAlpha = opts.cull && z < 0 ? CONFIG.farAlpha : 1;
      disc(ctx, x, y, g.small / 2, getComputedStyle(canvas).getPropertyValue('--trisphere-dot').trim() || '#ffffff');
      ctx.globalAlpha = 1;
      p.dotScreen = [x, y, z];
      p.drawMs = performance.now() - t0;
    };
    return p;
  }
  const shared = { rotation: home.slice() };
  const inputPanel = panel(inputCanvas, sphere, { extent: CONFIG.extentMargin, cull: true, marks: () => acts, dot: () => dotPos });
  // The output panel's centre of rotation (CONFIG.outputCenter), subtracted from everything it draws.
  const center = [0, 0, 0];
  if (CONFIG.outputCenter === 'surface') for (let j = 0; j < 3; j++) { let a = Infinity, b = -Infinity; for (let k = j; k < image.length; k += 3) { if (image[k] < a) a = image[k]; if (image[k] > b) b = image[k]; } center[j] = (a + b) / 2; }
  const centered = image.map((v, k) => v - center[k % 3]), shift = v => v.map((x, j) => x - center[j]);
  const outputPanel = panel(outputCanvas, centered, { extent: extentOf(centered), marks: () => DATA.images.map(shift), dot: () => shift(imageAt(dotPos)) });
  const panels = [inputPanel, outputPanel];

  // Speed plot under the output panel: the current segment's exact speed profile along its great circle (DATA.segments: segment k runs
  // from activation k (x = 0) to k + 1 (x = 1); x = angle / segment angle, samples every dx from x0[k]), spanning the width of the output
  // panel's drawing square (the farthest the surface can reach), with the dot's place on it.
  const SEG = DATA.segments, DX = SEG.dx;
  const speeds = SEG.arc, xAt = (k, i) => SEG.x0[k] + i * DX, indexAt = (k, x) => Math.round((x - SEG.x0[k]) / DX);
  // Shown range [lo, hi] per segment (see CONFIG.plot).
  const ranges = speeds.map((v, k) => {
    const level = CONFIG.plot.riseTo * v[indexAt(k, 0.5)], n = v.length;
    const reach = (start, dir) => {   // segment lengths past the activation at index start until the speed is back at level (or the cap)
      for (let i = start + dir; i >= 0 && i < n; i += dir) {
        const past = Math.abs(xAt(k, i) - xAt(k, start));
        if (past > CONFIG.plot.maxExtent) break;
        if (v[i] >= level) return Math.min(CONFIG.plot.maxExtent, past + CONFIG.plot.pastRise);
      }
      return CONFIG.plot.maxExtent;
    };
    const i0 = indexAt(k, 0), i1 = indexAt(k, 1);
    let before = Math.min(reach(i0, -1), -xAt(k, 0)), after = Math.min(reach(i1, 1), xAt(k, n - 1) - 1);
    const room = xAt(k, n - 1) - xAt(k, 0) - 1;   // the rest of the circle, in segment lengths: the two continuations may not overlap
    if (before + after > room) { const f = room / (before + after); before *= f; after *= f; }
    return [-before, 1 + after];
  });
  const peak = k => { const v = speeds[k], [lo, hi] = ranges[k]; let m = 0; for (let i = 0; i < v.length; i++) { const x = xAt(k, i); if (x >= lo && x <= hi && v[i] > m) m = v[i]; } return m; };
  const sharedMax = Math.max(peak(0), peak(1), peak(2));
  const plot = { canvas: plotCanvas, ctx: plotCanvas.getContext('2d'), marker: null };
  plot.draw = () => {
    const c = plot.canvas, ctx = plot.ctx, W = c.width, H = c.height;
    if (W <= 1) return;
    const k = legFrom >= 0 ? legFrom : (target + acts.length - 1) % acts.length, v = speeds[k], [lo, hi] = ranges[k];
    // The row holding the play button and this canvas is sized (in resize) so the curve spans the output panel's drawing square.
    const g = outputPanel.geometry(), pad = g.small / 2 + 1, x0 = pad, x1 = W - pad;
    const style = getComputedStyle(c), font = parseFloat(style.fontSize) * g.dpr;
    const labelH = CONFIG.plot.yLabel ? font * (1.2 + CONFIG.plot.axisLabelGap) : 0;
    const top = pad + labelH, base = H - pad, ymax = (CONFIG.plot.scale === 'segment' ? peak(k) : sharedMax) * CONFIG.plot.headroom;
    const X = x => x0 + (x - lo) / (hi - lo) * (x1 - x0), Y = s => base - s / ymax * (base - top);
    ctx.clearRect(0, 0, W, H);
    if (CONFIG.plot.yLabel) {
      ctx.font = `${font}px ${style.fontFamily}`; ctx.fillStyle = style.getPropertyValue('--trisphere-curve').trim() || CONFIG.markColor;
      ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(CONFIG.plot.yLabel, x0 - pad / 2, 0);
    }
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.strokeStyle = style.getPropertyValue('--trisphere-axis').trim() || CONFIG.guideColor; ctx.lineWidth = CONFIG.guideWidth * g.dpr;
    ctx.beginPath(); ctx.moveTo(x0, base); ctx.lineTo(x1, base); ctx.stroke();
    ctx.strokeStyle = style.getPropertyValue('--trisphere-curve').trim() || CONFIG.markColor;
    ctx.lineWidth = (parseFloat(style.getPropertyValue('--trisphere-curve-width')) || 2) * g.dpr;
    const m = parseFloat(style.getPropertyValue('--trisphere-curve-muted-alpha')), muted = Number.isFinite(m) ? m : 0.3;
    const i0 = indexAt(k, 0), i1 = indexAt(k, 1), ilo = Math.max(0, Math.ceil((lo - SEG.x0[k]) / DX - 1e-9)), ihi = Math.min(v.length - 1, Math.floor((hi - SEG.x0[k]) / DX + 1e-9));
    // Three runs of the same curve: past the start activation (muted), past the end activation (muted), the segment itself.
    for (const [from, to, alpha] of [[ilo, i0, muted], [i1, ihi, muted], [i0, i1, 1]]) {
      ctx.globalAlpha = alpha; ctx.beginPath();
      for (let i = from; i <= to; i++) { const x = X(xAt(k, i)), y = Y(v[i]); i === from ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    plot.marker = null;
    if (legFrom < 0 || mode === 'drag') return;
    // The dot's place on the segment: its angle from the start activation, in segment angles.
    const x = Math.acos(Math.max(-1, Math.min(1, dot(acts[k], dotPos)))) / SEG.angle[k];
    const fi = (x - SEG.x0[k]) / DX, i = Math.max(0, Math.min(v.length - 2, Math.floor(fi))), f = fi - i, s = v[i] + (v[i + 1] - v[i]) * f;
    disc(ctx, X(x), Y(s), g.small / 2, getComputedStyle(inputCanvas).getPropertyValue('--trisphere-dot').trim() || '#ffffff');
    plot.marker = { segment: k, x, speed: s };
  };

  // Redraws: batched into one animation frame, never while the slide is hidden.
  let queued = false;
  function render() { queued = false; if (slide.hidden) return; panels.forEach(p => p.draw()); plot.draw(); }
  function schedule() { if (!queued) { queued = true; requestAnimationFrame(render); } }
  const plotRow = plotCanvas.parentElement;   // button | plot | spacer (see .trisphere-plot-row)
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    for (const p of [...panels, plot]) { const r = p.canvas.getBoundingClientRect(); p.canvas.width = Math.max(1, Math.round(r.width * dpr)); p.canvas.height = Math.max(1, Math.round(r.height * dpr)); p.stale = true; }
    // The plot spans the output panel's drawing square (plus room for the marker at its ends), with the button at its left.
    if (plotRow !== outputCanvas.parentElement && outputCanvas.width > 1) {
      const g = outputPanel.geometry(), sq = (Math.min(g.W, g.H) - 2 * g.small + 2 * (g.small / 2 + 1)) / dpr;   // square + the plot's end pads
      plotRow.style.width = `calc(${sq.toFixed(1)}px + 2 * (1.2em + var(--trisphere-play-gap)))`;
    }
    schedule();
  }
  for (const c of [inputCanvas, outputCanvas, plotCanvas]) new ResizeObserver(resize).observe(c);
  document.fonts.ready.then(() => schedule());   // the plot's label is drawn in the slide's font

  // The motion loop: advances the dot along its leg; runs only while the slide and tab are shown, the dot is free, and the slide was
  // touched or shown less than idleStop ms ago.
  let running = false, lastTick = 0, wokeAt = performance.now(), resumeTimer = 0;
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function step(dt, now) {
    if (now < holdUntil) return;
    const e = acts[target], { path, speed, hold } = CONFIG.motion;
    if (path === 'arc') {
      const theta = Math.acos(Math.max(-1, Math.min(1, dot(from, e))));
      legT = theta < 1e-9 ? 1 : Math.min(1, legT + speed * dt / theta);
      const s = Math.sin(theta);
      dotPos = s < 1e-9 ? e.slice() : norm(from.map((v, k) => (Math.sin((1 - legT) * theta) * v + Math.sin(legT * theta) * e[k]) / s));
    } else {
      const len = Math.hypot(e[0] - from[0], e[1] - from[1], e[2] - from[2]);
      legT = len < 1e-9 ? 1 : Math.min(1, legT + speed * dt / len);
      const q = from.map((v, k) => (1 - legT) * v + legT * e[k]);
      if (Math.hypot(...q) > 1e-6) dotPos = norm(q);   // (a leg between opposite points passes the origin: keep the last direction there)
    }
    if (legT >= 1) { dotPos = e.slice(); from = e.slice(); legFrom = target; target = (target + 1) % acts.length; legT = 0; holdUntil = now + hold; }
  }
  // The play button (the deck's .slider-play): pause icon while the dot moves (pressing pauses it, and it stays paused, with no idle
  // restart, until pressed again), play icon otherwise (pressing sets it moving from where it is toward the next activation).
  const syncButton = () => { if (!playButton) return; playButton.classList.toggle('playing', running); playButton.setAttribute('aria-label', running ? 'Pause' : 'Play'); };
  function tick(now) {
    if (!running) return;
    if (slide.hidden || document.hidden || mode !== 'auto' || now - wokeAt > CONFIG.idleStop) { running = false; syncButton(); return; }
    const dt = Math.min(0.1, (now - lastTick) / 1000);
    if (dt >= 1 / CONFIG.maxFps - 0.002) { step(dt, now); lastTick = now; render(); }
    requestAnimationFrame(tick);
  }
  let pressedPlay = false;   // the play button was pressed: play even with prefers-reduced-motion (which otherwise keeps the dot still)
  function start() {
    if (running || (reduced && !pressedPlay) || slide.hidden || document.hidden || mode !== 'auto') return;
    running = true; lastTick = performance.now(); syncButton(); requestAnimationFrame(tick);
  }
  const wake = () => { wokeAt = performance.now(); start(); };
  // Setting off again after the dot was dragged: the next leg runs from where it is to the activation it was heading to (or the one
  // after, if it was left on it). Without a drag (a plain pause), the leg just carries on.
  let offPath = false;
  function go() {
    clearTimeout(resumeTimer);
    if (offPath) {
      legFrom = -1;
      if (dot(dotPos, acts[target]) > 1 - 1e-9) { legFrom = target; target = (target + 1) % acts.length; }
      from = dotPos.slice(); legT = 0; holdUntil = 0; offPath = false;
    }
    mode = 'auto'; wake();
  }
  function release() {   // the dot was let go: it sets off resumeDelay ms later, unless the motion was paused with the button
    clearTimeout(resumeTimer);
    if (userPaused) { mode = 'paused'; return; }
    mode = 'wait'; resumeTimer = setTimeout(go, CONFIG.motion.resumeDelay);
  }
  let userPaused = false;
  if (playButton) playButton.addEventListener('click', e => {
    if (e.detail > 0) playButton.blur();   // after a mouse click, so the arrow keys go back to changing slides
    if (mode === 'drag') return;
    if (running) { userPaused = true; mode = 'paused'; clearTimeout(resumeTimer); running = false; syncButton(); render(); }
    else { userPaused = false; pressedPlay = true; go(); }
  });

  // Pointer: on the left panel, pressing on (or near) the dot drags it over the sphere; anywhere else on either panel rotates.
  for (const p of panels) {
    let drag = null;
    const local = e => { const r = p.canvas.getBoundingClientRect(), k = p.canvas.width / (r.width || 1); return [(e.clientX - r.left) * k, (e.clientY - r.top) * k]; };
    const dragDot = e => {
      const g = p.geometry(), [px, py] = local(e);
      let x = (px - g.cx) / g.s, y = -(py - g.cy) / g.s; const r = Math.hypot(x, y);
      if (r > 1) { x /= r; y /= r; }
      dotPos = norm(applyT(p.view(), [x, y, Math.sqrt(Math.max(0, 1 - x * x - y * y))]));
      schedule();
    };
    p.canvas.addEventListener('pointerdown', e => {
      e.preventDefault(); p.canvas.setPointerCapture(e.pointerId); wake();
      const [px, py] = local(e), d = p.dotScreen, g = p.geometry();
      if (p === inputPanel && d && d[2] > -0.05 && Math.hypot(px - d[0], py - d[1]) <= g.small / 2 + CONFIG.grabSlop * g.dpr) {
        drag = { dot: true }; mode = 'drag'; legFrom = -1; offPath = true; clearTimeout(resumeTimer); p.canvas.style.cursor = 'grabbing'; dragDot(e);
      } else { drag = { x: e.clientX, y: e.clientY }; p.canvas.style.cursor = 'grabbing'; }
    });
    p.canvas.addEventListener('pointermove', e => {
      if (!drag) {   // hovering the dot shows that it can be grabbed
        if (p === inputPanel) { const d = p.dotScreen, g = p.geometry(), [px, py] = local(e); p.canvas.style.cursor = d && d[2] > -0.05 && Math.hypot(px - d[0], py - d[1]) <= g.small / 2 + CONFIG.grabSlop * g.dpr ? 'pointer' : ''; }
        return;
      }
      if (drag.dot) { dragDot(e); return; }
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag = { x: e.clientX, y: e.clientY };
      const turn = R => mul(mul(rotX(dy * CONFIG.dragSpeed), rotY(dx * CONFIG.dragSpeed)), R);
      if (CONFIG.lockRotation) { shared.rotation = turn(shared.rotation); panels.forEach(q => { q.stale = true; }); }
      else { p.rotation = turn(p.rotation); p.stale = true; }
      wokeAt = performance.now(); schedule();
    });
    const end = e => {
      if (drag?.dot) release();
      drag = null; p.canvas.style.cursor = ''; if (p.canvas.hasPointerCapture(e.pointerId)) p.canvas.releasePointerCapture(e.pointerId);
    };
    p.canvas.addEventListener('pointerup', end); p.canvas.addEventListener('pointercancel', end);
  }

  new MutationObserver(() => { if (!slide.hidden) { resize(); wake(); } }).observe(slide, { attributes: true, attributeFilter: ['hidden'] });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) wake(); });

  window.triSphereState = () => ({ config: CONFIG, mode, running, target, legT, dot: dotPos.slice(), dotImage: imageAt(dotPos),
    distanceToActivations: acts.map(a => Math.acos(Math.max(-1, Math.min(1, dot(a, dotPos)))) / deg),
    dotScreen: { input: inputPanel.dotScreen, output: outputPanel.dotScreen }, view: { input: inputPanel.view(), output: outputPanel.view() },
    surfaceMs: { input: inputPanel.ms, output: outputPanel.ms }, frameMs: { input: inputPanel.drawMs, output: outputPanel.drawMs },
    faces: nf, tokens: DATA.tokens, blocks: DATA.blocks, legFrom, userPaused, button: playButton?.getAttribute('aria-label'),
    plot: plot.marker, plotRanges: ranges });   // for console inspection (degrees; surfaceMs: last full repaint, frameMs: last frame; plot: the marker's segment, x along it (0..1) and speed, null when hidden; plotRanges: the shown x range per segment)
  if (!slide.hidden) { resize(); wake(); }
})();
