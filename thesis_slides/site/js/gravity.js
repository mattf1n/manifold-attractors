// Slide after "Activations inhabit separate attracting basins": the rubber-sheet (trampoline) picture of gravity. A round elastic
// sheet, seen in perspective, sags into a well under each heavy ball (CONFIG.masses); light marbles (CONFIG.planets) roll on it,
// pulled downhill by its slope, so they circle in the wells. The sheet is colored by its height with the deck's density colormap
// (magma, as on the probe and three-activation slides): the deep wells are the dark basins.
// Physics, in the sheet's plane (units: sheet radii, seconds): the height is a sum of softened wells,
// h(x, y) = -sum depth / sqrt(1 + d^2 / width^2) (d: the distance to that heavy ball), and a marble's acceleration is
// -gravity * grad h, integrated with velocity Verlet in fixed substeps. The heavy balls stay put; the marbles neither pull on
// anything nor dent the sheet, and they bounce off its rim.
// Drag the sheet to turn it (around freely; the elevation of the view is clamped); drag a marble to carry it over the sheet and let
// go to fling it. The sheet is painted (back to front in depth slices, grouped by color) into an offscreen canvas only when the
// view or size changes; each frame copies it, draws the trails and balls, and repaints only the few cells of sheet in front of a
// ball (a well's near rim hiding part of what is inside, in low views).
// The motion sleeps while the slide or tab is hidden and stops idleStop ms after the slide was shown or last touched (touching it
// wakes it); with prefers-reduced-motion the marbles stay still (a dragged one stays where it is dropped). Showing the slide puts
// the marbles back on their starting orbits (CONFIG.resetOnShow).
// Edit CONFIG to tweak; sizes, line widths and line colors are CSS variables on #gravity-slide (index.html).
(() => {
  const IDS = { slide: '#gravity-slide', canvas: '#gravity-canvas' };
  const CONFIG = {
    // matplotlib 'magma' (the deck's density colormap, as in js/trisphere.js), 33 evenly spaced stops, interpolated linearly. The
    // sheet's lowest point gets the first (dark) stop, its highest the last; colorGamma bends the mapping:
    // t = ((h - lowest) / (highest - lowest)) ^ colorGamma (below 1: only the bottoms of the wells stay dark; above 1: wider basins).
    colormap: [[0,0,4],[3,3,18],[10,8,34],[19,13,52],[29,17,71],[41,17,90],[54,16,107],[68,15,118],[81,18,124],[93,23,127],[106,28,129],
      [118,33,129],[131,38,129],[144,42,129],[156,46,127],[170,51,125],[183,55,121],[196,60,117],[208,65,111],[220,72,105],[231,82,99],
      [239,93,94],[245,107,92],[249,121,93],[252,137,97],[253,152,105],[254,167,114],[254,182,124],[254,196,136],[254,211,149],
      [253,226,163],[252,240,178],[252,253,191]],
    colorGamma: 1,
    // Heavy balls (fixed): they make the wells. x, y: where on the sheet (sheet radii from its center; at yaw 0, +x is right and +y
    // away from the viewer). depth: how far the sheet sags under it; width: the radius of the well's round bottom (the softening;
    // smaller = a sharper funnel). color: RGB fill; size: the CSS variable holding its diameter (both used only with drawMasses).
    masses: [
      { x: -0.2, y: 0.15, depth: 0.5, width: 0.16, color: [236, 160, 32], size: '--gravity-mass-large' },
      { x: 0.58, y: -0.42, depth: 0.32, width: 0.09, color: [214, 40, 57], size: '--gravity-mass-small' }
    ],
    drawMasses: false,        // draw the heavy balls themselves (false: only the wells they make show)
    // Drawn heavy balls go over their wells' walls even where, in low views, the near wall of a steep well would hide part of a ball
    // this small sitting at its bottom (false: hidden like the marbles).
    massesOnTop: true,
    // Marbles. Each starts `distance` from heavy ball number `around` (0 = the first above), in the direction `angle` (degrees,
    // counterclockwise from +x), moving at right angles to that at `speed` times the speed of a circular orbit there (below 1 it
    // falls into an oval that turns as it goes round, a rosette); direction: 1 counterclockwise, -1 clockwise (seen from above).
    // The first two stay in their wells (checked over 15 simulated minutes). The white one is the wanderer: it circles the small
    // ball a few times, then escapes over the saddle between the wells (about 8 s after it starts, about 6 s after the slide is
    // shown, as the trails' lead-in runs first) and falls into the big ball's well. The motion is deterministic (fixed substeps), so
    // this happens every time. cycle: seconds (simulated) after which a marble fades out and starts over from its starting point
    // (0 or absent: never), so the escape repeats while presenting; flinging a marble restarts its cycle. Moving the heavy balls or
    // changing the wells changes all of these orbits (the stable ones can start wandering, the wanderer can stay or leave later).
    planets: [
      { around: 0, distance: 0.18, angle: 90, speed: 1, direction: 1, color: [42, 111, 219], size: '--gravity-planet' },
      { around: 0, distance: 0.3, angle: 145, speed: 0.8, direction: 1, color: [40, 160, 110], size: '--gravity-planet' },
      { around: 1, distance: 0.14, angle: 210, speed: 1.1, direction: 1, color: [255, 255, 255], size: '--gravity-planet', cycle: 20 }
    ],
    respawnFade: 0.6,         // seconds over which a cycling marble (and its trail) fades out before starting over and fades back in
    physics: {
      gravity: 1.2,           // acceleration per unit of slope (sheet radii / s^2): more = faster orbits (same shapes)
      timeScale: 1,           // simulated seconds per real second
      substep: 1 / 240,       // fixed integration step (simulated seconds); smaller = more accurate, more work
      maxSpeed: 3,            // sheet radii per second; faster marbles (a hard fling) are slowed to this
      rim: 0.97,              // the marbles' centers stay within this radius; they bounce off it
      rimBounce: 0.9          // fraction of the speed into the rim kept when they bounce off it (1 = elastic)
    },
    resetOnShow: true,        // showing the slide puts the marbles back where CONFIG.planets starts them
    // Trails behind the marbles: `seconds` of simulated path, faded toward the tail (alpha at the marble; 0 hides them). Width: CSS.
    trail: { seconds: 1.6, alpha: 0.75, samplesPerSecond: 60, fadeSteps: 8 },
    // The sheet's mesh: rings x spokes cells (polar, so the rim is round); more = smoother wells, slower repaints while turning.
    mesh: { rings: 48, spokes: 160 },
    // The sheet's grid lines (the classic look): straight lines x = k * spacing and y = k * spacing on the flat sheet, following it
    // down into the wells, drawn as pieces `step` long. spacing 0 hides them. Color and width: CSS (--gravity-grid, --gravity-rim).
    grid: { spacing: 0.1, step: 0.02 },
    // Camera: perspective from `distance` sheet radii away (smaller = stronger perspective), aimed at height targetZ above the
    // center of the flat sheet. Starting view: yaw (degrees around the vertical) and elevation (degrees above the sheet's plane);
    // dragging keeps the elevation within [minElevation, maxElevation]. Fit: the sheet is scaled as large as fits the canvas at
    // every yaw for the current elevation (so turning it around never changes its size; tilting it does, smoothly), and the
    // current view's sheet is centered in the canvas (with room for the balls' radius at its edges).
    camera: { distance: 3.5, targetZ: -0.15 },
    view: { yaw: -20, elevation: 40, minElevation: 15, maxElevation: 75 },
    dragSpeed: 0.4,           // degrees of turn per CSS pixel dragged
    // Flinging a marble: its speed on release is its motion over the last `window` ms of the drag. grabSlop: CSS pixels around a
    // marble that still grab it (for fingers).
    fling: { window: 90, grabSlop: 10 },
    occlusionMargin: 3,       // a cell of sheet is drawn over a ball only if it is this many ball radii nearer the camera than its center
    depthSlabs: 96,           // the sheet is painted back to front in this many depth slices (more = finer occlusion, more fills)
    seam: 0.5,                // each cell is grown by this many CSS pixels around its center so neighbors overlap (no hairline seams)
    ballOutline: 0.25,        // the balls' outline width, as a fraction of their radius (color: --gravity-ball-outline)
    idleStop: 90000,          // ms after the slide was shown or last touched before the motion stops (sleeps)
    maxFps: 60                // the marbles redraw at most this often
  };
  const slide = document.querySelector(IDS.slide), canvas = document.querySelector(IDS.canvas);
  if (!slide || !canvas) return;
  const ctx = canvas.getContext('2d'), cache = document.createElement('canvas'), cctx = cache.getContext('2d');
  const deg = Math.PI / 180, TAU = 2 * Math.PI, MASSES = CONFIG.masses;

  // The sheet's height, and a marble's acceleration there (-gravity * the height's gradient).
  function height(x, y) {
    let h = 0;
    for (const m of MASSES) { const dx = x - m.x, dy = y - m.y; h -= m.depth / Math.sqrt(1 + (dx * dx + dy * dy) / (m.width * m.width)); }
    return h;
  }
  function accel(x, y, out) {
    let ax = 0, ay = 0;
    for (const m of MASSES) {
      const dx = x - m.x, dy = y - m.y, w2 = m.width * m.width, s = 1 + (dx * dx + dy * dy) / w2, f = CONFIG.physics.gravity * m.depth / (w2 * s * Math.sqrt(s));
      ax -= f * dx; ay -= f * dy;
    }
    out[0] = ax; out[1] = ay;
    return out;
  }

  // Mesh: vertex 0 is the center, then ring i (1..NR, radius i / NR) spoke j at index 1 + (i - 1) * NS + j. Cell i * NS + j spans
  // rings i..i+1 and spokes j..j+1: a triangle around the center for i = 0 (fourth corner -1), a quad elsewhere.
  const NR = CONFIG.mesh.rings, NS = CONFIG.mesh.spokes, nv = 1 + NR * NS, nc = NR * NS;
  const vid = (i, j) => i === 0 ? 0 : 1 + (i - 1) * NS + j % NS;
  const world = new Float32Array(3 * nv);
  for (let i = 0; i <= NR; i++) for (let j = 0; j < (i ? NS : 1); j++) {
    const r = i / NR, a = j / NS * TAU, x = r * Math.cos(a), y = r * Math.sin(a);
    world.set([x, y, height(x, y)], 3 * vid(i, j));
  }
  const cellV = new Int32Array(4 * nc);
  for (let i = 0; i < NR; i++) for (let j = 0; j < NS; j++) cellV.set(i === 0 ? [0, vid(1, j), vid(1, j + 1), -1] : [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)], 4 * (i * NS + j));
  // Colors: a 256-entry palette from the colormap and each cell's palette index (from the mean height of its corners).
  const cmap = CONFIG.colormap, last = cmap.length - 1;
  const palette = Array.from({ length: 256 }, (_, i) => {
    const u = i / 255 * last, k = Math.min(last - 1, Math.floor(u)), f = u - k;
    return `rgb(${cmap[k].map((c, j) => Math.round(c + (cmap[k + 1][j] - c) * f)).join(',')})`;
  });
  let hLo = Infinity, hHi = -Infinity;
  for (let v = 0; v < nv; v++) { const h = world[3 * v + 2]; if (h < hLo) hLo = h; if (h > hHi) hHi = h; }
  const cellBin = new Uint8Array(nc);
  for (let c = 0; c < nc; c++) {
    let h = 0, k = 0;
    for (let q = 0; q < 4; q++) { const v = cellV[4 * c + q]; if (v >= 0) { h += world[3 * v + 2]; k++; } }
    cellBin[c] = Math.round(Math.pow(Math.max(0, Math.min(1, (h / k - hLo) / (hHi - hLo || 1))), CONFIG.colorGamma) * 255);
  }
  // Grid lines and the rim, as short pieces on the sheet, each filed under the cell its middle lies in (so it is painted with that
  // cell, in depth order). segs: x0, y0, z0, x1, y1, z1 per piece; per cell, its pieces are segIndex[segStart[c]..segStart[c + 1]).
  const cellOf = (x, y) => {
    const i = Math.min(NR - 1, Math.floor(Math.hypot(x, y) * NR)), j = Math.floor(((Math.atan2(y, x) / TAU) + 1) % 1 * NS) % NS;
    return i * NS + j;
  };
  const segList = [], segCellList = [], segRimList = [];
  const addSeg = (x0, y0, x1, y1, rim) => { segList.push(x0, y0, height(x0, y0), x1, y1, height(x1, y1)); segCellList.push(cellOf((x0 + x1) / 2, (y0 + y1) / 2)); segRimList.push(rim ? 1 : 0); };
  const { spacing, step } = CONFIG.grid;
  if (spacing > 0) for (let k = -Math.floor(1 / spacing); k <= Math.floor(1 / spacing); k++) {
    const c = k * spacing, half = Math.sqrt(Math.max(0, 1 - c * c));
    if (half < 1e-6) continue;
    const n = Math.max(1, Math.ceil(2 * half / step));
    for (let q = 0; q < n; q++) {
      const t0 = -half + 2 * half * q / n, t1 = -half + 2 * half * (q + 1) / n;
      addSeg(c, t0, c, t1, false); addSeg(t0, c, t1, c, false);
    }
  }
  for (let j = 0; j < NS; j++) { const a0 = j / NS * TAU, a1 = (j + 1) / NS * TAU; addSeg(Math.cos(a0), Math.sin(a0), Math.cos(a1), Math.sin(a1), true); }
  const segs = new Float32Array(segList), nseg = segCellList.length, segRim = Uint8Array.from(segRimList);
  const segStart = new Uint32Array(nc + 1);
  for (const c of segCellList) segStart[c + 1]++;
  for (let c = 0; c < nc; c++) segStart[c + 1] += segStart[c];
  const segIndex = new Uint32Array(nseg), fillAt = segStart.slice(0, nc);
  segCellList.forEach((c, s) => { segIndex[fillAt[c]++] = s; });

  // View: yaw about the vertical, then elevation; perspective from CONFIG.camera.distance. View coordinates: X right, Y up,
  // Z toward the camera; screen = origin + (X, -Y) * k * scale with k = D / (D - Z).
  const D = CONFIG.camera.distance, TZ = CONFIG.camera.targetZ;
  const view = { yaw: CONFIG.view.yaw * deg, elevation: CONFIG.view.elevation * deg };
  const trig = v => ({ cy: Math.cos(v.yaw), sy: Math.sin(v.yaw), ce: Math.cos(v.elevation), se: Math.sin(v.elevation) });
  let T = trig(view);
  const fit = { scale: 1, ox: 0, oy: 0, dpr: 1 };
  const P = new Float64Array(4);   // project()'s result: screen x, screen y, distance to the camera, perspective factor k
  function toView(x, y, z, t, out) {
    z -= TZ;
    const x1 = t.cy * x - t.sy * y, y1 = t.sy * x + t.cy * y;
    out[0] = x1; out[1] = t.se * y1 + t.ce * z; out[2] = -t.ce * y1 + t.se * z;
    return out;
  }
  const V = new Float64Array(3);
  function project(x, y, z) {
    toView(x, y, z, T, V);
    const k = D / (D - V[2]);
    P[0] = fit.ox + V[0] * k * fit.scale; P[1] = fit.oy - V[1] * k * fit.scale; P[2] = Math.hypot(V[0], V[1], D - V[2]); P[3] = k;
    return P;
  }
  // From view coordinates back to the sheet's (the rotation only).
  function fromView(X, Y, Z, out) {
    const y1 = T.se * Y - T.ce * Z, z1 = T.ce * Y + T.se * Z;
    out[0] = T.cy * X + T.sy * y1; out[1] = -T.sy * X + T.cy * y1; out[2] = z1;
    return out;
  }
  // The point of the sheet under a canvas pixel (device pixels), or null: march along the camera ray, refine by bisection.
  function sheetAt(px, py) {
    const u = (px - fit.ox) / fit.scale, v = -(py - fit.oy) / fit.scale;
    const cam = fromView(0, 0, D, [0, 0, 0]); cam[2] += TZ;
    const dir = fromView(u, v, -D, [0, 0, 0]), len = Math.hypot(...dir);
    const at = t => [cam[0] + dir[0] / len * t, cam[1] + dir[1] / len * t, cam[2] + dir[2] / len * t];
    const above = p => p[2] - height(p[0], p[1]);
    const N = 400, tMax = 2 * D;
    let prev = 0;
    for (let i = 1; i <= N; i++) {
      const t = tMax * i / N, p = at(t);
      if (Math.hypot(p[0], p[1]) <= 1 && above(p) <= 0) {
        let a = prev, b = t;
        for (let k = 0; k < 30; k++) { const m = (a + b) / 2, q = at(m); (Math.hypot(q[0], q[1]) <= 1 && above(q) <= 0) ? b = m : a = m; }
        return at(b);
      }
      prev = t;
    }
    return null;
  }

  // Fit (CONFIG.camera): the scale makes the sheet fit at every yaw for the current elevation (recomputed only when the size or the
  // elevation changes, so turning around never zooms); the origin centers the current view's sheet in the canvas. Both from the
  // sheet's extent over a sample of its points (the rim, a coarse share of the mesh, the wells' bottoms), padded by the largest
  // ball's radius.
  const cssPx = name => parseFloat(getComputedStyle(canvas).getPropertyValue(name)) || 0;
  const fitPts = [];
  for (let i = 2; i <= NR; i += 2) for (let j = 0; j < NS; j += i === NR ? 1 : 4) fitPts.push(world.subarray(3 * vid(i, j), 3 * vid(i, j) + 3));
  for (const m of MASSES) fitPts.push([m.x, m.y, height(m.x, m.y)]);
  const box = new Float64Array(4), out3 = new Float64Array(3);
  function extent(t, b) {   // b: x0, y0, x1, y1 of the view-plane coordinates (X k, Y k)
    b[0] = b[1] = Infinity; b[2] = b[3] = -Infinity;
    for (const p of fitPts) {
      toView(p[0], p[1], p[2], t, out3);
      const k = D / (D - out3[2]), X = out3[0] * k, Y = out3[1] * k;
      if (X < b[0]) b[0] = X; if (X > b[2]) b[2] = X; if (Y < b[1]) b[1] = Y; if (Y > b[3]) b[3] = Y;
    }
    return b;
  }
  let fitElevation = NaN, fitPad = 0;
  function fitView() {
    const W = canvas.width, H = canvas.height;
    if (fitElevation !== view.elevation) {
      let w = 0, h = 0;
      for (let k = 0; k < 36; k++) { extent(trig({ yaw: k / 36 * TAU, elevation: view.elevation }), box); w = Math.max(w, box[2] - box[0]); h = Math.max(h, box[3] - box[1]); }
      fitPad = Math.max(...(CONFIG.drawMasses ? MASSES : []).map(m => cssPx(m.size)), ...CONFIG.planets.map(p => cssPx(p.size))) * fit.dpr / 2 + 2 * fit.dpr;
      fit.scale = Math.max(1e-6, Math.min((W - 2 * fitPad) / w, (H - 2 * fitPad) / h));
      fitElevation = view.elevation;
    }
    extent(T, box);
    fit.ox = W / 2 - fit.scale * (box[0] + box[2]) / 2; fit.oy = H / 2 + fit.scale * (box[1] + box[3]) / 2;
  }
  function layout() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2), r = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(r.width * dpr)); canvas.height = Math.max(1, Math.round(r.height * dpr));
    fit.dpr = dpr; fitElevation = NaN; stale = true;
  }

  // Painting the sheet (when the view or size changes): project the vertices and grid pieces, sort the cells back to front by depth
  // slice and color (one counting sort), fill each color run with one path, and stroke each slice's grid pieces after its cells.
  const sx = new Float32Array(nv), sy = new Float32Array(nv), sd = new Float32Array(nv);
  const segScreen = new Float32Array(4 * nseg);
  const cellDist = new Float32Array(nc), cellBox = new Float32Array(4 * nc), order = new Uint32Array(nc), key = new Uint32Array(nc);
  const slabs = CONFIG.depthSlabs, counts = new Uint32Array(slabs * 256 + 1);
  let stale = true, sheetMs = 0, frameMs = 0, physicsMs = 0;
  const style = () => { const s = getComputedStyle(canvas); return { grid: s.getPropertyValue('--gravity-grid').trim() || 'rgba(255,255,255,0.25)', gridWidth: parseFloat(s.getPropertyValue('--gravity-grid-width')) || 1, rim: s.getPropertyValue('--gravity-rim').trim() || '#202a36', rimWidth: parseFloat(s.getPropertyValue('--gravity-rim-width')) || 1.5, outline: s.getPropertyValue('--gravity-ball-outline').trim() || '#ffffff', trailWidth: parseFloat(s.getPropertyValue('--gravity-trail-width')) || 2 }; };
  let look = null;
  function cellPath(c2d, c, grow) {
    const n = cellV[4 * c + 3] < 0 ? 3 : 4;
    let gx = 0, gy = 0;
    for (let q = 0; q < n; q++) { const v = cellV[4 * c + q]; gx += sx[v]; gy += sy[v]; }
    gx /= n; gy /= n;
    for (let q = 0; q < n; q++) {
      const v = cellV[4 * c + q], dx = sx[v] - gx, dy = sy[v] - gy, k = 1 + grow / (Math.hypot(dx, dy) || 1);
      q ? c2d.lineTo(gx + dx * k, gy + dy * k) : c2d.moveTo(gx + dx * k, gy + dy * k);
    }
    c2d.closePath();
  }
  // Strokes the grid pieces (then the rim pieces) of the cells order[from..to), or of the single cell `only`.
  function strokeSegs(c2d, from, to, only = -1) {
    for (const rim of [0, 1]) {
      let any = false;
      c2d.beginPath();
      for (let i = from; i < to; i++) {
        const c = only >= 0 ? only : order[i];
        for (let q = segStart[c]; q < segStart[c + 1]; q++) {
          const s = segIndex[q];
          if (segRim[s] !== rim) continue;
          c2d.moveTo(segScreen[4 * s], segScreen[4 * s + 1]); c2d.lineTo(segScreen[4 * s + 2], segScreen[4 * s + 3]); any = true;
        }
      }
      if (!any) continue;
      c2d.strokeStyle = rim ? look.rim : look.grid; c2d.lineWidth = (rim ? look.rimWidth : look.gridWidth) * fit.dpr;
      c2d.stroke();
    }
  }
  function paintSheet() {
    const t0 = performance.now(), W = canvas.width, H = canvas.height;
    if (cache.width !== W || cache.height !== H) { cache.width = W; cache.height = H; }
    look = style();
    fitView();
    for (let v = 0; v < nv; v++) { project(world[3 * v], world[3 * v + 1], world[3 * v + 2]); sx[v] = P[0]; sy[v] = P[1]; sd[v] = P[2]; }
    for (let s = 0; s < nseg; s++) {
      project(segs[6 * s], segs[6 * s + 1], segs[6 * s + 2]); segScreen[4 * s] = P[0]; segScreen[4 * s + 1] = P[1];
      project(segs[6 * s + 3], segs[6 * s + 4], segs[6 * s + 5]); segScreen[4 * s + 2] = P[0]; segScreen[4 * s + 3] = P[1];
    }
    let dlo = Infinity, dhi = -Infinity;
    for (let c = 0; c < nc; c++) {
      const n = cellV[4 * c + 3] < 0 ? 3 : 4;
      let d = 0, x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let q = 0; q < n; q++) { const v = cellV[4 * c + q]; d += sd[v]; if (sx[v] < x0) x0 = sx[v]; if (sx[v] > x1) x1 = sx[v]; if (sy[v] < y0) y0 = sy[v]; if (sy[v] > y1) y1 = sy[v]; }
      cellDist[c] = d / n; cellBox[4 * c] = x0; cellBox[4 * c + 1] = y0; cellBox[4 * c + 2] = x1; cellBox[4 * c + 3] = y1;
      if (cellDist[c] < dlo) dlo = cellDist[c]; if (cellDist[c] > dhi) dhi = cellDist[c];   // the stored (float32) values, so no slice index falls below 0
    }
    // Slice 0 is the farthest.
    const slabScale = (slabs - 1e-6) / Math.max(dhi - dlo, 1e-9);
    counts.fill(0);
    for (let c = 0; c < nc; c++) { key[c] = Math.floor((dhi - cellDist[c]) * slabScale) * 256 + cellBin[c]; counts[key[c] + 1]++; }
    for (let b = 1; b < counts.length; b++) counts[b] += counts[b - 1];
    for (let c = 0; c < nc; c++) order[counts[key[c]]++] = c;
    const c2d = cctx, grow = CONFIG.seam * fit.dpr;
    c2d.clearRect(0, 0, W, H);
    c2d.lineCap = 'round';
    let bin = -1, slab = -1, slabFrom = 0, len = 0;
    for (let i = 0; i <= nc; i++) {
      const c = i < nc ? order[i] : -1, s = i < nc ? key[c] >> 8 : -2;
      if (s !== slab || i === nc) {
        if (bin >= 0) c2d.fill();
        if (slab >= 0) strokeSegs(c2d, slabFrom, i);
        slab = s; slabFrom = i; bin = -1;
        if (i === nc) break;
      }
      if (cellBin[c] !== bin || len === 512) { if (bin >= 0) c2d.fill(); bin = cellBin[c]; c2d.fillStyle = palette[bin]; c2d.beginPath(); len = 0; }
      len++;
      cellPath(c2d, c, grow);
    }
    stale = false;
    sheetMs = performance.now() - t0;
  }

  // Balls: the heavy ones (fixed) and the marbles, each sitting on the sheet (its center one radius above the sheet's height there).
  const planets = CONFIG.planets.map(cfg => ({ cfg, x: 0, y: 0, vx: 0, vy: 0, ax: 0, ay: 0, held: false, trail: null, n: 0, head: 0, screen: null, age: 0, well: -1 }));
  // How visible a marble is: 1, except near the start and end of a cycle (CONFIG.respawnFade).
  const fade = p => { const c = p.cfg.cycle, f = CONFIG.respawnFade; return c > 0 && f > 0 && !p.held ? Math.max(0, Math.min(1, p.age / f, (c - p.age) / f)) : 1; };
  const trailCap = Math.max(2, Math.round(CONFIG.trail.seconds * CONFIG.trail.samplesPerSecond));
  for (const p of planets) p.trail = new Float32Array(2 * trailCap);
  const balls = [...(CONFIG.drawMasses ? MASSES : []).map(m => ({ body: m, size: m.size, color: m.color })), ...planets.map(p => ({ body: p, size: p.cfg.size, color: p.cfg.color }))];
  const flag = new Uint8Array(nc);
  const rgba = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  function drawBall(b) {
    const alpha = b.body.cfg ? fade(b.body) : 1;
    if (alpha <= 0) return;
    ctx.globalAlpha = alpha;
    ctx.beginPath(); ctx.arc(b.sx, b.sy, b.r, 0, TAU);
    ctx.fillStyle = rgba(b.color); ctx.fill();
    ctx.lineWidth = CONFIG.ballOutline * b.r; ctx.strokeStyle = look.outline; ctx.stroke();
    ctx.globalAlpha = 1;
  }
  function drawTrails() {
    const steps = CONFIG.trail.fadeSteps;
    if (!(CONFIG.trail.alpha > 0)) return;
    ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.lineWidth = look.trailWidth * fit.dpr;
    for (const p of planets) {
      if (p.n < 2) continue;
      // Oldest first; the trail is cut into `steps` runs, more opaque toward the marble.
      const pts = new Float32Array(2 * p.n);
      for (let i = 0; i < p.n; i++) {
        const k = (p.head - p.n + i + trailCap) % trailCap, x = p.trail[2 * k], y = p.trail[2 * k + 1];
        project(x, y, height(x, y)); pts[2 * i] = P[0]; pts[2 * i + 1] = P[1];
      }
      for (let s = 0; s < steps; s++) {
        const a = Math.floor(s * (p.n - 1) / steps), b = Math.floor((s + 1) * (p.n - 1) / steps);
        if (b <= a) continue;
        ctx.beginPath(); ctx.moveTo(pts[2 * a], pts[2 * a + 1]);
        for (let i = a + 1; i <= b; i++) ctx.lineTo(pts[2 * i], pts[2 * i + 1]);
        ctx.strokeStyle = rgba(p.cfg.color, CONFIG.trail.alpha * fade(p) * (s + 1) / steps); ctx.stroke();
      }
    }
  }
  function draw() {
    if (canvas.width <= 1) return;
    const t0 = performance.now();
    if (stale) paintSheet();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(cache, 0, 0);
    drawTrails();
    // Each ball on screen; key: its center's distance minus the occlusion margin (cells nearer than that may cover it).
    for (const b of balls) {
      const base = cssPx(b.size) * fit.dpr / 2, x = b.body.x, y = b.body.y, lift = base / fit.scale;
      project(x, y, height(x, y) + lift);
      b.sx = P[0]; b.sy = P[1]; b.r = base * P[3]; b.key = P[2] - CONFIG.occlusionMargin * lift;
      if (b.body.cfg) b.body.screen = [b.sx, b.sy, b.r];
    }
    balls.sort((a, b) => b.key - a.key);   // farthest first
    flag.fill(0);
    let flagged = 0;
    for (const b of balls) for (let c = 0; c < nc && !(CONFIG.massesOnTop && !b.body.cfg); c++) {
      if (cellDist[c] >= b.key || cellBox[4 * c] > b.sx + b.r || cellBox[4 * c + 2] < b.sx - b.r || cellBox[4 * c + 1] > b.sy + b.r || cellBox[4 * c + 3] < b.sy - b.r) continue;
      if (!flag[c]) { flag[c] = 1; flagged++; }
    }
    // Balls and the cells in front of them, back to front.
    let bi = 0;
    const grow = CONFIG.seam * fit.dpr;
    if (flagged) for (let i = 0; i < nc; i++) {
      const c = order[i];
      if (!flag[c]) continue;
      while (bi < balls.length && balls[bi].key > cellDist[c]) drawBall(balls[bi++]);
      ctx.beginPath(); cellPath(ctx, c, grow); ctx.fillStyle = palette[cellBin[c]]; ctx.fill();
      strokeSegs(ctx, 0, 1, c);
    }
    while (bi < balls.length) drawBall(balls[bi++]);
    overlayCells = flagged;
    frameMs = performance.now() - t0;
  }
  let overlayCells = 0;

  // Physics.
  const A = [0, 0];
  function place(p) {   // a marble at its starting point and speed (CONFIG.planets), its cycle and trail restarted
    const m = MASSES[p.cfg.around] || MASSES[0], a = p.cfg.angle * deg, ux = Math.cos(a), uy = Math.sin(a);
    p.x = m.x + p.cfg.distance * ux; p.y = m.y + p.cfg.distance * uy;
    accel(p.x, p.y, A);
    const inward = Math.max(0, -(A[0] * ux + A[1] * uy)), v = p.cfg.speed * Math.sqrt(inward * p.cfg.distance), dir = p.cfg.direction < 0 ? -1 : 1;
    p.vx = -uy * v * dir; p.vy = ux * v * dir; p.ax = A[0]; p.ay = A[1]; p.held = false; p.n = 0; p.head = 0; p.age = 0;
    p.well = wellOf(p.x, p.y);
  }
  // Which well a point is in: the heavy ball whose own dent is deepest there. Changes are logged (for gravityState()).
  function wellOf(x, y) {
    let best = 0, deepest = -1;
    MASSES.forEach((m, i) => { const dx = x - m.x, dy = y - m.y, d = m.depth / Math.sqrt(1 + (dx * dx + dy * dy) / (m.width * m.width)); if (d > deepest) { deepest = d; best = i; } });
    return best;
  }
  const wellLog = [];
  let simTime = 0, pending = 0, sinceSample = 0;
  const sampleEvery = 1 / CONFIG.trail.samplesPerSecond;
  function substep(p, h) {
    p.x += p.vx * h + 0.5 * p.ax * h * h; p.y += p.vy * h + 0.5 * p.ay * h * h;
    accel(p.x, p.y, A);
    p.vx += 0.5 * (p.ax + A[0]) * h; p.vy += 0.5 * (p.ay + A[1]) * h; p.ax = A[0]; p.ay = A[1];
    const { rim, rimBounce, maxSpeed } = CONFIG.physics, r = Math.hypot(p.x, p.y);
    if (r > rim) {
      const nx = p.x / r, ny = p.y / r, vn = p.vx * nx + p.vy * ny;
      p.x = nx * rim; p.y = ny * rim;
      if (vn > 0) { p.vx -= (1 + rimBounce) * vn * nx; p.vy -= (1 + rimBounce) * vn * ny; }
      accel(p.x, p.y, A); p.ax = A[0]; p.ay = A[1];
    }
    const v = Math.hypot(p.vx, p.vy);
    if (v > maxSpeed) { p.vx *= maxSpeed / v; p.vy *= maxSpeed / v; }
  }
  function sample() {
    for (const p of planets) { p.trail[2 * p.head] = p.x; p.trail[2 * p.head + 1] = p.y; p.head = (p.head + 1) % trailCap; p.n = Math.min(trailCap, p.n + 1); }
  }
  function simulate(dt) {
    const t0 = performance.now(), h = CONFIG.physics.substep;
    pending += dt * CONFIG.physics.timeScale;
    while (pending >= h) {
      pending -= h;
      simTime += h;
      planets.forEach((p, i) => {
        if (!p.held) substep(p, h);
        p.age += h;
        if (p.cfg.cycle > 0 && p.age >= p.cfg.cycle && !p.held) { place(p); wellLog.push({ t: +simTime.toFixed(2), planet: i, respawn: true, well: p.well }); }
        const w = wellOf(p.x, p.y);
        if (w !== p.well) { wellLog.push({ t: +simTime.toFixed(2), planet: i, from: p.well, to: w }); p.well = w; }
      });
      if (wellLog.length > 100) wellLog.splice(0, wellLog.length - 100);
      sinceSample += h;
      if (sinceSample >= sampleEvery) { sinceSample -= sampleEvery; sample(); }
    }
    physicsMs = performance.now() - t0;
  }
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function reset() {
    planets.forEach(place); pending = 0; sinceSample = 0; simTime = 0; wellLog.length = 0;
    if (!reduced) { sample(); simulate(CONFIG.trail.seconds / CONFIG.physics.timeScale); }   // run the trail's length ahead, so the trails are there from the start
  }

  // Redraws (batched into one animation frame, never while the slide is hidden) and the motion loop.
  let queued = false, sized = false;
  function render() { queued = false; if (slide.hidden) return; if (!sized) { layout(); sized = true; } draw(); }
  function schedule() { if (!queued) { queued = true; requestAnimationFrame(render); } }
  new ResizeObserver(() => { sized = false; schedule(); }).observe(canvas);
  let running = false, lastTick = 0, wokeAt = performance.now();
  function tick(now) {
    if (!running) return;
    if (slide.hidden || document.hidden || now - wokeAt > CONFIG.idleStop) { running = false; return; }
    const dt = Math.min(0.05, (now - lastTick) / 1000);
    if (dt >= 1 / CONFIG.maxFps - 0.002) { simulate(dt); lastTick = now; render(); }
    requestAnimationFrame(tick);
  }
  function start() {
    if (running || reduced || slide.hidden || document.hidden) return;
    running = true; lastTick = performance.now(); requestAnimationFrame(tick);
  }
  const wake = () => { wokeAt = performance.now(); start(); };

  // Pointer: pressing on (or near) a marble carries it over the sheet and flings it on release; anywhere else turns the view.
  let drag = null;
  const local = e => { const r = canvas.getBoundingClientRect(), k = canvas.width / (r.width || 1); return [(e.clientX - r.left) * k, (e.clientY - r.top) * k]; };
  const marbleAt = (px, py) => {
    let best = null, bestD = Infinity;
    for (const p of planets) {
      if (!p.screen) continue;
      const d = Math.hypot(px - p.screen[0], py - p.screen[1]);
      if (d <= p.screen[2] + CONFIG.fling.grabSlop * fit.dpr && d < bestD) { best = p; bestD = d; }
    }
    return best;
  };
  function carry(e) {
    const q = sheetAt(...local(e));
    if (!q) return;
    let { x, y } = { x: q[0], y: q[1] };
    const r = Math.hypot(x, y), rim = CONFIG.physics.rim;
    if (r > rim) { x *= rim / r; y *= rim / r; }
    const p = drag.planet;
    p.x = x; p.y = y; p.vx = 0; p.vy = 0;
    const now = performance.now();
    drag.samples.push([now, x, y]);
    while (drag.samples.length > 2 && now - drag.samples[1][0] > CONFIG.fling.window) drag.samples.shift();
    schedule();
  }
  canvas.addEventListener('pointerdown', e => {
    e.preventDefault(); canvas.setPointerCapture(e.pointerId); wake();
    const planet = marbleAt(...local(e));
    if (planet) { drag = { planet, samples: [] }; planet.held = true; carry(e); }
    else drag = { x: e.clientX, y: e.clientY };
    canvas.style.cursor = 'grabbing';
  });
  canvas.addEventListener('pointermove', e => {
    if (!drag) { canvas.style.cursor = marbleAt(...local(e)) ? 'pointer' : ''; return; }
    wokeAt = performance.now();
    if (drag.planet) { carry(e); return; }
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    view.yaw += dx * CONFIG.dragSpeed * deg;
    view.elevation = Math.max(CONFIG.view.minElevation * deg, Math.min(CONFIG.view.maxElevation * deg, view.elevation + dy * CONFIG.dragSpeed * deg));
    T = trig(view); stale = true; schedule();
  });
  const end = e => {
    if (drag?.planet) {
      const p = drag.planet, s = drag.samples, now = performance.now();
      if (s.length >= 2 && now - s[s.length - 1][0] < CONFIG.fling.window) {
        const a = s[0], b = s[s.length - 1], dt = (b[0] - a[0]) / 1000;
        if (dt > 0) { p.vx = (b[1] - a[1]) / dt; p.vy = (b[2] - a[2]) / dt; }
        const v = Math.hypot(p.vx, p.vy), max = CONFIG.physics.maxSpeed;
        if (v > max) { p.vx *= max / v; p.vy *= max / v; }
      }
      accel(p.x, p.y, A); p.ax = A[0]; p.ay = A[1]; p.held = false; p.age = CONFIG.respawnFade;   // its cycle restarts, fully visible
      wake();
    }
    drag = null; canvas.style.cursor = '';
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
  };
  canvas.addEventListener('pointerup', end); canvas.addEventListener('pointercancel', end);

  function open() { sized = false; if (CONFIG.resetOnShow || !opened) reset(); opened = true; schedule(); wake(); }
  let opened = false;
  new MutationObserver(() => { if (!slide.hidden) open(); }).observe(slide, { attributes: true, attributeFilter: ['hidden'] });
  document.addEventListener('visibilitychange', () => { if (!document.hidden && !slide.hidden) wake(); });

  window.gravityState = () => ({
    config: CONFIG, running, reduced, view: { yaw: view.yaw / deg, elevation: view.elevation / deg }, fit: { ...fit },
    planets: planets.map(p => ({ x: p.x, y: p.y, vx: p.vx, vy: p.vy, held: p.held, well: p.well, age: p.age, energy: 0.5 * (p.vx * p.vx + p.vy * p.vy) + CONFIG.physics.gravity * height(p.x, p.y), screen: p.screen })),
    simTime, wellLog: wellLog.slice(), sheetMs, frameMs, physicsMs, overlayCells, cells: nc, gridPieces: nseg, heightRange: [hLo, hHi]
  });   // for console inspection (degrees; energy per unit mass, conserved between bounces; well: index into CONFIG.masses; age: simulated seconds since the marble (re)started; wellLog: well changes and respawns, t in simulated seconds since the slide was shown; sheetMs: last full repaint of the sheet, frameMs: last frame's drawing, physicsMs: last frame's simulation)
  if (!slide.hidden) open();
})();
