// "Language basins" slide. The paper's language expansion maps (Figure "Activation space expansion along en-zh and en-el tangent
// directions") as globes: for each of six Llama 3.1 8B activations (two each from English, Greek and Chinese Wikipedia documents), the
// 2-D sphere of radius |x| through the activation x, spanned by x and the en-zh and en-el language directions, colored by how much
// blocks 16-19 stretch its surface there (the paper's intrinsic area ratio, log scale, magma: dark = little expansion, the basins).
// One panel per language with its two globes (assets/lang-globes.js; provenance there). All globes share one rotation, and because
// every sphere uses the same kind of frame (activation, en-zh, en-el), the same rotation shows the same directions on each globe:
// at the start, the paper's view (the activation, marked +, faces the viewer; toward Chinese is right, toward Greek is up).
// Drag any globe to rotate them all. On their own they wobble gently about the paper's view, the activation circling each globe's center
// on the near side; after a drag they ease back onto the circle. They sleep when the slide or tab is hidden or the slide is left idle.
// Each globe is ray-cast per pixel (orthographic) from the stored longitude-latitude grid, bilinearly interpolated.
// Edit CONFIG to tweak.
(() => {
  const IDS = { slide: '#langglobes-slide', canvases: '#langglobes-slide .langglobes-canvas' };   // one canvas per language, data-language="en"
  const CONFIG = {
    labels: { en: 'English', el: 'Greek', zh: 'Chinese' },   // panel titles (drawn above each panel's globes)
    colorRange: 'paper',      // each globe's colors span its own range of log10 expansion (as in the paper, per panel): 'paper': the min..max
                              // inside the paper's displayed window (longitude and latitude within ±45° of the activation; the far side
                              // clips to the ends), 'sphere': the min..max over the whole sphere
    // Colormap: matplotlib's magma, the paper's, sampled at 33 even stops (low → high expansion), interpolated linearly.
    colormap: '0000040303120a0822130d341d114729115a36106b440f7651127c5d177f6a1c81762181832681902a819c2e7faa337db73779c43c75d0416fdc4869e75263ef5d5ef56b5cf9795dfc8961fd9869fea772feb67cfec488fed395fde2a3fcf0b2fcfdbf',
    view: { yaw: 0, pitch: 0 },   // the home view (radians away from the paper's view): where the globes start and what they sway about
    globeGap: 0.18,           // space between a panel's two globes, as a fraction of the globe diameter
    labelGap: 0.5,            // space between a panel's title and its globes, in title font sizes
    outlineColor: 'rgba(32,42,54,0.25)', outlineWidth: 1,   // each globe's rim
    markColor: '#ffffff',     // the activation (+, white as in the paper's figure; size is the deck's --mark-small), when on the near side
    dragSpeed: 0.01,          // radians per CSS pixel dragged
    // Auto-motion: the globes wobble so the activation circles the globes' centers at a constant angle from the viewer: at time t the
    // home view is tilted by `angle` toward screen direction 90° + w t (starting at the top, then counterclockwise), i.e. yaw ≈ -angle
    // sin(w t), pitch ≈ angle cos(w t), but exact, so the activation stays on the near side. Dragging pauses it; spinResumeDelay ms after a drag the globes
    // ease from wherever they were dragged (even with the activation at the back) onto the circle over easeTime ms.
    sway: { angle: 18, period: 18 },   // the activation's angle from the viewer (degrees, keep well below 90) and one lap (seconds); angle 0: off
    easeTime: 2000,
    spinResumeDelay: 5000, spinFps: 30,
    spinIdleStop: 90000,      // the motion stops (sleeps) after this many ms without a drag; showing the slide again or dragging wakes it
  };
  const DATA = window.LANG_GLOBES;
  const slide = document.querySelector(IDS.slide), canvases = [...document.querySelectorAll(IDS.canvases)];
  if (!DATA || !slide || !canvases.length) return;

  // Grid: rows latitude -90..90, columns longitude -180..180 (the last column repeats the first).
  const NLAT = DATA.grid.lat.count, NLON = DATA.grid.lon.count;
  const decode = s => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new Uint16Array(u.buffer); };
  // Per map: the color coordinate t in [0, 1] at every grid point (log10 area mapped through the chosen range, clipped).
  const maps = DATA.maps.map(m => {
    const q = decode(m.values), lo = CONFIG.colorRange === 'paper' ? m.paperLog10Min : m.log10Min, hi = CONFIG.colorRange === 'paper' ? m.paperLog10Max : m.log10Max;
    const t = new Float32Array(q.length), scale = (m.log10Max - m.log10Min) / 65535;
    for (let i = 0; i < q.length; i++) t[i] = Math.min(1, Math.max(0, (m.log10Min + q[i] * scale - lo) / (hi - lo)));
    return { ...m, t };
  });
  // 256-entry lookup table of packed RGBA (little-endian Uint32: 0xAABBGGRR).
  const stops = CONFIG.colormap.match(/.{6}/g).map(h => [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)));
  const LUT = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    const s = i / 255 * (stops.length - 1), k = Math.min(stops.length - 2, Math.floor(s)), f = s - k;
    const [r, g, b] = [0, 1, 2].map(c => Math.round(stops[k][c] + f * (stops[k + 1][c] - stops[k][c])));
    LUT[i] = (255 << 24 | b << 16 | g << 8 | r) >>> 0;
  }
  const panels = canvases.map(canvas => ({
    canvas, ctx: canvas.getContext('2d'), label: CONFIG.labels[canvas.dataset.language] ?? canvas.dataset.language,
    globes: maps.filter(m => m.language === canvas.dataset.language), image: null, centers: [],
  }));

  // View: frame coordinates (pole = activation, e1 = en-zh, e2 = zh-el; see the asset) -> screen (x right, y up, z toward the viewer).
  // The paper reflects both chart axes (en->zh right, en->el up), so the base view puts -e1 right, -e2 up and the pole toward the viewer.
  const rotX = a => [1, 0, 0, 0, Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a)];
  const rotY = a => [Math.cos(a), 0, Math.sin(a), 0, 1, 0, -Math.sin(a), 0, Math.cos(a)];
  const mul = (A, B) => { const M = []; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) M.push(A[3*i]*B[j] + A[3*i+1]*B[3+j] + A[3*i+2]*B[6+j]); return M; };
  const BASE = [0, -1, 0, 0, 0, -1, 1, 0, 0];
  const HOME = mul(mul(rotX(CONFIG.view.pitch), rotY(CONFIG.view.yaw)), BASE);
  let view = HOME.slice();
  // Rotation by angle a about unit axis u (Rodrigues), and the rotation Q = B A^T taking view A to view B as an axis and angle.
  const rotAxis = ([x, y, z], a) => { const c = Math.cos(a), s = Math.sin(a), t = 1 - c; return [t*x*x + c, t*x*y - s*z, t*x*z + s*y, t*x*y + s*z, t*y*y + c, t*y*z - s*x, t*x*z - s*y, t*y*z + s*x, t*z*z + c]; };
  const transpose = A => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
  function axisAngle(Q) {
    const angle = Math.acos(Math.max(-1, Math.min(1, (Q[0] + Q[4] + Q[8] - 1) / 2)));
    let u = [Q[7] - Q[5], Q[2] - Q[6], Q[3] - Q[1]], n = Math.hypot(...u);
    if (n < 1e-6) {   // angle 0 or pi: the axis is Q's +1 eigenvector, the largest column of Q + I
      const P = Q.map((q, i) => q + (i % 4 === 0 ? 1 : 0)), cols = [0, 1, 2].map(j => [P[j], P[3 + j], P[6 + j]]);
      u = cols.reduce((a, b) => Math.hypot(...b) > Math.hypot(...a) ? b : a); n = Math.hypot(...u) || 1; if (Math.hypot(...u) === 0) u = [0, 1, 0];
    }
    return { axis: u.map(v => v / n), angle };
  }
  // Circle view at time t: tilt HOME's activation (screen +z) by the sway angle toward screen direction phi, rotating about the in-plane
  // axis perpendicular to phi (no twist about the view direction).
  const deg = Math.PI / 180, swayAngle = CONFIG.sway.angle * deg;
  const swayView = t => { const phi = Math.PI / 2 + 2 * Math.PI * t / CONFIG.sway.period; return mul(rotAxis([-Math.sin(phi), Math.cos(phi), 0], swayAngle), HOME); };

  // Disk of one globe (radius R device pixels, tile D x D): unit-sphere screen coordinates of the pixels inside it.
  let R = 0, D = 0, inside = null, sx = null, sy = null, sz = null, cell = null, fu = null, fv = null;
  function buildDisk() {
    D = 2 * Math.ceil(R); const c = D / 2, idx = [], X = [], Y = [];
    for (let j = 0; j < D; j++) for (let i = 0; i < D; i++) {
      const x = (i + 0.5 - c) / R, y = -(j + 0.5 - c) / R;
      if (x * x + y * y <= 1) { idx.push(j * D + i); X.push(x); Y.push(y); }
    }
    inside = Int32Array.from(idx); sx = Float32Array.from(X); sy = Float32Array.from(Y);
    sz = sx.map((x, k) => Math.sqrt(Math.max(0, 1 - x * x - sy[k] * sy[k])));
    cell = new Int32Array(inside.length); fu = new Float32Array(inside.length); fv = new Float32Array(inside.length);
  }
  // Shared rotation: each pixel's grid cell and bilinear weights, computed once per frame for all six globes.
  function locate() {
    const M = view, lonScale = (NLON - 1) / (2 * Math.PI), latScale = (NLAT - 1) / Math.PI;
    for (let k = 0; k < inside.length; k++) {
      const x = sx[k], y = sy[k], z = sz[k];
      const c0 = M[0] * x + M[3] * y + M[6] * z, c1 = M[1] * x + M[4] * y + M[7] * z, c2 = M[2] * x + M[5] * y + M[8] * z;   // M^T s
      const u = (Math.atan2(c1, c0) + Math.PI) * lonScale, v = (Math.asin(Math.max(-1, Math.min(1, c2))) + Math.PI / 2) * latScale;
      const i0 = Math.min(NLON - 2, Math.floor(u)), j0 = Math.min(NLAT - 2, Math.floor(v));
      cell[k] = j0 * NLON + i0; fu[k] = u - i0; fv[k] = v - j0;
    }
  }
  function paint(px, t, x0, y0, W) {
    for (let k = 0; k < inside.length; k++) {
      const a = cell[k], u = fu[k], v = fv[k];
      const value = (1 - v) * ((1 - u) * t[a] + u * t[a + 1]) + v * ((1 - u) * t[a + NLON] + u * t[a + NLON + 1]);
      const p = inside[k], i = p % D, j = (p - i) / D;
      px[(y0 + j) * W + x0 + i] = LUT[Math.round(value * 255)];
    }
  }

  const css = c => getComputedStyle(c);
  // Layout: each panel's title, then its two globes side by side or stacked (whichever is bigger); one radius for all panels.
  function layout() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let best = Infinity;
    for (const p of panels) {
      const r = p.canvas.getBoundingClientRect(); p.canvas.width = Math.max(1, Math.round(r.width * dpr)); p.canvas.height = Math.max(1, Math.round(r.height * dpr));
      p.font = parseFloat(css(p.canvas).fontSize) * dpr; p.top = p.font * (1.25 + CONFIG.labelGap);
      const W = p.canvas.width, H = p.canvas.height - p.top, n = p.globes.length, across = 2 * n + 2 * CONFIG.globeGap * (n - 1);   // in radii
      const side = Math.min(W / across, H / 2), stack = Math.min(W / 2, H / across);
      p.side = side >= stack; best = Math.min(best, Math.max(side, stack));
    }
    R = Math.max(2, Math.floor(best) - 1); buildDisk();
    for (const p of panels) {
      const W = p.canvas.width, H = p.canvas.height, n = p.globes.length, gap = Math.round(CONFIG.globeGap * D);
      const bw = p.side ? n * D + (n - 1) * gap : D, bh = p.side ? D : n * D + (n - 1) * gap;
      const x0 = Math.round((W - bw) / 2), y0 = Math.round(p.top + (H - p.top - bh) / 2);
      p.labelY = y0 - CONFIG.labelGap * p.font;
      p.centers = p.globes.map((_, i) => [x0 + (p.side ? i * (D + gap) : 0), y0 + (p.side ? 0 : i * (D + gap))]);
      p.image = p.ctx.createImageData(W, H);
    }
  }
  function draw() {
    if (!inside) return;
    locate();
    const M = view;
    for (const p of panels) {
      const { ctx, canvas, image } = p, W = canvas.width, px = new Uint32Array(image.data.buffer);
      p.globes.forEach((g, i) => paint(px, g.t, p.centers[i][0], p.centers[i][1], W));
      ctx.putImageData(image, 0, 0);
      const style = css(canvas), dpr = W / canvas.getBoundingClientRect().width, mark = (parseFloat(style.getPropertyValue('--mark-small')) || 12) * dpr * 0.6;
      ctx.fillStyle = style.color; ctx.font = `${p.font}px ${style.fontFamily}`; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      ctx.fillText(p.label, W / 2, p.labelY);
      for (const [x0, y0] of p.centers) {
        const cx = x0 + D / 2, cy = y0 + D / 2;
        ctx.strokeStyle = CONFIG.outlineColor; ctx.lineWidth = CONFIG.outlineWidth * dpr;
        ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke();
        if (M[6] > 0) {   // the activation (the pole) on the near side
          const x = cx + M[0] * R, y = cy - M[3] * R;
          ctx.strokeStyle = CONFIG.markColor; ctx.lineWidth = 2 * dpr; ctx.beginPath();
          ctx.moveTo(x - mark, y); ctx.lineTo(x + mark, y); ctx.moveTo(x, y - mark); ctx.lineTo(x, y + mark); ctx.stroke();
        }
      }
    }
  }

  let queued = false, sized = false;
  function render() { queued = false; if (slide.hidden) return; if (!sized) { layout(); sized = true; } draw(); }
  function schedule() { if (!queued) { queued = true; requestAnimationFrame(render); } }
  const resize = () => { sized = false; schedule(); };
  const observer = new ResizeObserver(resize); canvases.forEach(c => observer.observe(c));

  // Circling and drag; the motion sleeps when the slide or tab is hidden and after spinIdleStop ms without a drag. Opening the slide puts
  // the globes on the circle at its top (the activation above the center) and goes on from there; any other start (tab back, drag
  // released) restarts the circle at the top and eases onto it from the current view along the shortest rotation.
  const canSpin = CONFIG.sway.angle > 0 && CONFIG.sway.period > 0 && !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let spinPaused = false, spinning = false, lastSpin = 0, resumeTimer = 0, drag = null, wokeAt = performance.now(), swayStart = 0, easeFrom = null;
  const smooth = e => e * e * (3 - 2 * e);
  function spin(now) {
    if (!canSpin || spinPaused || slide.hidden || document.hidden || now - wokeAt > CONFIG.spinIdleStop) { spinning = false; return; }
    if (now - lastSpin >= 1000 / CONFIG.spinFps - 2) {
      const target = swayView((now - swayStart) / 1000), e = easeFrom ? (now - swayStart) / CONFIG.easeTime : 1;
      if (e >= 1) { view = target; easeFrom = null; }
      else { const { axis, angle } = axisAngle(mul(target, transpose(easeFrom))); view = mul(rotAxis(axis, smooth(e) * angle), easeFrom); }
      lastSpin = now; schedule();
    }
    requestAnimationFrame(spin);
  }
  function startSpin() {
    if (!canSpin || spinPaused || spinning || slide.hidden || document.hidden) return;
    spinning = true; lastSpin = swayStart = performance.now(); easeFrom = view.slice(); requestAnimationFrame(spin);
  }
  const wake = () => { wokeAt = performance.now(); startSpin(); };
  function open() {
    if (canSpin) { view = swayView(0); easeFrom = null; lastSpin = swayStart = performance.now(); }
    resize(); wokeAt = performance.now();
    if (canSpin && !spinPaused && !spinning) { spinning = true; requestAnimationFrame(spin); }
  }
  new MutationObserver(() => { if (!slide.hidden) open(); }).observe(slide, { attributes: true, attributeFilter: ['hidden'] });
  document.addEventListener('visibilitychange', startSpin);
  for (const c of canvases) {
    c.addEventListener('pointerdown', e => { spinPaused = true; clearTimeout(resumeTimer); e.preventDefault(); c.setPointerCapture(e.pointerId); c.style.cursor = 'grabbing'; drag = { x: e.clientX, y: e.clientY }; });
    c.addEventListener('pointermove', e => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag = { x: e.clientX, y: e.clientY };
      view = mul(mul(rotX(dy * CONFIG.dragSpeed), rotY(dx * CONFIG.dragSpeed)), view); schedule();
    });
    const endDrag = e => {
      if (drag) resumeTimer = setTimeout(() => { spinPaused = false; wake(); }, CONFIG.spinResumeDelay);
      drag = null; c.style.cursor = ''; if (c.hasPointerCapture(e.pointerId)) c.releasePointerCapture(e.pointerId);
    };
    c.addEventListener('pointerup', endDrag); c.addEventListener('pointercancel', endDrag);
  }

  window.langGlobesState = () => ({ config: CONFIG, view: view.slice(), activationAngle: Math.acos(Math.max(-1, Math.min(1, view[6]))) / deg, activationDirection: Math.atan2(view[3], view[0]) / deg, radius: R, spinning, easing: !!easeFrom, maps: maps.map(m => `${m.language}: ${m.title}`) });   // for console inspection (degrees; direction: of the activation on screen, from the right, counterclockwise)
  if (!slide.hidden) open();
})();
