// "Steering between basins" slide. A Llama 3 8B Instruct response to a benign request, steered along the refusal direction d
// with strength alpha at the first output step only (the paper's one-position intervention). One speed profile per output
// step, stacked vertically with a shared horizontal axis (the coefficient along d). The slider picks alpha among the
// precomputed strengths (assets/basins-midpoint.js, computed on carc with the paper's code); each strength has its own greedy
// response, so the rows (tokens and curves) change with it. Probe: for step t, the line m_t + b d through the midpoint m_t of
// step t's activations in the unsteered and the 3.37-steered responses (0 on the axis is m_t); the current activation is
// projected onto it (the dot). Each curve is || d/db f(m_t + b d) ||, f = blocks 13-16 under the step's full prefix.
// Edit CONFIG to tweak.
(() => {
  const IDS = { slide: '#basins-midpoint-slide', canvas: '#basins-midpoint-plot', slider: '#basins-midpoint-strength', play: '#basins-midpoint-play' };
  const CONFIG = {
    startAlpha: 3.37,         // strength shown when the slide opens (snaps to the nearest precomputed one; the paper uses 3.37)
    maxStrength: 3.8,         // the slider stops at this strength (precomputed strengths above it are left out)
    alphaDecimals: 2,         // digits of the strength shown at the slider's end
    steps: 8,                 // output steps shown (rows), from the first; at most the precomputed count
    alphaRange: [-4, 4],      // horizontal range of each plot, in units of d (the precomputed grid covers [-4, 4])
    sharedScale: false,       // false: each row is scaled to its own speed range; true: all rows share one speed scale
    rowGap: 0.2,              // empty space between rows, as a fraction of a row's height
    curveColor: '#202a36', curveWidth: 2,    // the speed profile (width in CSS pixels)
    zeroColor: 'rgba(32,42,54,0.18)', zeroWidth: 1,   // vertical line at 0 along d (the step's midpoint m_t)
    markColor: [42, 111, 219],              // dot at the step's actual activation (its position on the probe line); size is the deck's --mark-small
    showMinima: false, minimaColor: 'rgba(32,42,54,0.55)',   // dots at the speed minima (basin centres); size --mark-tiny
    tokenLabels: true,        // label each row with its token, as the paper's refusal figure labels each panel
    labelGap: 0.6,            // between a row's label and its plot, in label ems
    yLabel: 'Density \u2193',  // y-axis label, upright, above the stacked plots at their left edge ('' hides it); the arrow: density grows downward (\u2193 = ↓)
    xLabel: '\u2190 Comply\u2001Refuse \u2192',   // under the shared horizontal axis, centered on the plots ('' hides it); canvas text: use \u escapes, not HTML entities (\u2001 is an em quad)
    axisLabelGap: 0.5,        // between an axis label and the plots, in label ems
    highlightRows: [3],       // rows (1 = first step) drawn on a highlighted band, label and plot; [] for none
    highlightColor: 'rgba(250, 204, 21, 0.25)',   // the band's color (soft yellow, like the prompt slide's token highlight)
    // The same probe along the step's own direction from its unsteered to its 3.37-steered activation (scaled to the refusal
    // direction's length), drawn lighter under the refusal-direction curve.
    showBetween: false,       // off: the refusal-direction curve is the one shown; true draws the between-activations curve too
    betweenColor: [32, 42, 54], betweenAlpha: 0.3,   // curve, and its dot (size --mark-tiny), at this opacity
    // The play button at the start of the slider row sweeps the slider up and down (js/slider-autoplay.js, a sine: one full cycle
    // low → high → low takes period seconds) until pressed again. Adjusting the slider or leaving the slide stops it. Nothing plays on its own.
    autoplay: { period: 14 }
  };
  const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  const tokenLabel = t => t.replace(/\n/g, '\\n').trim() || t;   // "\n\n" is written out, leading spaces dropped (as in the paper)

  setup();

  function setup() {
    const DATA = window.BASINS_MIDPOINT;
    const slide = document.querySelector(IDS.slide), canvas = document.querySelector(IDS.canvas), slider = document.querySelector(IDS.slider);
    if (!DATA || !slide || !canvas || !slider) return;
    const ctx = canvas.getContext('2d');
    const strengths = DATA.strengths.filter(s => s.alpha <= CONFIG.maxStrength + 1e-9);   // sorted by alpha; alpha 0 is the unsteered response
    let current = strengths.reduce((best, s, i) => Math.abs(s.alpha - CONFIG.startAlpha) < Math.abs(strengths[best].alpha - CONFIG.startAlpha) ? i : best, 0);
    let plotExtent = null;   // the plots' left and right edge in canvas pixels, from the last draw (for console inspection)

    function draw() {
      const all = strengths[current].steps, steps = all.slice(0, Math.min(CONFIG.steps, all.length));
      const w = canvas.width, h = canvas.height, k = w / (canvas.clientWidth || w);   // canvas pixels per CSS pixel
      const style = getComputedStyle(canvas), font = parseFloat(style.fontSize) * k;
      const markR = parseFloat(style.getPropertyValue('--mark-small')) / 2 * k, tinyR = parseFloat(style.getPropertyValue('--mark-tiny')) / 2 * k;
      ctx.clearRect(0, 0, w, h);
      ctx.font = `${font}px ${style.fontFamily}`;
      // Horizontal layout: the token labels at the left edge, then the plots to the right edge (minus room for the dots).
      // The token column fits the longest token of every strength's shown steps, so the plots don't move as the slider changes the tokens.
      const labels = steps.map(s => tokenLabel(s.token));
      const labelW = Math.max(...strengths.flatMap(st => st.steps.slice(0, CONFIG.steps)).map(s => ctx.measureText(tokenLabel(s.token)).width));
      const yLabelW = 0;   // the y label sits above the plots (upright), so it takes no column
      const gutter = yLabelW + (CONFIG.tokenLabels ? labelW + CONFIG.labelGap * font : 0);
      const x0 = gutter + markR, x1 = w - markR, [a0, a1] = CONFIG.alphaRange;
      const X = a => x0 + (a - a0) / (a1 - a0) * (x1 - x0);
      plotExtent = [x0, x1];
      // Vertical layout: the y label on top, then one row per step of the configured count (so rows keep their size when a response stops
      // early), the x label below.
      const xLabelH = CONFIG.xLabel ? font * (1.2 + CONFIG.axisLabelGap) : 0, yLabelH = CONFIG.yLabel ? font * (1.2 + CONFIG.axisLabelGap) : 0;
      const top = markR + yLabelH, rowH = (h - 2 * markR - xLabelH - yLabelH) / CONFIG.steps, gap = rowH * CONFIG.rowGap;
      ctx.fillStyle = CONFIG.curveColor;
      if (CONFIG.xLabel) { ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(CONFIG.xLabel, (x0 + x1) / 2, h); }
      if (CONFIG.yLabel) { ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(CONFIG.yLabel, x0 - markR, 0); }
      // The grid's values along d and, per row, the visible speeds and their range.
      const { min, max, count } = DATA.alpha, alphaAt = i => min + (max - min) * i / (count - 1);
      const visible = []; for (let i = 0; i < count; i++) { const a = alphaAt(i); if (a >= a0 - 1e-9 && a <= a1 + 1e-9) visible.push(i); }
      // Each row's range covers both series, so they share its speed scale.
      const series = s => s.between ? (CONFIG.showBetween ? [s.speed, s.between.speed] : [s.speed]) : [s.speed];
      const ranges = steps.map(s => { let lo = Infinity, hi = -Infinity; series(s).forEach(sp => visible.forEach(i => { lo = Math.min(lo, sp[i]); hi = Math.max(hi, sp[i]); })); return [lo, hi]; });
      const shared = [Math.min(...ranges.map(r => r[0])), Math.max(...ranges.map(r => r[1]))];
      const speedAt = (sp, a) => { const t = (a - min) / (max - min) * (count - 1), i = Math.min(count - 2, Math.max(0, Math.floor(t))), f = t - i; return sp[i] * (1 - f) + sp[i + 1] * f; };
      // Line at 0 along d through all rows.
      ctx.strokeStyle = CONFIG.zeroColor; ctx.lineWidth = CONFIG.zeroWidth * k;
      // Highlighted rows: a soft band behind the row's label and plot.
      ctx.fillStyle = CONFIG.highlightColor;
      CONFIG.highlightRows.forEach(n => { const r = n - 1; if (r >= 0 && r < steps.length) ctx.fillRect(yLabelW, top + r * rowH + gap / 2, w - yLabelW, rowH); });
      if (a0 <= 0 && a1 >= 0) { ctx.beginPath(); ctx.moveTo(X(0), top); ctx.lineTo(X(0), top + rowH * steps.length); ctx.stroke(); }
      steps.forEach((s, r) => {
        const [lo, hi] = CONFIG.sharedScale ? shared : ranges[r], yTop = top + r * rowH + gap, yBot = top + (r + 1) * rowH;
        const Y = v => yBot - (v - lo) / ((hi - lo) || 1) * (yBot - yTop);
        const curve = (sp, color, width) => { ctx.beginPath(); visible.forEach((i, j) => { const x = X(alphaAt(i)), y = Y(sp[i]); j ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.strokeStyle = color; ctx.lineWidth = width * k; ctx.lineJoin = 'round'; ctx.stroke(); };
        const dot = (sp, a, radius, color) => { ctx.beginPath(); ctx.arc(X(a), Y(speedAt(sp, a)), radius, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill(); ctx.lineWidth = radius * 0.3; ctx.strokeStyle = '#fff'; ctx.stroke(); };
        if (s.between && CONFIG.showBetween) {
          const b = s.between, color = rgb(CONFIG.betweenColor, CONFIG.betweenAlpha);
          curve(b.speed, color, CONFIG.curveWidth);
          if (b.marker >= a0 && b.marker <= a1) dot(b.speed, b.marker, tinyR, color);
        }
        curve(s.speed, CONFIG.curveColor, CONFIG.curveWidth);
        if (CONFIG.showMinima) s.minima.filter(a => a >= a0 && a <= a1).forEach(a => { ctx.beginPath(); ctx.arc(X(a), Y(speedAt(s.speed, a)), tinyR, 0, Math.PI * 2); ctx.fillStyle = CONFIG.minimaColor; ctx.fill(); });
        if (s.marker >= a0 && s.marker <= a1) dot(s.speed, s.marker, markR, rgb(CONFIG.markColor));
        if (CONFIG.tokenLabels) { ctx.fillStyle = CONFIG.curveColor; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(labels[r], gutter - CONFIG.labelGap * font, (yTop + yBot) / 2); }
      });
    }

    let queued = false;
    function render() { queued = false; if (!slide.hidden && canvas.width > 1) draw(); }
    function schedule() { if (!queued) { queued = true; requestAnimationFrame(render); } }
    function resize() { const dpr = Math.min(window.devicePixelRatio || 1, 2), r = canvas.getBoundingClientRect(); canvas.width = Math.max(1, Math.round(r.width * dpr)); canvas.height = Math.max(1, Math.round(r.height * dpr)); schedule(); }
    new ResizeObserver(resize).observe(canvas);
    new MutationObserver(() => { if (!slide.hidden) resize(); }).observe(slide, { attributes: true, attributeFilter: ['hidden'] });
    document.fonts.ready.then(schedule);   // labels are measured in the slide's font

    // Slider: one position per precomputed strength (it snaps to them); the value shown is alpha.
    slider.min = 0; slider.max = strengths.length - 1; slider.step = 1; slider.value = current;
    const sync = () => {
      slider.setAttribute('aria-valuetext', strengths[current].alpha.toFixed(CONFIG.alphaDecimals));   // shown by <output data-auto>
      slider.style.setProperty('--fill', `${strengths.length > 1 ? current / (strengths.length - 1) * 100 : 0}%`);
    };
    sync();
    slider.addEventListener('input', () => { current = Number(slider.value); sync(); schedule(); });
    slider.addEventListener('pointerup', () => slider.blur());      // so the arrow keys go back to changing slides
    slider.addEventListener('keydown', e => e.stopPropagation());   // ...but while the slider is focused they adjust it instead
    const autoplay = window.sliderAutoplay(slider, slide, { ...CONFIG.autoplay, playButton: document.querySelector(IDS.play) });   // js/slider-autoplay.js

    window.basinsDemoState = () => ({ config: CONFIG, alpha: strengths[current].alpha, steps: strengths[current].steps, plotExtent, autoplaying: autoplay.playing() });   // for console inspection
    if (!slide.hidden) resize();
  }
})();
