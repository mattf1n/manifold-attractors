// "Steering between language feature basins" slide: the paper's English-Chinese steering figure ("Speed profiles along the en-zh
// steering direction"), one panel per document. A token's activation x entering block 16 of Llama 3.1 8B is rotated on its sphere by
// the angle theta toward Chinese (positive: along the en-zh centroid difference) or toward English (negative). Each panel plots the
// speed of blocks 16-19 along that rotation (the curve), with bands colored by the script of the token the rotated activation predicts
// (red: English/Latin, blue: Chinese, grey: neutral or other), like the paper. The slider sets theta (the steering strength); a dot
// marks it on every curve, and above each plot the document's tail is followed by the token predicted at that strength, highlighted in
// its script's color, with its gloss, and under a Chinese document the paper's free translation of it (glosses and translations: the
// paper's). Data: assets/enzh-steering.js (the paper's frozen outputs, see its header). Colors and text sizes are CSS variables on
// #enzh-slide (index.html); edit CONFIG for the rest.
(() => {
  const IDS = { slide: '#enzh-slide', figure: '#enzh-figure', slider: '#enzh-strength', play: '#enzh-play' };
  const CONFIG = {
    panels: [0, 1, 2, 3, 4, 5, 6, 7],   // which documents, in reading order (indices into ENZH_STEERING.panels: 0-3 English, 4-7 Chinese, as in the paper)
    limit: 45,                // the slider and each plot's horizontal range: -limit..+limit degrees (the paper shows ±45; the data cover ±90)
    start: 0,                 // angle shown when the slide opens (0 = the original, unsteered activation)
    decimals: 1,              // digits of the angle shown at the slider's end
    glosses: true,            // English glosses under Chinese (and other non-Latin) text; false hides them (their row stays, so the layout does not move)
    translations: true,       // the paper's free translation under each Chinese document, in single quotes; false hides it (its line stays)
    sharedScale: true,        // true: every panel uses one speed scale (as the paper's figure); false: each panel is scaled to its own range
    speedPadding: 0.08,       // empty space above and below the curve, as a fraction of the plot's height
    curveColor: '#202a36', curveWidth: 2,             // the speed profile (width in CSS pixels)
    zeroColor: 'rgba(255,255,255,0.9)', zeroWidth: 1.5,   // vertical line at 0 (the original activation), white as in the paper
    bandAlpha: [0.2, 0.36],   // band opacities, alternating light/dark so neighbouring bands of the same script stay distinguishable (as the paper)
    dotColor: '#202a36',      // dot at the current strength on each curve (size: the deck's --mark-small), with a ring in dotRing
    dotRing: '#ffffff',
    grabSlop: 8,              // a panel's dot can be grabbed (and dragged to change the strength) within this many CSS pixels of its edge
    yLabel: 'Density \u2193',   // y-axis label (as on the refusal steering slide), shown once: upright, above the first English document at the figure's left edge ('' hides it; \u2193 = ↓)
    // The play button at the start of the slider row sweeps the slider (js/slider-autoplay.js), as on the next slide: a sine between
    // -amplitude and +amplitude degrees (the slider itself keeps -limit..+limit for manual use; amplitude >= limit sweeps it all), one
    // full cycle low → high → low taking period seconds, from the current position (heading toward +), until pressed again. Started
    // from outside ±amplitude it glides into the sweep without a jump. Adjusting the slider or leaving the slide stops it. Nothing
    // plays on its own.
    autoplay: { period: 14, amplitude: 25 }
  };

  // Script of a token -> CSS class and color variable on the slide (--enzh-latin, --enzh-chinese, --enzh-other).
  const scriptKey = script => script === 'Latin' ? 'latin' : script === 'Chinese' ? 'chinese' : 'other';
  // How a token is written in the document: newlines as \n (as the paper and the previous slide). A whitespace-only token is drawn as ␣ (below).
  const shown = text => text.replace(/\n/g, '\\n');

  setup();

  function setup() {
    const DATA = window.ENZH_STEERING;
    const slide = document.querySelector(IDS.slide), figure = document.querySelector(IDS.figure), slider = document.querySelector(IDS.slider);
    if (!DATA || !slide || !figure || !slider) return;
    const { min, step, count } = DATA.theta, thetaAt = i => min + step * i;
    // The y-axis label: one line at the top of the figure, spanning its columns, so it sits above the first document (top left) and
    // the panel rows below keep equal heights (styled by #enzh-slide .enzh-ylabel).
    if (CONFIG.yLabel) { const label = document.createElement('div'); label.className = 'enzh-ylabel'; label.textContent = CONFIG.yLabel; figure.append(label); }
    // The grid points in -limit..limit: one slider position each.
    const visible = []; for (let i = 0; i < count; i++) if (Math.abs(thetaAt(i)) <= CONFIG.limit + 1e-9) visible.push(i);
    const lo = thetaAt(visible[0]), hi = thetaAt(visible[visible.length - 1]);
    let current = visible.reduce((best, i, k) => Math.abs(thetaAt(i) - CONFIG.start) < Math.abs(thetaAt(visible[best]) - CONFIG.start) ? k : best, 0);

    // Panels: the document (units of text over gloss, then the stack of tokens it can predict) above a canvas.
    const panels = CONFIG.panels.map(p => DATA.panels[p]).filter(Boolean).map(data => {
      const el = document.createElement('div'); el.className = 'enzh-panel';
      const doc = document.createElement('div'); doc.className = `enzh-doc enzh-doc-${data.language}`;
      const unit = (text, gloss, grammatical) => {
        const u = document.createElement('span'); u.className = 'enzh-unit';
        const src = document.createElement('span'); src.textContent = text;
        const g = document.createElement('span'); g.className = 'enzh-gloss' + (grammatical ? ' grammatical' : ''); g.textContent = CONFIG.glosses ? gloss : '';
        u.append(src, g); return u;
      };
      let shrink = null;   // English text: dropped a word at a time from its left when the panel is too narrow (see fitDocs)
      const words = [];    // Chinese words (units after the "…"): dropped one at a time from the left when the panel is too narrow
      data.units.forEach(u => {
        const node = unit(shown(u.source), u.gloss, u.grammatical);
        if (data.language === 'en' && u.source !== '…') { node.classList.add('enzh-shrink'); shrink = { node, words: shown(u.source).split(' ') }; }
        else if (u.source !== '…') words.push(node);
        doc.append(node);
      });
      // One unit per distinct token this panel predicts in the slider's range; only the current one is visible.
      const next = document.createElement('span'); next.className = 'enzh-next';
      const options = new Map();
      visible.forEach(i => {
        const id = data.tokens[i];
        if (options.has(id)) return;
        const t = DATA.tokens[id], text = t.text.trimStart(), lead = text.length < t.text.length;
        const u = unit('', t.gloss, t.grammatical), src = u.firstChild;
        if (lead && text) { const s = document.createElement('span'); s.className = 'lead'; src.append(s); }   // the token's leading space
        const mark = document.createElement('mark'); mark.className = scriptKey(t.script); mark.textContent = text ? shown(text) : '\u2423';
        src.append(mark);
        next.append(u); options.set(id, u);
      });
      doc.append(next);
      const canvas = document.createElement('canvas'); canvas.className = 'enzh-plot';
      // The panel is two rows of the figure's grid (CSS): its text (the document, and under a Chinese document the paper's free
      // translation; English ones have none, so their plot sits right under the document), then its plot. The text rows of a panel row
      // share one height and every plot row an equal share of the rest, so all plots are the same size and a row's plots line up.
      const text = document.createElement('div'); text.className = 'enzh-text';
      text.append(doc);
      if (data.translation) {
        const translation = document.createElement('div'); translation.className = 'enzh-translation';
        translation.textContent = CONFIG.translations ? `\u2018${data.translation}\u2019` : '';
        text.append(translation);
      }
      el.append(text, canvas); figure.append(el);
      return { data, canvas, ctx: canvas.getContext('2d'), doc, options, shownUnit: null, shrink, words };
    });

    // Speed scale: over the range shown, for all panels at once (shared) or per panel; fixed for every strength.
    const ranges = panels.map(p => { let a = Infinity, b = -Infinity; visible.forEach(i => { a = Math.min(a, p.data.speed[i]); b = Math.max(b, p.data.speed[i]); }); return [a, b]; });
    const shared = [Math.min(...ranges.map(r => r[0])), Math.max(...ranges.map(r => r[1]))];
    let plotExtent = null;   // the plots' left and right edge in CSS pixels, from the last draw (for console inspection)

    function draw(p, r) {
      const { canvas, ctx, data } = p, w = canvas.width, h = canvas.height, k = w / (canvas.clientWidth || w);
      const style = getComputedStyle(slide), markR = parseFloat(getComputedStyle(canvas).getPropertyValue('--mark-small')) / 2 * k;
      const color = (key, a) => `rgba(${style.getPropertyValue(`--enzh-${key}`).trim().split(/[\s,]+/).join(',')},${a})`;
      ctx.clearRect(0, 0, w, h);
      // Plot area: inset by the dot's radius so the dot stays inside the canvas at the ends.
      const x0 = markR, x1 = w - markR, y0 = markR, y1 = h - markR;
      const X = t => x0 + (t - lo) / (hi - lo) * (x1 - x0);
      const [s0, s1] = CONFIG.sharedScale ? shared : ranges[r], pad = CONFIG.speedPadding * (y1 - y0);
      const Y = v => y1 - pad - (v - s0) / ((s1 - s0) || 1) * (y1 - y0 - 2 * pad);
      if (r === 0) plotExtent = [x0 / k, x1 / k];
      // Bands: each grid point's token owns the half-step on either side of it (the paper's Voronoi cells); runs of one token are one band.
      let band = 0;
      for (let j = 0; j < visible.length;) {
        let e = j; while (e + 1 < visible.length && data.tokens[visible[e + 1]] === data.tokens[visible[j]]) e++;
        const a = Math.max(lo, thetaAt(visible[j]) - step / 2), b = Math.min(hi, thetaAt(visible[e]) + step / 2);
        ctx.fillStyle = color(scriptKey(DATA.tokens[data.tokens[visible[j]]].script), CONFIG.bandAlpha[band++ % CONFIG.bandAlpha.length]);
        ctx.fillRect(X(a), y0, X(b) - X(a), y1 - y0);
        j = e + 1;
      }
      // Line at 0 (the original activation).
      if (lo <= 0 && hi >= 0) { ctx.strokeStyle = CONFIG.zeroColor; ctx.lineWidth = CONFIG.zeroWidth * k; ctx.beginPath(); ctx.moveTo(X(0), y0); ctx.lineTo(X(0), y1); ctx.stroke(); }
      // The speed profile.
      ctx.beginPath(); visible.forEach((i, j) => { const x = X(thetaAt(i)), y = Y(data.speed[i]); j ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.strokeStyle = CONFIG.curveColor; ctx.lineWidth = CONFIG.curveWidth * k; ctx.lineJoin = 'round'; ctx.stroke();
      // The current strength.
      const i = visible[current], dotX = X(thetaAt(i)), dotY = Y(data.speed[i]);
      ctx.beginPath(); ctx.arc(dotX, dotY, markR, 0, Math.PI * 2);
      ctx.fillStyle = CONFIG.dotColor; ctx.fill(); ctx.lineWidth = markR * 0.35; ctx.strokeStyle = CONFIG.dotRing; ctx.stroke();
      // This panel's angle axis and dot, in CSS pixels from the canvas's top left, for dragging the dot (see the drag handlers below).
      p.geom = { x0: x0 / k, x1: x1 / k, dotX: dotX / k, dotY: dotY / k, r: markR / k };
    }

    // The document's predicted token at the current strength.
    function showTokens() {
      panels.forEach(p => {
        const u = p.options.get(p.data.tokens[visible[current]]);
        if (u === p.shownUnit) return;
        if (p.shownUnit) p.shownUnit.classList.remove('current');
        u.classList.add('current'); p.shownUnit = u;
      });
    }

    let queued = false;
    function render() { queued = false; if (slide.hidden) return; showTokens(); panels.forEach((p, r) => { if (p.canvas.width > 1) draw(p, r); }); }
    function schedule() { if (!queued) { queued = true; requestAnimationFrame(render); } }
    // Documents that do not fit beside the widest predicted token keep as much of their end as fits: English text whole words at a time
    // (so no letter is cut), Chinese whole words with their glosses. The steered token (the last word) always stays.
    function fitDocs() {
      panels.forEach(({ doc, shrink, words }) => {
        if (shrink) {
          const text = shrink.node.firstChild;
          for (let n = shrink.words.length; n >= 1; n--) {
            text.textContent = shrink.words.slice(-n).join(' ');
            if (shrink.node.scrollWidth <= shrink.node.clientWidth + 0.5) break;
          }
        }
        words.forEach(w => { w.hidden = false; });
        for (let n = 0; n < words.length - 1 && doc.scrollWidth > doc.clientWidth + 0.5; n++) words[n].hidden = true;
      });
    }
    function resize() {
      fitDocs();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      panels.forEach(p => { const r = p.canvas.getBoundingClientRect(); p.canvas.width = Math.max(1, Math.round(r.width * dpr)); p.canvas.height = Math.max(1, Math.round(r.height * dpr)); });
      schedule();
    }
    document.fonts.ready.then(() => { if (!slide.hidden) resize(); });   // the documents are measured in the slide's fonts
    const observer = new ResizeObserver(() => { if (!slide.hidden) resize(); });
    panels.forEach(p => observer.observe(p.canvas));
    new MutationObserver(() => { if (!slide.hidden) resize(); }).observe(slide, { attributes: true, attributeFilter: ['hidden'] });
    showTokens();

    // Slider: one position per grid angle in -limit..limit (0.5° apart), centered on 0; the value shown is the angle.
    slider.min = 0; slider.max = visible.length - 1; slider.step = 1; slider.value = current;
    const format = t => `${t > 0 ? '+' : t < 0 ? '\u2212' : ''}${Math.abs(t).toFixed(CONFIG.decimals)}\u00b0`;
    const sync = () => {
      slider.setAttribute('aria-valuetext', format(thetaAt(visible[current])));   // shown by <output data-auto>
      slider.style.setProperty('--fill', `${visible.length > 1 ? current / (visible.length - 1) * 100 : 0}%`);
    };
    sync();
    slider.addEventListener('input', () => { current = Number(slider.value); sync(); schedule(); });
    slider.addEventListener('pointerup', () => slider.blur());      // so the arrow keys go back to changing slides
    slider.addEventListener('keydown', e => e.stopPropagation());   // ...but while the slider is focused they adjust it instead
    // The sweep's range in slider positions: the grid angles closest to -amplitude and +amplitude.
    const nearest = t => visible.reduce((best, i, k) => Math.abs(thetaAt(i) - t) < Math.abs(thetaAt(visible[best]) - t) ? k : best, 0);
    const autoplay = window.sliderAutoplay(slider, slide, { ...CONFIG.autoplay, playButton: document.querySelector(IDS.play),
      min: nearest(-(CONFIG.autoplay.amplitude ?? CONFIG.limit)), max: nearest(CONFIG.autoplay.amplitude ?? CONFIG.limit) });

    // Dragging a panel's dot moves the steering strength: the pointer's x on that panel's angle axis (the inverse of its angle → x map,
    // snapped to the nearest grid angle, clamped to the slider's range) becomes the slider's value. It goes through the slider (value +
    // 'input'), so the slider, its readout, every panel and the documents' tokens update together, and the sweep stops as when the
    // slider is moved by hand. The dot can be grabbed within CONFIG.grabSlop CSS pixels of its edge.
    let dragging = null;   // the panel being dragged, while the pointer is held
    const local = (p, e) => { const r = p.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    const overDot = (p, e) => { if (!p.geom) return false; const [x, y] = local(p, e), g = p.geom; return Math.hypot(x - g.dotX, y - g.dotY) <= g.r + CONFIG.grabSlop; };
    function dragTo(p, e) {
      const g = p.geom, f = Math.min(1, Math.max(0, (local(p, e)[0] - g.x0) / ((g.x1 - g.x0) || 1)));
      const position = nearest(lo + f * (hi - lo));
      if (position === current) return;
      slider.value = position;
      slider.dispatchEvent(new Event('input', { bubbles: true }));
    }
    panels.forEach(p => {
      const c = p.canvas;
      c.addEventListener('pointerdown', e => {
        if (!overDot(p, e)) return;
        dragging = p; c.setPointerCapture(e.pointerId); c.style.cursor = 'grabbing'; e.preventDefault();
      });
      c.addEventListener('pointermove', e => {
        if (dragging === p) dragTo(p, e);
        else if (!dragging) c.style.cursor = overDot(p, e) ? 'grab' : '';
      });
      const release = e => { if (dragging !== p) return; dragging = null; c.style.cursor = overDot(p, e) ? 'grab' : ''; };
      c.addEventListener('pointerup', release);
      c.addEventListener('pointercancel', release);
    });

    window.enzhDemoState = () => ({ config: CONFIG, theta: thetaAt(visible[current]), plotExtent, dots: panels.map(p => p.geom),
      tokens: panels.map(p => DATA.tokens[p.data.tokens[visible[current]]].text), autoplaying: autoplay.playing() });   // for console inspection (dots: each panel's dot and axis, CSS px in its canvas)
    if (!slide.hidden) resize();
  }
})();
