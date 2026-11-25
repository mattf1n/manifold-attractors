// Slider play button: a <button class="slider-play"> at the start of a slider's row plays the slider, as if someone were dragging
// it. Nothing plays on its own: only pressing the button starts it, pressing it again pauses. It drives the slider the way a user
// does (sets the value and dispatches 'input'), so the slide's script and any <output data-auto> update as usual.
// Usage, from a slide script: sliderAutoplay(slider, slide, { playButton, ...motion options below }).
// Button markup and styles: see .slider-play in index.html. It shows a pause icon (class "playing", aria-label "Pause") while
// playing and a play icon (aria-label "Play") otherwise; after a mouse click it gives up focus, so the arrow keys go back to
// changing slides.
// Adjusting the slider itself (pressing it, keys on it, its own 'input') stops playback, and so does using an element listed in
// pauseOn (pressing it, or moving the pointer over it with a button held: a drag), e.g. a rotatable 3-D panel. Leaving the slide
// stops it (it does not continue on return); switching browser tabs and back carries on. Nothing runs (no timer, no animation
// frames) while it is not playing.
// Motion:
// - mode 'sine' (default): sweeps up and down for as long as it plays, following a sine over time (slowing at the ends), through
//   every step; one full cycle min → max → min takes period seconds. It starts from the slider's value (heading up unless at the
//   top). min/max (optional, in slider values) limit the sweep to part of the slider's range (the slider itself keeps its full range
//   for manual use). Started from a value outside min..max (left there by hand), it first glides to the nearer end (starting and
//   arriving at rest, at about the sweep's speed) and sweeps on from there, so it never jumps.
// - mode 'loop': holds at min for pauseStart ms, steps up to max at an even pace (every value gets the same time; the whole climb
//   takes duration seconds), holds at max for pauseEnd ms, then jumps back to min and repeats. It starts from the slider's value
//   (from min if the value is at max or outside min..max). repeat: false plays the climb once and stops at max (no end pause).
window.sliderAutoplay = (() => {
  const CONFIG = {
    period: 14,         // sine: seconds for a full cycle (min → max → min)
    pauseStart: 1500,   // loop: ms held at min before climbing
    pauseEnd: 2500,     // loop: ms held at max before jumping back to min
    duration: 8         // loop: seconds for the climb from min to max
  };

  return function sliderAutoplay(slider, slide, options = {}) {
    const button = options.playButton, period = options.period ?? CONFIG.period;
    const loop = options.mode === 'loop', pauseStart = options.pauseStart ?? CONFIG.pauseStart, pauseEnd = options.pauseEnd ?? CONFIG.pauseEnd,
      climb = (options.duration ?? CONFIG.duration) * 1000, cycle = pauseStart + climb + pauseEnd;
    const repeat = options.repeat ?? true;
    let on = false;   // playing, as the button shows (it may be waiting for the browser tab to be shown again)
    let frame = 0, driving = false, start = 0, phase = 0;
    let glide = null;   // { from, to, duration (s) } while moving from outside the sweep range to its nearer end
    // The sweep's range: the slider's own, narrowed to options.min..options.max when given.
    const range = () => {
      const lo = Number(slider.min), hi = Number(slider.max);
      const min = Math.max(lo, options.min ?? lo), max = Math.min(hi, options.max ?? hi);
      return { min, max, span: max - min };
    };
    const visible = () => !slide.hidden && !document.hidden;

    function syncButton() {   // the button shows what pressing it will do
      button.classList.toggle('playing', on); button.setAttribute('aria-label', on ? 'Pause' : 'Play');
    }
    function halt() { cancelAnimationFrame(frame); frame = 0; }
    function stop() { on = false; halt(); syncButton(); }
    function run() {   // (re)start the motion from the slider's value
      halt();
      const { min, max, span } = range();
      if (span <= 0 || !visible()) return;
      const value = Number(slider.value);
      if (loop) {
        // Loop clock: 0..pauseStart holds min, then the climb, then pauseEnd at max. Start where the slider is (min if outside or at max).
        const step = Number(slider.step) || 1, slots = Math.round(span / step) + 1;
        const offset = value > min && value < max ? pauseStart + climb * Math.round((value - min) / step) / slots : 0;
        glide = null; start = performance.now() - offset; frame = requestAnimationFrame(tick); return;
      }
      if (value < min || value > max) {
        // Outside the sweep: glide to the nearer end with an ease-in-out (at rest at both ends), taking as long as the sweep would to
        // cover that distance on average (half a period per span), then sweep on from that end, where the sine is at rest too.
        const to = value < min ? min : max;
        glide = { from: value, to, duration: Math.max(0.4, period / 2 * Math.abs(to - value) / span) };
        phase = value < min ? 0 : Math.PI;   // at min the sweep heads up, at max down
      } else {
        // value = min + span * (1 - cos(phase + 2 pi t / period)) / 2; start at the phase that gives the slider's current value
        // (on the rising half, so it heads up unless it is at the top), so there is no jump.
        glide = null;
        phase = Math.acos(1 - 2 * (value - min) / span);
      }
      start = performance.now();
      frame = requestAnimationFrame(tick);
    }
    function set(value) {
      if (value === Number(slider.value)) return;
      slider.value = value;
      driving = true; slider.dispatchEvent(new Event('input', { bubbles: true })); driving = false;
    }
    function tick(now) {
      const { min, max, span } = range();
      let t = (now - start) / 1000, value;
      if (loop) {
        // Even pace: the climb is split into one equal slot per slider value (min and max included), on top of the two pauses.
        const step = Number(slider.step) || 1, slots = Math.round(span / step) + 1;
        const c = repeat ? (now - start) % cycle : now - start, u = Math.max(0, (c - pauseStart) / climb);
        value = Math.min(max, min + step * Math.min(slots - 1, Math.floor(u * slots)));
        if (!repeat && c >= pauseStart + climb) { set(max); frame = 0; stop(); return; }   // played once: leave it at max and stop
      } else if (glide && t < glide.duration) value = Math.round(glide.from + (glide.to - glide.from) * (1 - Math.cos(Math.PI * t / glide.duration)) / 2);
      else {
        if (glide) { start += glide.duration * 1000; t -= glide.duration; glide = null; }   // the glide is over: the sweep's clock starts now
        value = Math.min(max, Math.max(min, Math.round(min + span * (1 - Math.cos(phase + 2 * Math.PI * t / period)) / 2)));
      }
      set(value);
      frame = requestAnimationFrame(tick);
    }

    button.addEventListener('click', () => {
      if (on) { stop(); return; }   // pause
      on = true; syncButton(); run();
    });
    button.addEventListener('pointerup', () => button.blur());   // so Enter and the arrow keys go back to changing slides
    button.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') e.stopPropagation(); });   // ...but while it is focused they press it
    const adjusted = () => { if (on && !driving) stop(); };
    slider.addEventListener('pointerdown', adjusted);
    slider.addEventListener('keydown', adjusted);
    slider.addEventListener('input', adjusted);
    for (const el of options.pauseOn ?? []) {
      el.addEventListener('pointerdown', adjusted);
      el.addEventListener('pointermove', e => { if (e.buttons) adjusted(); });
    }
    new MutationObserver(() => { if (slide.hidden) stop(); }).observe(slide, { attributes: true, attributeFilter: ['hidden'] });   // leaving the slide stops it
    document.addEventListener('visibilitychange', () => (document.hidden ? halt() : on && run()));   // tab hidden and back: carries on
    syncButton();
    return { stop, playing: () => on };   // for console inspection
  };
})();
