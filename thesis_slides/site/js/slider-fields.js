// Slider values shown with no per-slider code. Any <output for="SLIDER-ID" data-auto></output> shows that slider's current
// value: its aria-valuetext (the readable value a slide script sets, e.g. "8,192" or "$48.00"), or its raw value if no
// script sets one. Kept in sync on every input event and whenever a script changes aria-valuetext (snapping, reset, start).
// Outputs added to the page later are filled in as well. Load after the slide scripts so their starting values are shown.
(() => {
  const AUTO = 'output[data-auto][for]';
  const sliderOf = output => document.getElementById(output.htmlFor[0]);   // the first id in the output's for="..."
  const textOf = slider => slider.getAttribute('aria-valuetext') ?? slider.value;
  function show(output) {
    const slider = sliderOf(output);
    if (!slider) return;
    const text = textOf(slider);
    if (output.textContent !== text) output.textContent = text;   // only on change, so the observer below does not loop
  }
  // Every auto output that refers to this slider by id.
  const showFor = slider => { if (slider.id) document.querySelectorAll(AUTO).forEach(o => { if (o.htmlFor.contains(slider.id)) show(o); }); };

  document.querySelectorAll(AUTO).forEach(show);
  // Capture phase, so it runs even if a slide script stops the event; a script's new aria-valuetext arrives through the observer.
  document.addEventListener('input', e => { if (e.target instanceof HTMLInputElement) showFor(e.target); }, true);
  new MutationObserver(records => {
    for (const r of records) {
      if (r.type === 'attributes') showFor(r.target);   // a script set aria-valuetext
      else r.addedNodes.forEach(n => {   // new outputs, alone or inside an added element
        if (n.nodeType !== Node.ELEMENT_NODE) return;
        if (n.matches(AUTO)) show(n);
        n.querySelectorAll(AUTO).forEach(show);
      });
    }
  }).observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ['aria-valuetext'] });
})();
