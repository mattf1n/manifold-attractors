const diagram = document.querySelector('#lifecycle-slide');
const divider = document.querySelector('.divider');
// Tab-local presentation state survives live-server reloads.
const boundaryStepKey = 'thesis-boundary-step';
const boundaryPositionKey = 'thesis-boundary-position';
const savedStep = Number(sessionStorage.getItem(boundaryStepKey));
let boundaryStep = Number.isInteger(savedStep)
  ? Math.max(0, savedStep) : 0;
const savedPosition = sessionStorage.getItem(boundaryPositionKey);
let manualBoundary = savedPosition === null ? null : Number(savedPosition);
if (!Number.isFinite(manualBoundary)) manualBoundary = null;
let boundary = 100;
// Editable display labels, bottom to top. Renaming does not change example associations.
const releaseModels = ['OLMo', 'Llama', 'GPT-4', 'GPT-2', 'Grok', 'GPT-5', 'Mythos'];
boundaryStep = Math.min(boundaryStep, releaseModels.length - 1);
// The lifecycle slide's divider is optional: a deck built without that slide (publish.sh) still gets slide navigation.
const dividerLabel = divider?.querySelector('.divider-label');
const illustrations = diagram ? [...diagram.querySelectorAll('[data-release-level]')] : [];
// Shrinks the visible example to fit the slide, so tall or wide cards survive small windows.
function fitIllustration() {
  illustrations.forEach(el => { el.style.transform = ''; });
  const card = illustrations.find(el => !el.hidden);
  if (!card || !card.offsetHeight) return;
  const availH = diagram.querySelector('.lifecycle').clientHeight;   // height of the row the example shares with the diagram
  const scaleH = availH / card.offsetHeight;
  const scaleW = card.scrollWidth > card.clientWidth ? card.clientWidth / card.scrollWidth : 1;
  const scale = Math.min(1, scaleH, scaleW) * 0.98;
  if (scale < 0.98) card.style.transform = `scale(${scale})`;
}
function moveBoundary(value, manual = false) {
  boundary = Math.max(0, Math.min(100, value));
  diagram.style.setProperty('--divider-position', `${boundary}%`);
  divider.setAttribute('aria-valuenow', Math.round(boundary));
  const itemCenters = [...diagram.querySelectorAll('.lifecycle li span')].map(item => {
    const rect = item.getBoundingClientRect();
    return (rect.top + rect.bottom) / 2;
  });
  const lineY = diagram.getBoundingClientRect().top + diagram.getBoundingClientRect().height * boundary / 100;
  const hiddenCount = itemCenters.filter(center => center > lineY).length;
  if (manual) {
    boundaryStep = Math.min(hiddenCount, releaseModels.length - 1);
    manualBoundary = boundary;
  }
  sessionStorage.setItem(boundaryStepKey, String(boundaryStep));
  if (manualBoundary === null) sessionStorage.removeItem(boundaryPositionKey);
  else sessionStorage.setItem(boundaryPositionKey, String(manualBoundary));
  const level = Math.min(hiddenCount, releaseModels.length - 1);
  dividerLabel.textContent = releaseModels[level];
  illustrations.forEach(illustration => {
    illustration.hidden = Number(illustration.dataset.releaseLevel) !== level;
  });
  fitIllustration();
}
function dragBoundary(event) {
  const rect = diagram.getBoundingClientRect();
  moveBoundary(100 * (event.clientY - rect.top) / rect.height, true);
}
if (divider) divider.addEventListener('pointerdown', event => {
  if (event.button !== 0) return;
  event.preventDefault();
  divider.setPointerCapture(event.pointerId);
  dragBoundary(event);
});
if (divider) divider.addEventListener('pointermove', event => {
  if (divider.hasPointerCapture(event.pointerId)) dragBoundary(event);
});
if (divider) divider.addEventListener('pointerup', event => {
  if (divider.hasPointerCapture(event.pointerId)) divider.releasePointerCapture(event.pointerId);
});
if (divider) divider.addEventListener('keydown', event => {
  if (!['ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  event.stopPropagation();
  moveBoundary(event.key === 'Home' ? 0 : event.key === 'End' ? 100
    : boundary + (event.key === 'ArrowUp' ? -1 : 1), true);
});

const stages = diagram ? [...diagram.querySelectorAll('.lifecycle li span')] : [];
// Stops run from below Data to above Text, including the final unreleased-model tier.
function stepBoundary(direction = 1) {
  boundaryStep = Math.max(0, Math.min(stages.length, releaseModels.length - 1, boundaryStep + direction));
  manualBoundary = null;
  positionBoundary();
}

// Slide order follows the sections above; left arrow returns to the title.
const slides = [...document.querySelectorAll('.slide')];
// Persist the current slide in this tab across refreshes.
const slideStorageKey = 'thesis-current-slide';
const savedSlide = Number(sessionStorage.getItem(slideStorageKey));
let currentSlide = Number.isInteger(savedSlide)
  ? Math.max(0, Math.min(slides.length - 1, savedSlide))
  : 0;
// Optional on-page controls: an element with data-next-slide / data-prev-slide moves one slide on click, and hides on the last /
// first slide (publish.sh adds them to the published demo; the talk itself has none).
const stepControls = [...document.querySelectorAll('[data-next-slide], [data-prev-slide]')];
const showCurrentSlide = () => {
  slides.forEach((slide, index) => { slide.hidden = index !== currentSlide; });
  sessionStorage.setItem(slideStorageKey, String(currentSlide));
  stepControls.forEach(el => { el.hidden = el.hasAttribute('data-next-slide') ? currentSlide >= slides.length - 1 : currentSlide <= 0; });
};
stepControls.forEach(el => el.addEventListener('click', event => {
  event.preventDefault();
  currentSlide = Math.max(0, Math.min(slides.length - 1, currentSlide + (el.hasAttribute('data-next-slide') ? 1 : -1)));
  showCurrentSlide();
  positionBoundary();
  el.blur();   // so the arrow keys keep changing slides
}));
function positionBoundary() {
  if (!diagram || diagram.hidden) return;
  if (manualBoundary !== null) {
    moveBoundary(manualBoundary);
    return;
  }
  const rect = diagram.getBoundingClientRect();
  const itemEl = stages[Math.max(0, boundaryStep - 1)];
  const item = itemEl.getBoundingClientRect();
  // After the first stop, the line sits inside the first hidden box: below its top edge, above its text.
  const insideOffset = parseFloat(getComputedStyle(itemEl).paddingTop) * 0.5;
  const y = boundaryStep === 0 ? item.bottom + 16 : item.top + insideOffset;
  moveBoundary(100 * (y - rect.top) / rect.height);
}
showCurrentSlide();
positionBoundary();
window.addEventListener('resize', positionBoundary);
// Logos and fonts change card heights after load.
diagram?.addEventListener('load', fitIllustration, true);
document.fonts.ready.then(positionBoundary);
document.addEventListener('keydown', (event) => {
  // Up/Down move the current slide's slider (a focused slider handles its own arrows, and stops the event before it gets here).
  // One press moves 1/40 of the range, or one step if that is larger; holding the key repeats.
  if ((event.key === 'ArrowUp' || event.key === 'ArrowDown') && !event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey) {
    const slider = slides[currentSlide].querySelector('input[type="range"]:not(:disabled)');
    if (slider) {
      event.preventDefault();
      const min = Number(slider.min || 0), max = Number(slider.max || 100), step = Number(slider.step) || 1;
      const move = Math.max(step, (max - min) / 40) * (event.key === 'ArrowUp' ? 1 : -1);
      slider.value = Math.min(max, Math.max(min, Number(slider.value) + move));
      slider.dispatchEvent(new Event('input', { bubbles: true }));
      return;
    }
  }
  if (event.repeat || event.altKey || event.ctrlKey || event.metaKey) return;
  if (slides[currentSlide] === diagram) {
    // Up/Down step the blur boundary (Up moves it up, revealing fewer stages); Left/Right/Enter always change slides.
    const boundaryMove = { ArrowUp: 1, ArrowDown: -1 }[event.key];
    if (boundaryMove) {
      event.preventDefault();
      const next = boundaryStep + boundaryMove;
      if (next >= 0 && next <= releaseModels.length - 1) stepBoundary(boundaryMove);
      return;
    }
  }
  const direction = { Enter: 1, ArrowRight: 1, ArrowLeft: -1 }[event.key];
  if (!direction) return;
  event.preventDefault();
  currentSlide = Math.max(0, Math.min(slides.length - 1, currentSlide + direction));
  showCurrentSlide();
  positionBoundary();
});
