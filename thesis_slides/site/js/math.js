// LaTeX math in the slides. Write \( ... \) for inline math and \[ ... \] for display math directly in the HTML text;
// on load, this script converts each formula to MathML with Temml (assets/temml/temml.min.js, loaded just before this
// file) and then adjusts the MathML to the deck's typesetting conventions (see CONFIG). The browser draws the result in
// Libertinus Math (font and sizes: the math CSS at the top of index.html). Dollar signs are not delimiters, so prices
// like $0.50 are safe. Text inside <script>, <style>, <code>, <pre>, <textarea> or any element with a data-no-math
// attribute is left alone (use data-no-math to show a literal \( in the text). A formula must sit in one piece of text:
// no HTML tags inside it (write < and > as &lt; and &gt;). Runs before the slide scripts, so they see the final MathML.
(() => {
  const CONFIG = {
    inline: ['\\(', '\\)'],    // inline math delimiters
    display: ['\\[', '\\]'],   // display (block) math delimiters
    skip: 'script, style, code, pre, textarea, math, [data-no-math]',   // elements whose text is never converted
    macros: {},                // extra LaTeX macros for every formula, e.g. { '\\R': '\\mathbb{R}' }

    // TeX raises an exponent above a single-symbol base as if the base had no height (and lowers a subscript as if it
    // had no depth); the browser uses the base's real height, which puts d² 0.09em higher than LaTeX. When on, the base
    // of msup/msub/msubsup is wrapped in <mpadded height="0"> (and depth="0" for subscripts) if it is a single token
    // (mi, mn or mo; 10 counts too, since TeX attaches the exponent of 10^6 to the 0 alone).
    flatScriptBase: true,

    // Implicit multiplication: an invisible times (U+2062, no width) between adjacent letters/numbers, e.g. n d p, 2x.
    // Only for screen readers; it doesn't change the layout.
    invisibleTimes: true,

    // Optical spacing between letters. The font's italic letters have their italic correction folded into their advance
    // width (so d² and d( don't collide; see the math CSS), and TeX also adds it, but after a letter that leans right,
    // like d, the next letter then looks set apart: ndp reads as "nd p". When on, a letter followed by another letter
    // (n d p, or a letter carrying a script like p²) is pulled back by its italic correction from the table below
    // (em, from Libertinus Math's MATH table). Letters missing from the table are left as is; edit to taste.
    tightenLetters: true,
    italicCorrection: {
      a: 0.026, b: 0.034, c: 0.054, d: 0.078, e: 0.043, f: 0.168, g: 0.059, j: 0.063, k: 0.054, l: 0.065, m: 0.023,
      n: 0.022, o: 0.038, p: 0.040, q: 0.036, r: 0.092, s: 0.044, t: 0.076, u: 0.036, v: 0.097, w: 0.097, x: 0.081,
      y: 0.096, z: 0.051,
      B: 0.024, C: 0.074, D: 0.046, E: 0.047, F: 0.116, G: 0.062, H: 0.124, I: 0.127, J: 0.126, K: 0.105, M: 0.090,
      N: 0.142, O: 0.058, P: 0.095, Q: 0.058, R: 0.010, S: 0.059, T: 0.132, U: 0.146, V: 0.148, W: 0.151, X: 0.124,
      Y: 0.131, Z: 0.115,
    },
  };
  // Already done by Temml, so no option here: \( ( \) parentheses are non-stretchy (\left( \right) stretch), and a / is
  // set with no space around it. Temml's own CSS and fonts are not used: they would replace Libertinus Math.

  const MATHML = 'http://www.w3.org/1998/Math/MathML';
  const TOKENS = new Set(['mi', 'mn', 'mo']);
  const SCRIPTS = new Set(['msup', 'msub', 'msubsup']);
  const isLetter = el => el?.localName === 'mi' && !el.hasAttribute('mathvariant') && /^[A-Za-z]$/.test(el.textContent);
  const isInvisibleTimes = el => el?.localName === 'mo' && el.textContent === '\u2062';
  const scriptBase = el => (SCRIPTS.has(el?.localName) ? el.firstElementChild : null);
  const baseLetter = el => {   // the letter starting el: el itself, or the (possibly mpadded-wrapped) base of a script
    const base = scriptBase(el);
    if (!base) return isLetter(el) ? el : null;
    return isLetter(base) ? base : base.localName === 'mpadded' && isLetter(base.firstElementChild) ? base.firstElementChild : null;
  };

  // Adjust Temml's MathML to the conventions in CONFIG.
  function postProcess(math) {
    if (CONFIG.flatScriptBase) {
      for (const script of math.querySelectorAll('msup, msub, msubsup')) {
        const base = script.firstElementChild;
        if (!base || !TOKENS.has(base.localName)) continue;
        const pad = document.createElementNS(MATHML, 'mpadded');
        if (script.localName !== 'msub') pad.setAttribute('height', '0');
        if (script.localName !== 'msup') pad.setAttribute('depth', '0');
        script.replaceChild(pad, base);
        pad.append(base);
      }
    }
    // Rows of letters: Temml puts the letters of ndp side by side in an <mrow> (or straight in <math>).
    for (const row of [math, ...math.querySelectorAll('mrow, mstyle, mpadded, mtd, msqrt, menclose')]) {
      for (let el = row.firstElementChild; el; el = el.nextElementSibling) {
        const next = el.nextElementSibling;
        const letter = isLetter(el) || el.localName === 'mn';
        if (CONFIG.invisibleTimes && letter && baseLetter(next)) {
          const times = document.createElementNS(MATHML, 'mo');
          times.textContent = '\u2062';
          el.after(times);
        }
        const following = isInvisibleTimes(el.nextElementSibling) ? el.nextElementSibling.nextElementSibling : el.nextElementSibling;
        const ic = CONFIG.italicCorrection[el.textContent];
        if (CONFIG.tightenLetters && isLetter(el) && ic && baseLetter(following)) el.style.marginRight = `-${ic}em`;
      }
    }
    return math;
  }

  function toMathML(tex, displayMode) {
    const holder = document.createElement('span');
    temml.render(tex, holder, { displayMode, macros: CONFIG.macros, throwOnError: false });
    const math = holder.firstElementChild;
    if (math?.localName !== 'math') console.warn('math.js: could not convert', tex, holder.textContent);
    return math?.localName === 'math' ? postProcess(math) : holder;   // on error Temml returns a red message instead
  }

  // Split one text node into text and formulas.
  function convert(node) {
    const text = node.data, parts = document.createDocumentFragment();
    let pos = 0;
    while (pos < text.length) {
      const i = text.indexOf(CONFIG.inline[0], pos), j = text.indexOf(CONFIG.display[0], pos);
      const display = j >= 0 && (i < 0 || j < i);
      const [open, close] = display ? CONFIG.display : CONFIG.inline;
      const start = display ? j : i;
      const end = start < 0 ? -1 : text.indexOf(close, start + open.length);
      if (end < 0) break;   // no (complete) formula left
      parts.append(text.slice(pos, start), toMathML(text.slice(start + open.length, end), display));
      pos = end + close.length;
    }
    if (pos === 0) return;
    parts.append(text.slice(pos));
    node.replaceWith(parts);
  }

  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT, node =>
    node.nodeType === Node.ELEMENT_NODE ? (node.matches(CONFIG.skip) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_SKIP)
      : node.data.includes(CONFIG.inline[0]) || node.data.includes(CONFIG.display[0]) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP);
  const found = [];
  while (walker.nextNode()) found.push(walker.currentNode);   // collect first: converting changes the tree
  found.forEach(convert);
})();
