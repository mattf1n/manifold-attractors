// Footnote citations from a BibTeX file.
// In the markup, write <p class="footnote" data-cite="key1 key2"></p> (space-separated BibTeX keys); on load this script
// fetches CONFIG.bibFile, parses it and fills each [data-cite] element with one <span class="reference"> per reference
// (a block that never breaks across footnote columns; see .footnote .reference in index.html):
//   <a href=url>Title</a>. Authors. <em>Venue</em>. Year.
// The venue is the entry's booktitle (or journal) as written, so keep it short in the .bib (e.g. booktitle = {ICLR}).
// A key missing from the .bib is shown as the bare key and reported with console.warn, so typos are visible.
// References to the deck author's own papers end with a row of the co-authors' headshots (CONFIG.headshots), in author order,
// under the reference text; slides with such a row keep room for it (see .footnote .headshots in index.html).
(() => {
  const CONFIG = {
    bibFile: 'references.bib',            // path of the BibTeX file, relative to index.html
    deckAuthor: 'Matthew Finlayson',      // the deck author ("First Last"); their papers name everyone up to and including them
    // Author lists of the deck author's papers: with this many authors or more, stop after the deck author and add "et al."
    // ("A, Matthew Finlayson, et al."; "Matthew Finlayson et al." when first). Shorter lists, or the deck author last, list everyone.
    ownEtAlFrom: 4,
    // Author lists of other papers: with this many authors or more, show "FirstAuthor et al."; with fewer, list them all
    // (one author: "A"; two: "A and B"). Set to Infinity to always list everyone.
    etAlFrom: 3,
    etAl: 'et al.',
    venueFields: ['booktitle', 'journal'], // fields tried, in order, for the venue
    // Headshots of the deck author's co-authors, shown in a row under each of the deck author's references.
    // Key: the name exactly as it renders ("First Last", after LaTeX accents are converted); value: image path relative to index.html.
    // A co-author without an entry (or whose file fails to load) is simply left out. Photos are small squares (160x160);
    // where each one comes from is listed in assets/headshots/SOURCES.md. Add or replace photos here.
    headshots: {
      'John Hewitt': 'assets/headshots/john-hewitt.jpg',
      'Alexander Koller': 'assets/headshots/alexander-koller.jpg',
      'Swabha Swayamdipta': 'assets/headshots/swabha-swayamdipta.jpg',
      'Ashish Sabharwal': 'assets/headshots/ashish-sabharwal.jpg',
      'Xiang Ren': 'assets/headshots/xiang-ren.jpg',
      'Murtaza Nazir': 'assets/headshots/murtaza-nazir.jpg',
      'John X. Morris': 'assets/headshots/jack-morris.jpg',
      'Francisco Pernice': 'assets/headshots/francisco-pernice.jpg',
      'Eric Todd': 'assets/headshots/eric-todd.jpg',
      'Amir Zur': 'assets/headshots/amir-zur.jpg',
      'Daniel Wurgaft': 'assets/headshots/daniel-wurgaft.jpg',
      'Fenil R. Doshi': 'assets/headshots/fenil-doshi.jpg',
      'Vasudev Shyam': 'assets/headshots/vasudev-shyam.jpg',
      'Matt Feiszli': 'assets/headshots/matt-feiszli.jpg',
      'Satchel Grant': 'assets/headshots/satchel-grant.jpg',
      'Lucius Bushnaq': 'assets/headshots/lucius-bushnaq.jpg',
      'Tal Haklay': 'assets/headshots/tal-haklay.jpg',
      'Usha Bhalla': 'assets/headshots/usha-bhalla.jpg',
      'Matthew Kowal': 'assets/headshots/matthew-kowal.jpg',
      'Thomas Fel': 'assets/headshots/thomas-fel.jpg',
      'Jack Merullo': 'assets/headshots/jack-merullo.jpg',
      'Atticus Geiger': 'assets/headshots/atticus-geiger.jpg',
      // 'Owen Lewis': no trustworthy photo found yet (see SOURCES.md); add his file here when you have one.
      'Ekdeep Singh Lubana': 'assets/headshots/ekdeep-singh-lubana.jpg'
    }
  };

  // ---- BibTeX parsing ----

  // LaTeX accent commands -> Unicode combining marks (applied to the following letter, then normalized to one character).
  const ACCENTS = { '`': '\u0300', "'": '\u0301', '^': '\u0302', '~': '\u0303', '=': '\u0304', '.': '\u0307',
    '"': '\u0308', 'H': '\u030B', 'v': '\u030C', 'u': '\u0306', 'c': '\u0327', 'k': '\u0328' };
  // LaTeX commands that stand for a single character.
  const SYMBOLS = { ss: 'ß', o: 'ø', O: 'Ø', ae: 'æ', AE: 'Æ', oe: 'œ', OE: 'Œ', aa: 'å', AA: 'Å', l: 'ł', L: 'Ł', i: 'ı', j: 'ȷ' };

  // Turn a field value into plain text: accents to Unicode, escapes to characters, grouping braces removed.
  function latexToText(s) {
    return s
      // {\`e}, \`{e}, \`e, \c{c}, \v c ... -> letter + combining mark
      .replace(/\{?\\([`'^~=."Hvuck])\s*\{?\s*(\\[ij]|[A-Za-z])\s*\}?\}?/g, (_, cmd, letter) => {
        const base = letter === '\\i' ? 'i' : letter === '\\j' ? 'j' : letter;
        return (base + ACCENTS[cmd]).normalize('NFC');
      })
      .replace(/\{?\\(ss|ae|AE|oe|OE|aa|AA|o|O|l|L|i|j)\b\s*\}?/g, (_, cmd) => SYMBOLS[cmd])
      .replace(/\\([&%$#_{}])/g, '$1')    // \& \% ... -> & % ...
      .replace(/---/g, '\u2014').replace(/--/g, '\u2013')
      .replace(/~/g, '\u00A0')            // ~ is a non-breaking space
      .replace(/[{}]/g, '')               // grouping braces, e.g. {GPT}
      .replace(/\s+/g, ' ')
      .trim();
  }

  // Split s on `separator` (a regex matching at the current position) only outside braces.
  function splitTopLevel(s, separator) {
    const parts = [];
    let depth = 0, start = 0;
    for (let i = 0; i < s.length; i++) {
      if (s[i] === '{') depth++;
      else if (s[i] === '}') depth--;
      else if (depth === 0) {
        const m = separator.exec(s.slice(i));
        if (m && m.index === 0) { parts.push(s.slice(start, i)); i += m[0].length - 1; start = i + 1; }
      }
    }
    parts.push(s.slice(start));
    return parts.map(p => p.trim()).filter(Boolean);
  }

  // One BibTeX name -> "First Last". Handles "Last, First", "Last, Jr., First" and "First Last".
  function displayName(raw) {
    const parts = splitTopLevel(raw, /^,/);
    let name = raw;
    if (parts.length === 2) name = parts[1] + ' ' + parts[0];
    else if (parts.length >= 3) name = parts[2] + ' ' + parts[0] + ', ' + parts[1];
    return latexToText(name);
  }

  // Parse the whole file into { key: { type, fields: { name: rawValue } } }.
  function parseBib(text) {
    const entries = {};
    let i = 0;
    const skipSpace = () => { while (i < text.length && /\s/.test(text[i])) i++; };
    // Read a {braced} or "quoted" value (nested braces allowed) or a bare word/number, starting at i.
    function readValue() {
      skipSpace();
      if (text[i] === '{' || text[i] === '"') {
        const close = text[i] === '{' ? '}' : '"';
        let depth = 0, start = ++i;
        for (; i < text.length; i++) {
          if (text[i] === '\\') { i++; continue; }
          if (text[i] === '{') depth++;
          else if (text[i] === '}' && depth > 0) depth--;
          else if (text[i] === close && depth === 0) break;
        }
        return text.slice(start, i++);
      }
      const m = /^[^\s,}#]+/.exec(text.slice(i));
      i += m ? m[0].length : 0;
      return m ? m[0] : '';
    }
    while ((i = text.indexOf('@', i)) !== -1) {
      const head = /^@(\w+)\s*[{(]/.exec(text.slice(i));
      if (!head) { i++; continue; }
      const type = head[1].toLowerCase();
      i += head[0].length;
      if (type === 'comment' || type === 'string' || type === 'preamble') { readValueUntilClose(); continue; }
      const keyMatch = /^\s*([^\s,]+)\s*,/.exec(text.slice(i));
      if (!keyMatch) continue;
      i += keyMatch[0].length;
      const fields = {};
      for (;;) {
        skipSpace();
        if (text[i] === '}' || text[i] === ')' || i >= text.length) { i++; break; }
        const name = /^([\w-]+)\s*=/.exec(text.slice(i));
        if (!name) { i++; continue; }
        i += name[0].length;
        let value = readValue();
        skipSpace();
        while (text[i] === '#') { i++; value += readValue(); skipSpace(); }   // "a" # "b" concatenation
        fields[name[1].toLowerCase()] = value;
        if (text[i] === ',') i++;
      }
      entries[keyMatch[1]] = { type, fields };
    }
    return entries;

    // Skip the body of an @comment/@string/@preamble block (balanced braces).
    function readValueUntilClose() {
      let depth = 1;
      for (; i < text.length && depth > 0; i++) {
        if (text[i] === '{' || text[i] === '(') depth++;
        else if (text[i] === '}' || text[i] === ')') depth--;
      }
    }
  }

  // ---- Formatting ----

  // ["A"] -> "A"; ["A", "B"] -> "A and B"; ["A", "B", "C"] -> "A, B, and C".
  function joinNames(names) {
    if (names.length <= 2) return names.join(' and ');
    return names.slice(0, -1).join(', ') + ', and ' + names[names.length - 1];
  }

  // A BibTeX author field -> ["First Last", ...].
  function authorNames(rawAuthors) {
    return splitTopLevel(rawAuthors || '', /^\s+and\s+/).map(displayName);
  }

  // The author rule. Deck author's papers: everyone, or up to the deck author + "et al." from CONFIG.ownEtAlFrom authors.
  // Other papers: everyone, or "First et al." from CONFIG.etAlFrom authors.
  function formatAuthors(names) {
    const own = names.indexOf(CONFIG.deckAuthor);
    if (own >= 0) {
      if (names.length < CONFIG.ownEtAlFrom || own === names.length - 1) return joinNames(names);
      return own === 0 ? names[0] + ' ' + CONFIG.etAl : names.slice(0, own + 1).join(', ') + ', ' + CONFIG.etAl;
    }
    if (names.length < CONFIG.etAlFrom) return joinNames(names);
    return names[0] + ' ' + CONFIG.etAl;
  }

  // Append the headshots of the deck author's co-authors (everyone, including those hidden behind "et al.") to `el`,
  // as <span class="headshots"><img alt="Name" title="Name">...</span>, only for the deck author's papers with at least
  // one co-author in CONFIG.headshots. CSS lays the span out as its own row under the reference text.
  function appendHeadshots(el, names) {
    if (!names.includes(CONFIG.deckAuthor)) return;
    const images = names.filter(name => name !== CONFIG.deckAuthor && CONFIG.headshots[name]).map(name => {
      const img = document.createElement('img');
      img.src = CONFIG.headshots[name];
      img.alt = name;
      img.title = name;
      img.decoding = 'async';
      img.addEventListener('error', () => {
        console.warn(`references.js: could not load the headshot of ${name} (${CONFIG.headshots[name]})`);
        img.remove();
      });
      return img;
    });
    if (!images.length) return;
    const row = document.createElement('span');
    row.className = 'headshots';
    row.append(...images);
    el.append(row);
  }

  // Append one reference to `el`: <a>Title</a>. Authors. <em>Venue</em>. Year. [headshots]
  function appendReference(el, entry) {
    const f = entry.fields;
    const names = authorNames(f.author);
    const title = latexToText(f.title || '');
    if (f.url) {
      const a = document.createElement('a');
      a.href = latexToText(f.url);
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      a.textContent = title;
      el.append(a);
    } else {
      el.append(title);
    }
    const authors = formatAuthors(names);
    el.append('. ' + authors + (authors.endsWith('.') ? ' ' : '. '));   // "et al." already ends with a period
    const venueField = CONFIG.venueFields.find(name => f[name]);
    if (venueField) {
      const em = document.createElement('em');
      em.textContent = latexToText(f[venueField]);
      el.append(em, '. ');
    }
    el.append(latexToText(f.year || '') + '.');
    appendHeadshots(el, names);
  }

  // Fill every [data-cite] element with one <span class="reference"> per key.
  function render(entries) {
    document.querySelectorAll('[data-cite]').forEach(el => {
      el.replaceChildren();
      el.dataset.cite.trim().split(/\s+/).forEach(key => {
        const span = document.createElement('span');
        span.className = 'reference';
        if (entries[key]) appendReference(span, entries[key]);
        else {
          console.warn(`references.js: no entry "${key}" in ${CONFIG.bibFile}`);
          span.append(key);
        }
        el.append(span);
      });
    });
  }

  fetch(CONFIG.bibFile)
    .then(response => {
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return response.text();
    })
    .then(text => render(parseBib(text)))
    .catch(error => {
      console.warn(`references.js: could not load ${CONFIG.bibFile} (${error.message})`);
      render({});
    });
})();
