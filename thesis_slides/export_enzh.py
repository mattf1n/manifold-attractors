"""Export the paper's frozen en-zh steering-speed data (Figure "Speed profiles along the en-zh steering direction",
fig:language_steering_speed) to the slides' assets/enzh-steering.js. No model inference runs here: it reads the outputs of
`python -m manifold_attractors.maps.language_speed compute` in ~/manifold-attractors/data/language_steering_speed
(Llama 3.1 8B, FP64) and keeps, like the paper's figure script (scripts/fig/language_steering_speed.py), English rows
0, 1, 3, 4 and Chinese rows 0-3, each document's tail, the paper's interlinear glosses, and the paper's token glosses.
Usage: ~/manifold-attractors/.venv/bin/python export_enzh.py > assets/enzh-steering.js"""
import json, re, subprocess
from pathlib import Path
import numpy as np

REPO = Path.home() / 'manifold-attractors'
DATA = REPO / 'data' / 'language_steering_speed'
EXAMPLES = {'': (0, 1, 3, 4), '_chinese': (0, 1, 2, 3)}   # as in the paper's figure
LIMIT = 90          # exported angle range (the whole frozen grid); the slide shows CONFIG.limit of it
WORDS = 3           # English document tails keep their last 3 words before the steered token, as the paper's figure does
g = lambda v: float(f'{v:.4g}')
commit = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
# Token glosses in the paper's own interlinear convention where the paper's token-gloss files leave them as bracketed notes:
# 的 is glossed GEN in the paper's interlinear gloss of the Chinese document 357 ("是 生物 的" = COP organism GEN).
GLOSS_OVERRIDES = {'的': 'GEN'}
prov = json.loads((DATA / 'provenance.json').read_text())
run = json.loads((REPO / 'runs/shared/data-language_steering_speed.json').read_text())

tokens, token_index, panels, theta = [], {}, [], None
abbreviations = json.loads((DATA / 'glosses_chinese.json').read_text())['grammatical_abbreviations']
for suffix, rows in EXAMPLES.items():
    z = np.load(DATA / f'llama3{suffix}.npz')
    glosses = json.loads((DATA / f'glosses{suffix}.json').read_text())
    selection = json.loads((DATA / f'selection{suffix}.json').read_text())
    validation = json.loads((DATA / f'validation{suffix}.json').read_text())['examples']
    contexts = {c['index']: c for c in selection['contexts']}
    t = z['theta']; mask = (t >= -LIMIT) & (t <= LIMIT)
    if theta is None: theta = t[mask]
    assert np.array_equal(theta, t[mask])
    for row in rows:
        ids = z['token_ids'][row][mask]
        index = []
        for tid in ids:
            key = str(int(tid))
            if key not in token_index:
                meta = glosses['tokens'][key]
                token_index[key] = len(tokens)
                gloss = meta['gloss'] if meta['script_class'] not in ('Latin', 'Other nonalphabetic', 'Neutral punctuation/symbols') else ''
                gloss = GLOSS_OVERRIDES.get(meta['text'], gloss)
                tokens.append(dict(id=int(tid), text=meta['text'], script=meta['script_class'], gloss=gloss,
                                   grammatical=gloss in abbreviations))
            index.append(token_index[key])
        source = contexts[int(z['source_indices'][row])]
        annotation = glosses.get('interlinear_contexts', {}).get(str(source['index']))
        if annotation:   # Chinese documents: the paper's word-aligned interlinear glosses ("…" first)
            units = [dict(source=u['source'], gloss=u['gloss'], grammatical=bool(u.get('grammatical')))
                     for u in annotation['units']]
        else:            # English documents: "…", then the last WORDS words before the steered token and the token (paper's context_title)
            prefix, token = source['prefix_through_target'], source['target_token']
            assert prefix.endswith(token)
            before = ' '.join(prefix[:-len(token)].split()[-WORDS:])
            units = [dict(source='…', gloss='', grammatical=False),
                     dict(source=before + (' ' if prefix[:-len(token)][-1:].isspace() else '') + token, gloss='', grammatical=False)]
        # translation: the paper's free translation of the Chinese document's tail (its interlinear annotation; '' for English documents).
        panels.append(dict(language='en' if suffix == '' else 'zh', sourceIndex=source['index'],
                           units=units, translation=annotation['translation'] if annotation else '',
                           tokens=index, speed=[g(v) for v in z['speeds'][row][mask]],
                           validation=validation[row]))
    assert np.allclose(z['step_degrees'], 0.5)

out = dict(model=prov['model'], revision=prov['revision'], precision='FP64', blocks=prov['blocks'],
           theta=dict(min=float(theta[0]), max=float(theta[-1]), step=0.5, count=len(theta)),
           tokens=tokens, panels=panels)
print(f"""// The paper's own frozen data, exported (no new inference) from the USC carc cluster with ~/thesis-slides/export_enzh.py:
// "Language Model Activations Inhabit Privileged Error-Correcting Basins" (Finlayson et al., arXiv 2026), Figure "Speed profiles
// along the en-zh steering direction" (main.tex, fig:language_steering_speed). Computed by the paper's code,
// `python -m manifold_attractors.maps.language_speed compute` (~/manifold-attractors, checkout at commit {commit[:12]};
// run record runs/shared/data-language_steering_speed.json, source digest {run['source_digest'][:16]}, finished 2026-11-20 00:26 UTC),
// output data/language_steering_speed/llama3*.npz, glosses*.json, selection*.json. Model {prov['model']} @ {prov['revision'][:12]}; source states
// captured in FP32 under their documents' causal prefixes, probe (speeds and tokens) in FP64 (FixedProbe).
// Probe: activation x entering block 16 of a token in an English (en) or Chinese (zh) document, rotated on the sphere of radius |x| by angle
// theta toward the en-zh direction (difference of layer-16 en and zh Wikipedia centroids, zh minus en, projected onto the tangent plane at x):
// x(theta) = |x| (cos theta x/|x| + sin theta u). Positive theta steers toward Chinese, negative toward English; 0 is the original activation.
// speed[i] = || d/dtheta f(x(theta)) ||, f = blocks 16-19 (per radian; central difference on the paper's converged grid, here 0.5 deg).
// tokens[i] = greedy next token after all remaining blocks and the final norm, decoded with the paper's ablated (two-BOS) context,
// not the source document. Panels: English rows 0, 1, 3, 4 and Chinese rows 0-3, as in the paper's figure.
// Validation (the paper's code): derivative refinement converged (relative L2 <= {max(p['validation']['relative_l2'] for p in panels):.1e},
// p99 <= {max(p['validation']['relative_p99'] for p in panels):.1e} against the half-step grid), full-hook reproduction error 0, and for Chinese rows
// the same tokens and speeds on a 0.25 deg grid. Units: the paper's document tails, interlinear glosses and free translations (Chinese rows);
// token glosses: the paper's editorial glosses (glosses*.json), kept for non-Latin, non-neutral tokens only; 的 is glossed GEN as in the
// paper's interlinear gloss of the Chinese document 357 (the token files give it only a bracketed note). Speeds to 4 significant digits.
window.ENZH_STEERING = {json.dumps(out, ensure_ascii=False, separators=(',', ':'))};""")
