#!/usr/bin/env python3
"""Publish only newly generated, execution-record-verified figure files to README."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image
from pypdf import PdfReader
from manifold_attractors.pipeline import ROOT, pixel_digest, sha256

START = '<!-- reproduced-figures:start -->'
END = '<!-- reproduced-figures:end -->'


def compare(entry, pdf, png, style):
    if entry.get('shared_pdf_image'):
        ref = entry['shared_pdf_image']; reader = PdfReader(pdf); page = reader.pages[0]
        matched = (len(reader.pages) == 1 and len(page.images) == 1
            and pixel_digest(page.images[0].image) == ref['pixel_sha256']
            and [float(x) for x in page.mediabox] == ref['mediabox']
            and hashlib.sha256(page.get_contents().get_data()).hexdigest() == ref['content_sha256'])
        return 'exact-image-and-placement' if matched else 'different-image-or-placement'
    expected = entry.get('pixel_sha256', {}).get(style)
    if not expected:
        return 'no-historical-reference'
    with Image.open(png) as image:
        return 'exact-pixels' if pixel_digest(image) == expected else 'different-pixels'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--require-all', action='store_true')
    args = parser.parse_args(); run = args.run.resolve()
    reference = json.loads((ROOT/'configs/reference.json').read_text())
    entries = {e['name']: e for e in reference['figures']}
    entries.update({e['name']: e for e in reference['inline_artworks']})
    records, pending = [], []
    for style in ('iclr', 'goodfire'):
        for name, entry in entries.items():
            shared = name == 'dead_fish_sequence'
            profile = 'shared' if shared else style
            source = run/'fig'/profile/name
            stamp = run/'runs'/profile/f'render-{name}.json'
            pdf, png = source.with_suffix('.pdf'), source.with_suffix('.png')
            if not stamp.exists() or not pdf.exists() or not png.exists():
                pending.append(f'{style}/{name}')
                continue
            record = json.loads(stamp.read_text())
            for path in (pdf, png):
                if record['outputs'].get(str(path.relative_to(run))) != sha256(path):
                    raise RuntimeError(f'Refusing unrecorded/changed output: {path}')
            status = compare(entry, pdf, png, style)
            records.append({'name': name, 'style': style, 'comparison': status,
                'source_digest': record.get('source_digest'),
                'generation_records': record.get('generation_records', {}),
                'pdf_sha256': sha256(pdf), 'png_sha256': sha256(png),
                'record': str(stamp.relative_to(run))})
    if args.require_all and pending:
        raise RuntimeError('Still missing: '+', '.join(pending))
    # Validate the whole candidate set before changing the publication directory.
    for item in records:
        name, style = item['name'], item['style']
        source = run/'fig'/('shared' if name == 'dead_fish_sequence' else style)/name
        dest = ROOT/'docs/figures'/style/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ('.png', '.pdf'):
            shutil.copyfile(source.with_suffix(suffix), dest.with_suffix(suffix))
    report = {'paper_commit': reference['paper_commit'],
        'run_source_commit': json.loads((run/'runs/launch.json').read_text())['source_commit'],
        'scope': 'Fresh model experiments; reconstructed typography; not a blanket exact-reproduction claim',
        'artworks': records, 'pending': pending}
    directory = ROOT/'docs/figures'; directory.mkdir(parents=True, exist_ok=True)
    (directory/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    exact = sum(item['comparison'].startswith('exact-') for item in records)
    lines = [START, '## Reproduced figures', '',
        'These are **newly generated results**, not copies of the paper figures. '
        'Numerical experiments use the standalone package; typography uses explicitly reconstructed styles.', '',
        f'Published **{len(records)}/36** style-specific artworks; **{exact}** pass their available exact image checks. '
        '“Different pixels” means the strict comparison failed—not a verified exact reproduction. '
        'The inline artworks have no supplied historical reference.', '',
        '[Per-artwork hashes and validation](docs/figures/validation.json). '
        '[Protocols and limitations](AUDIT.md).', '']
    labels = {'exact-pixels':'Exact reference pixels',
        'exact-image-and-placement':'Exact embedded image and placement',
        'different-pixels':'Different pixels — not exact',
        'different-image-or-placement':'Different image/placement — not exact',
        'no-historical-reference':'No historical reference — unverified'}
    order = ['intro', 'sphere_probe'] + [n for n in entries if n not in {'intro','sphere_probe','dead_fish_sequence'}]
    by_key = {(r['style'],r['name']):r for r in records}
    for style, title in [('iclr','ICLR'),('goodfire','Goodfire')]:
        lines += [f'<details {"open" if style == "iclr" else ""}>', f'<summary>{title} figures</summary>', '']
        for name in order:
            item = by_key.get((style,name))
            if item is None:
                continue
            title = name.replace('_',' ').capitalize()
            path = f'docs/figures/{style}/{name}'
            lines += [f'### {title}', '', f'{labels[item["comparison"]]}. [PDF]({path}.pdf)', '',
                      f'![{title}]({path}.png)', '']
        fish = by_key.get((style,'dead_fish_sequence'))
        if fish:
            lines += [f'Fish component: [{labels[fish["comparison"]]}](docs/figures/{style}/dead_fish_sequence.pdf).', '']
        lines += ['</details>', '']
    if pending:
        lines += ['Still pending: '+', '.join(f'`{name}`' for name in pending)+'.', '']
    lines += [END]
    readme = ROOT/'README.md'; text = readme.read_text(); section = '\n'.join(lines)
    if START in text:
        before, remainder = text.split(START,1)
        _, after = remainder.split(END,1)
        text = before+section+after
    else:
        text = text.rstrip()+'\n\n'+section+'\n'
    readme.write_text(text)
    print(f'Published {len(records)}/36 artifacts; {exact} exact image checks; {len(pending)} pending')


if __name__ == '__main__':
    main()
