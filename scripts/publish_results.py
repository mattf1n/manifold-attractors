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


def publish_tables(run, root, require_all=False):
    reference_path = root/'configs/reference_tables.json'
    if not reference_path.exists():
        return
    references = json.loads(reference_path.read_text())['tables']
    ready, pending = [], []
    for item in references:
        name = item['name']; stamp = run/'runs/shared'/f'render-table-{name}.json'
        paths = [run/'tab'/f'{name}{suffix}' for suffix in ('.tex','.md','.json')]
        if not stamp.exists() or not all(p.is_file() for p in paths):
            pending.append(name); continue
        record = json.loads(stamp.read_text())
        if any(record['outputs'].get(str(p.relative_to(run))) != sha256(p) for p in paths):
            raise RuntimeError(f'Refusing changed/unrecorded table: {name}')
        ready.append((item, paths, record))
    if require_all and pending:
        raise RuntimeError('Tables pending: '+', '.join(pending))
    destination = root/'docs/tables'; destination.mkdir(parents=True,exist_ok=True)
    reports, section = [], ['<!-- reproduced-tables:start -->','## Reproduced tables','',
        'These values come from fresh experiments, not the old paper tables. '
        '[Protocols](TABLES.md). [Validation](docs/tables/validation.json).','']
    for item, paths, record in ready:
        name = item['name']
        for path in paths: shutil.copyfile(path,destination/path.name)
        exact = sha256(paths[0]) == item['tex_sha256']
        evidence_path = run/'runs'/f'table-{name}-validation.json'
        evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else None
        reports.append({'name':name,'tex_matches_reference':exact,
                        'numerical_validation':evidence,
                        'source_digest':record.get('source_digest'),
                        'generation_records':record.get('generation_records',{}),
                        'output_sha256':{p.name:sha256(p) for p in paths}})
        if exact:
            assessment = 'Matches the historical formatted table exactly.'
        elif evidence and evidence.get('status') == 'accepted':
            delta = evidence.get('maximum_displayed_cell_difference_percentage_points')
            assessment = 'Scientifically reproduced; documented differences are inconsequential.'
            if delta is not None:
                assessment += f' Maximum cell difference: {delta:.4f} percentage points.'
        else:
            assessment = 'Fresh results; historical formatted-table comparison differs and requires assessment.'
        section += [f'### {name.replace("_"," ").capitalize()}','',
            assessment+f' [LaTeX](docs/tables/{name}.tex) · [Numeric JSON](docs/tables/{name}.json)','',
            paths[1].read_text().strip(),'']
    if pending:section += ['Pending: '+', '.join(f'`{name}`' for name in pending)+'.','']
    section += ['<!-- reproduced-tables:end -->']
    (destination/'validation.json').write_text(json.dumps({'tables':reports,'pending':pending},indent=2)+'\n')
    readme=root/'README.md';text=readme.read_text();start,end=section[0],section[-1];block='\n'.join(section)
    if start in text:
        before,remainder=text.split(start,1);_,after=remainder.split(end,1);text=before+block+after
    else:text=text.rstrip()+'\n\n'+block+'\n'
    readme.write_text(text)
    print(f'Published {len(ready)}/{len(references)} tables')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--require-all', action='store_true')
    args = parser.parse_args(); run = args.run.resolve()
    reference = json.loads((ROOT/'configs/reference.json').read_text())
    entries = {e['name']: e for e in reference['figures']}
    entries.update({e['name']: e for e in reference['inline_artworks']})
    validation_path = run/'runs/scientific-validation.json'
    scientific = json.loads(validation_path.read_text()) if validation_path.exists() else {}
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
                'scientific_validation': scientific.get(name, {'status': 'pending'}),
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
        'scope': 'Fresh model experiments; scientific fidelity required; inconsequential numerical/typographic differences accepted',
        'artworks': records, 'pending': pending}
    directory = ROOT/'docs/figures'; directory.mkdir(parents=True, exist_ok=True)
    (directory/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    exact = sum(item['comparison'].startswith('exact-') for item in records)
    lines = [START, '## Reproduced figures', '',
        'These are **newly generated results**, not copies of the paper figures. '
        'Numerical experiments use the standalone package; typography uses explicitly reconstructed styles.', '',
        f'Published **{len(records)}/36** style-specific artworks; **{exact}** pass the optional exact image check. '
        '**Pixel identity is not required**: small hardware-rounding and typographic differences are acceptable '
        'when scientifically inconsequential. Token selection, numerical results and conclusions must remain faithful. '
        'The per-artwork report distinguishes scientific validation from pixel identity; inline artworks lack historical image references.', '',
        '[Per-artwork hashes and validation](docs/figures/validation.json). '
        '[Protocols and limitations](AUDIT.md).', '']
    labels = {'exact-pixels':'Exact reference pixels',
        'exact-image-and-placement':'Exact embedded image and placement',
        'different-pixels':'Reference pixels differ',
        'different-image-or-placement':'Reference image/placement differs',
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
            science = item['scientific_validation']['status'].replace('-', ' ')
            lines += [f'### {title}', '', f'Scientific validation: **{science}**. '
                      f'{labels[item["comparison"]]}. [PDF]({path}.pdf)', '',
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
    publish_tables(run, ROOT, args.require_all)


if __name__ == '__main__':
    main()
