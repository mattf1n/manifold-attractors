#!/usr/bin/env python3
"""Render hash-verified fresh experiment outputs without rerunning computation.

Run from an editable checkout. Missing experiment stages are reported as pending;
--require-all fails unless every artwork rendered. Documented reconstructed
styles are allowed for scientific reproduction; strict provenance is optional.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from manifold_attractors.pipeline import ROOT, load_tasks, order_tasks, outputs, command, sha256, code_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--style', choices=['iclr', 'goodfire'], default='iclr')
    parser.add_argument('--scope', choices=['figures','tables','all'], default='figures')
    typography = parser.add_mutually_exclusive_group()
    typography.add_argument('--allow-reconstructed-styles', action='store_true', default=True)
    typography.add_argument('--require-original-styles', action='store_false', dest='allow_reconstructed_styles')
    parser.add_argument('--require-all', action='store_true')
    args = parser.parse_args()
    reference = json.loads((ROOT/'configs/reference.json').read_text())
    if args.scope != 'tables' and any(not item.get('resolved') for item in reference.get('exactness_blockers', [])) and not args.allow_reconstructed_styles:
        parser.error('Original styles are unavailable; --allow-reconstructed-styles explicitly acknowledges unverified typography')
    if args.scope != 'tables' and (not shutil.which('lualatex') or not shutil.which('pdftocairo')):
        parser.error('Activate TeX and the pinned rasterizer first; see RENDER.md')
    tasks = load_tasks()
    wanted = [name for name in tasks if name.startswith('render-') and
              (args.scope == 'all' or (name.startswith('render-table-') == (args.scope == 'tables')))]
    order = order_tasks(tasks, wanted)
    verified, unavailable = {}, {}
    for name in order:
        if name.startswith('render-'):
            continue
        stamp = ROOT/'runs/shared'/f'{name}.json'
        if not stamp.exists():
            unavailable[name] = 'no successful execution record'
            continue
        record = json.loads(stamp.read_text())
        for path in outputs(tasks[name], args.style):
            if not path.is_file() or record['outputs'].get(str(path.relative_to(ROOT))) != sha256(path):
                raise RuntimeError(f'Generated output changed or missing: {path}')
        verified[name] = record
    env = dict(os.environ, FIGURE_STYLE=args.style, PYTHONHASHSEED='0', PYTHONPATH=str(ROOT/'src'))
    directory = ROOT/'runs'/('tables' if args.scope == 'tables' else args.style)
    directory.mkdir(parents=True, exist_ok=True)
    report = {'style': args.style, 'reconstructed_styles': args.allow_reconstructed_styles,
              'source_digest': code_digest(ROOT), 'runner_sha256': sha256(Path(__file__)),
              'tex_version': subprocess.check_output(['lualatex', '--version'], text=True) if args.scope != 'tables' else None,
              'rendered': [], 'pending': {}, 'failed': {}}
    for name in [n for n in order if n.startswith('render-')]:
        missing = [n for n in order_tasks(tasks, [name]) if n in unavailable]
        if missing:
            report['pending'][name] = missing
            print('pending:', name, flush=True)
            continue
        if name in {'render-dead_fish_sequence', 'render-two_basin_field'}:
            shared_stamp = ROOT/'runs/shared'/f'{name}.json'
            if shared_stamp.exists():
                shared = json.loads(shared_stamp.read_text())
                if all(p.is_file() and shared['outputs'].get(str(p.relative_to(ROOT))) == sha256(p)
                       for p in outputs(tasks[name], args.style)):
                    report['rendered'].append(name)
                    print('verified shared:', name, flush=True)
                    continue
        cmd = command(tasks[name], args.style)
        start = time.time()
        with (directory/f'{name}.log').open('w') as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            unavailable[name] = f'exit {result.returncode}'
            report['failed'][name] = unavailable[name]
            print('FAILED:', name, flush=True)
            continue
        expected = outputs(tasks[name], args.style)
        if not all(p.is_file() and p.stat().st_size for p in expected):
            raise RuntimeError(f'{name}: missing declared output')
        # Inline TeX produces PDF+source. Generate an actual-artwork README preview.
        if name in {'render-intro', 'render-sphere_probe'}:
            pdf = next(p for p in expected if p.suffix == '.pdf')
            subprocess.run(['pdftocairo', '-png', '-singlefile', '-r', '150', str(pdf), str(pdf.with_suffix(''))], check=True)
            expected.append(pdf.with_suffix('.png'))
        record = {'task': name, 'command': cmd, 'source_digest': report['source_digest'],
                  'seconds': time.time()-start, 'finished': time.time(),
                  'reconstructed_styles': args.allow_reconstructed_styles,
                  'generation_records': {n: verified[n]['fingerprint'] for n in order_tasks(tasks, [name]) if n in verified},
                  'outputs': {str(p.relative_to(ROOT)): sha256(p) for p in expected}}
        record_directory = ROOT/'runs/shared' if name.startswith('render-table-') else directory
        record_directory.mkdir(parents=True, exist_ok=True)
        (record_directory/f'{name}.json').write_text(json.dumps(record, indent=2)+'\n')
        report['rendered'].append(name)
        print('rendered:', name, flush=True)
    (directory/'render-report.json').write_text(json.dumps(report, indent=2)+'\n')
    if report['failed'] or (args.require_all and report['pending']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
