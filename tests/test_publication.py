import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image
import pytest
from manifold_attractors.pipeline import ROOT, pixel_digest, sha256


def publisher():
    spec = importlib.util.spec_from_file_location('publish_results', ROOT/'scripts/publish_results.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publication_compares_pixels_not_png_compression(tmp_path):
    module = publisher()
    image = Image.new('RGB', (8, 8), 'white')
    png = tmp_path/'a.png'; image.save(png, compress_level=0)
    ref = {'pixel_sha256': {'iclr': pixel_digest(image)}}
    assert module.compare(ref, None, png, 'iclr') == 'exact-pixels'
    image.putpixel((0,0), (0,0,0)); image.save(png)
    assert module.compare(ref, None, png, 'iclr') == 'different-pixels'
    assert module.compare({}, None, png, 'iclr') == 'no-historical-reference'


def test_table_publication_requires_records_and_embeds_fresh_markdown(tmp_path):
    module=publisher(); root=tmp_path/'repo'; run=tmp_path/'run'
    (root/'configs').mkdir(parents=True); (run/'tab').mkdir(parents=True)
    (run/'runs/shared').mkdir(parents=True); (root/'README.md').write_text('Paper')
    table=run/'tab/example.tex';table.write_text('new numeric table')
    (run/'tab/example.md').write_text('| AUC |\n| --- |\n| 0.875 |\n')
    (run/'tab/example.json').write_text('{"auc": 0.875}')
    (root/'configs/reference_tables.json').write_text(json.dumps({'tables':[
        {'name':'example','tex_sha256':sha256(table)}]}))
    module.publish_tables(run,root)
    assert not (root/'docs/tables/example.tex').exists()
    files=list((run/'tab').glob('example.*'))
    (run/'runs/shared/render-table-example.json').write_text(json.dumps({
        'outputs':{str(p.relative_to(run)):sha256(p) for p in files}}))
    module.publish_tables(run,root,require_all=True)
    assert '0.875' in (root/'README.md').read_text()
    assert json.loads((root/'docs/tables/validation.json').read_text())['tables'][0]['tex_matches_reference']


def test_publication_rejects_unverified_files_before_touching_readme(tmp_path, monkeypatch):
    module = publisher(); root = tmp_path/'repo'; run = tmp_path/'run'
    (root/'configs').mkdir(parents=True)
    (root/'configs/reference.json').write_text(json.dumps({
        'figures': [{'name': 'example'}], 'inline_artworks': []}))
    (root/'README.md').write_text('untouched')
    (run/'runs/iclr').mkdir(parents=True); (run/'fig/iclr').mkdir(parents=True)
    (run/'runs/iclr/render-example.json').write_text(json.dumps({'outputs': {}}))
    (run/'fig/iclr/example.pdf').write_bytes(b'not a real PDF')
    Image.new('RGB', (1,1)).save(run/'fig/iclr/example.png')
    monkeypatch.setattr(module, 'ROOT', root)
    monkeypatch.setattr(sys, 'argv', ['publish_results.py', '--run', str(run)])
    with pytest.raises(RuntimeError, match='unrecorded/changed'):
        module.main()
    assert (root/'README.md').read_text() == 'untouched'
    assert not (root/'docs').exists()
