"""Fresh deterministic outputs checked against historical reference evidence."""
import hashlib
import json
import subprocess
import sys

from PIL import Image
from pypdf import PdfReader

from manifold_attractors.pipeline import ROOT, pixel_digest


def test_pixel_digest_ignores_encoding(tmp_path):
    image = Image.new("RGB", (3, 2), (10, 20, 30))
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    image.save(a, compress_level=0)
    image.save(b, compress_level=9)
    assert a.read_bytes() != b.read_bytes()
    with Image.open(a) as x, Image.open(b) as y:
        assert pixel_digest(x) == pixel_digest(y)


def test_fresh_fish_matches_historical_pdf(tmp_path):
    output = tmp_path / "fish.pdf"
    subprocess.run([sys.executable, "-m", "manifold_attractors.render.dead_fish_sequence",
                    "--figure", str(output)], check=True, cwd=ROOT)
    ref = json.loads((ROOT / "configs/reference.json").read_text())["figures"][0]["shared_pdf_image"]
    reader = PdfReader(output)
    assert len(reader.pages) == 1
    page = reader.pages[0]
    assert len(page.images) == 1
    assert [float(x) for x in page.mediabox] == ref["mediabox"]
    assert hashlib.sha256(page.get_contents().get_data()).hexdigest() == ref["content_sha256"]
    assert pixel_digest(page.images[0].image) == ref["pixel_sha256"]


def test_fresh_field_matches_historical_table(tmp_path):
    output = tmp_path / "field.dat"
    subprocess.run([sys.executable, "-m", "manifold_attractors.render.two_basin_field",
                    "--output", str(output)], check=True, cwd=ROOT)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == "990906b548647b1a8bfba8be80c9ff36fc6afa463df332673030d48979ecab90"
