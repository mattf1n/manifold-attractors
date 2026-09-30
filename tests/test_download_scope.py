"""Do not request unused Meta-format copies alongside HF checkpoints."""
import ast
from pathlib import Path
from unittest.mock import patch

from manifold_attractors.maps import inputs, refusal_direction


def test_map_snapshot_requests_transformers_files_only():
    with patch('huggingface_hub.snapshot_download', return_value='snapshot') as download:
        assert inputs.snapshot('refusal', refusal_direction.REVISION) == 'snapshot'
    options = download.call_args.kwargs
    assert options['revision'] == refusal_direction.REVISION
    assert '*.safetensors' in options['allow_patterns']
    assert '*.json' in options['allow_patterns']
    assert options['ignore_patterns'] == ['original/*']


def test_historical_extractor_has_same_download_scope():
    # Its historical dependencies run in a separate interpreter. Check the call
    # without importing that old stack into the modern compute environment.
    tree = ast.parse(Path(refusal_direction.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'snapshot_download']
    assert len(calls) == 1
    options = {arg.arg: arg.value for arg in calls[0].keywords}
    assert ast.literal_eval(options['ignore_patterns']) == ['original/*']
    assert '*.safetensors' in ast.literal_eval(options['allow_patterns'])
