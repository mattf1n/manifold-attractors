"""CPU-only classifier table tests: no downloads or archived arrays."""
import argparse
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from manifold_attractors.core.inputs import DATASET_SHA256, MODELS, digest
from manifold_attractors.experiments import norm_growth_classifier as classifier
from manifold_attractors.experiments.great_circle_synthetic_grid import synthetic_states


def test_auc_direction_and_ties():
    assert classifier.auc(np.array([[2, 2], [2, 2], [1, 1]])) == [0.5, 1.0]
    assert classifier.auc(np.array([[0, 0], [1, 1], [2, 2]])) == [0.0, 0.0]
    assert [classifier.model_seed(k) for k in MODELS] == [20261910, 20262910, 20263910]


def test_config_exports_all_models_and_readonly_dependencies():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'configs/classifier_table.json').read_text())
    core = json.loads((root / 'configs/core.json').read_text())
    ids = {t['id'] for t in core['tasks']}
    assert len(config['tasks']) == 3
    for key, task in zip(MODELS, config['tasks']):
        assert task['id'] == f'data-norm_growth_classifier-{key}'
        assert task['deps'] == [f'data-native_layers-{key}']
        assert task['deps'][0] in ids
        assert task['outputs'] == [f'data/norm_growth_classifier_scores/{key}.{ext}' for ext in ('npz', 'json')]
    assert config['protocol']['renderer_default_layers'] == np.rint(np.linspace(1, 36, 15)).astype(int).tolist()


def test_original_gaussian_and_shuffled_rng_math():
    pool = np.random.default_rng(42).normal(size=(256, 8)).astype(np.float32)
    for key in MODELS:
        seed = classifier.model_seed(key) + 1
        generated, construction, _ = synthetic_states(pool, 128, seed)
        rng = np.random.default_rng(seed)
        mean = pool.astype(np.float64).mean(0)
        factor = (pool.astype(np.float64) - mean) / np.sqrt(255)
        coefficients = rng.standard_normal((128, 256))
        permutations = np.stack([rng.permutation(256) for _ in range(8)], axis=1)
        np.testing.assert_array_equal(generated[1], (mean + coefficients @ factor).astype(np.float32))
        np.testing.assert_array_equal(generated[2], np.take_along_axis(pool, permutations, axis=0)[:128])
        np.testing.assert_array_equal(construction['gaussian_coefficients'], coefficients)
        np.testing.assert_array_equal(construction['shuffled_permutations'], permutations)


def make_native_fixture(root, key, width=8):
    """Fresh synthetic test fixture with the exact prepare_native file contract."""
    native = root / 'norm_growth_classifier_layers'
    native.mkdir(exist_ok=True)
    ref_path = root / 'great_circle_synthetic_grid' / f'{key}.npz'
    ref_path.parent.mkdir(exist_ok=True)
    # Only provenance is read from this upstream file; not reference arrays.
    ref_path.write_bytes(b'fresh upstream fixture')
    records = [dict(document_row=i, input_ids=[1, 3, 4], position=1, text_sha256=str(i)) for i in range(384)]
    common = dict(model=MODELS[key], snapshot='tiny-local-test', dataset='test-only',
                  dataset_sha256=DATASET_SHA256, probe_token_id=1)
    ref_info = dict(common, records=records[:256], seed=20260909, max_tokens=256, reference_samples=256)
    ref_path.with_suffix('.json').write_text(json.dumps(ref_info))
    evaluation = dict(common, records=records[256:], seed=classifier.SEED,
                      model_seed=classifier.model_seed(key), samples_per_class=128)
    eval_path = root / 'norm_growth_classifier' / f'{key}.json'
    eval_path.parent.mkdir(exist_ok=True)
    eval_path.write_text(json.dumps(evaluation))
    count = classifier.LAYER_COUNTS[key]
    info = dict(evaluation, reference_records=ref_info['records'], reference_samples=256,
                reference_path=str(ref_path), reference_sha256=digest(ref_path), dtype='float32',
                prefix_full_sequence_max_abs_error_by_layer=[0.] * count)
    (native / f'{key}.json').write_text(json.dumps(info))
    rng = np.random.default_rng(14)
    reference = rng.normal(size=(count, 256, width)).astype(np.float32)
    heldout = rng.normal(size=(count, 128, width)).astype(np.float32)
    # Identical pools at two layers make a layer-offset RNG bug observable.
    reference[1] = reference[0]
    np.savez(native / f'{key}.npz', layers=np.arange(1, count + 1), reference=reference, evaluation=heldout)
    return native


def test_input_protection_and_metadata_validation(tmp_path):
    native = make_native_fixture(tmp_path, 'llama3')
    for output in (native, tmp_path / 'great_circle_synthetic_grid', tmp_path / 'norm_growth_classifier'):
        with pytest.raises(ValueError, match='overwrite'):
            classifier.load_pools('llama3', tmp_path, native, output)
    alias = tmp_path / 'alias'
    alias.symlink_to(native, target_is_directory=True)
    with pytest.raises(ValueError, match='overwrite'):
        classifier.load_pools('llama3', tmp_path, native, alias)
    output = tmp_path / 'scores'
    layers, _, _, _, _ = classifier.load_pools('llama3', tmp_path, native, output)
    assert len(layers) == 32
    path = native / 'llama3.json'
    info = json.loads(path.read_text())
    info['model_seed'] += 1
    path.write_text(json.dumps(info))
    with pytest.raises(AssertionError):
        classifier.load_pools('llama3', tmp_path, native, output)


HAS_TORCH = importlib.util.find_spec('torch') is not None
HAS_HF = importlib.util.find_spec('transformers') is not None


@pytest.mark.skipif(not HAS_TORCH, reason='torch not installed')
def test_score_is_raw_block_ratio_and_batch_equivalence():
    import torch
    states = np.random.default_rng(9).normal(size=(3, 7, 8)).astype(np.float32)
    base = torch.ones(1, 2, 8)
    def forward(x):
        # Depend on prefix and injected state; no final normalization.
        return 3 * x + x[:, :1]
    expected = np.linalg.norm(3 * states + 1, axis=-1) / np.linalg.norm(states, axis=-1)
    for batch in (1, 4, 128):
        scores, tests = classifier.score_states(forward, base, states, batch)
        np.testing.assert_allclose(scores, expected, rtol=2e-7)
        assert tests['batch_elementwise_allclose']
        assert 'independent_unbatched_replay' not in tests


@pytest.mark.skipif(not HAS_TORCH, reason='torch not installed')
def test_cancellation_diagnostic_replays_every_vector():
    import torch
    states = np.ones((3, 4, 8), np.float32)
    def forward(x):
        result = torch.ones_like(x) * 1000
        result[..., 0] = 0.001 if len(x) > 1 else 0
        return result
    scores, tests = classifier.score_states(forward, torch.ones(1, 2, 8), states, 3)
    assert not tests['batch_elementwise_allclose']
    replay = tests['independent_unbatched_replay']
    assert [d['vectors'] for d in replay['classes']] == [4, 4, 4]
    assert replay['auc_difference'] == [0, 0]
    assert classifier.auc(scores) == [0.5, 0.5]


@pytest.mark.skipif(not (HAS_TORCH and HAS_HF), reason='compute dependencies not installed')
@pytest.mark.parametrize('key', list(MODELS))
def test_tiny_real_hf_all_layer_cpu_integration(tmp_path, key):
    import torch
    import transformers
    torch.manual_seed(4)
    torch.set_num_threads(1)
    family = {'llama3': 'Llama', 'olmo3': 'Olmo3', 'qwen3': 'Qwen3'}[key]
    config = getattr(transformers, family + 'Config')(
        vocab_size=16, hidden_size=8, intermediate_size=16,
        num_hidden_layers=classifier.LAYER_COUNTS[key], num_attention_heads=2,
        num_key_value_heads=2, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        attn_implementation='eager')
    model = getattr(transformers, family + 'Model')(config).eval()
    class Tokenizer:
        bos_token_id = None if key == 'qwen3' else 1
        eos_token_id = 1 if key == 'qwen3' else 2
        all_special_ids = [1, 2]
    native = make_native_fixture(tmp_path, key)
    hashes = {p: digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    args = argparse.Namespace(model=key, data_root=tmp_path, native_layers=None,
                              output=None, batch=128, threads=1, device='cpu')
    with patch.object(transformers.AutoModel, 'from_pretrained', return_value=model) as loader, \
         patch.object(transformers.AutoTokenizer, 'from_pretrained', return_value=Tokenizer()):
        classifier.compute(args)
    assert loader.call_args.kwargs['attn_implementation'] == 'eager'
    assert loader.call_args.kwargs['dtype'] == torch.float32
    assert all(digest(p) == old_hash for p, old_hash in hashes.items())
    output = tmp_path / 'norm_growth_classifier_scores' / f'{key}.npz'
    info = json.loads(output.with_suffix('.json').read_text())
    assert info['native_pools']['sha256'] == digest(native / f'{key}.npz')
    assert info['tests']['finite_positive_scores']
    assert max(info['tests']['direct_block_max_abs_error_by_layer']) < 1e-5
    with np.load(output) as data, np.load(native / f'{key}.npz') as inputs:
        assert data['scores'].shape == (classifier.LAYER_COUNTS[key], 3, 128)
        assert data['classes'].tolist() == classifier.CLASSES
        assert data['scores'].dtype == np.float32
        assert np.isfinite(data['scores']).all() and (data['scores'] > 0).all()
        np.testing.assert_array_equal(data['states'][:, 0], inputs['evaluation'])
        np.testing.assert_array_equal(data['states'][0, 1:], data['states'][1, 1:])
        for li in range(len(data['layers'])):
            generated, construction, _ = synthetic_states(inputs['reference'][li], 128, classifier.model_seed(key) + 1)
            np.testing.assert_array_equal(data['states'][li, 1:], generated[1:3])
            np.testing.assert_array_equal(data['reference_mean'][li], construction['reference_mean'])
            np.testing.assert_array_equal(data['covariance_factor'][li], construction['covariance_factor'])
        # Exact current table renderer's vectorized AUC protocol, no pandas needed.
        scores = data['scores']
        differences = scores[:, 0, None, :, None] - scores[:, 1:, None, :]
        values = ((differences > 0) + 0.5 * (differences == 0)).mean(axis=(-2, -1))
        np.testing.assert_array_equal(values, info['auc_by_layer'])
        # Independently capture and replay a late block with batch=1. This
        # checks actual score values, not just shape or the diagnostic metadata.
        number = classifier.LAYER_COUNTS[key]
        captured = {}
        def capture(module, positional, keyword):
            captured['base'], captured['kwargs'] = positional[0].clone(), keyword.copy()
        with torch.inference_mode():
            handle = model.layers[number - 1].register_forward_pre_hook(capture, with_kwargs=True)
            model(torch.tensor([[1, 1]]), use_cache=False)
            handle.remove()
            for ci in range(3):
                vector = torch.tensor(data['states'][-1, ci, 0])
                injected = captured['base'].clone()
                injected[:, 1] = vector
                out = model.layers[number - 1](injected, **captured['kwargs'])
                out = out[0] if isinstance(out, tuple) else out
                ratio = float(out[0, 1].norm() / vector.norm())
                np.testing.assert_allclose(scores[-1, ci, 0], ratio, rtol=2e-5)
