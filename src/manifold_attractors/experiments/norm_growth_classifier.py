"""Fresh all-layer norm-growth table scores from prepare_native's read-only pools.

No archived arrays or historical checkout is a runtime input. The fixed score is
raw block output norm / injected input norm; natural is always positive.
"""
import argparse
import json
import os
from pathlib import Path
import platform

import numpy as np

from ..core.inputs import MODELS, DATASET_SHA256, digest
from .great_circle_norm_grid import SAMPLING_POLICY
from .great_circle_synthetic_grid import synthetic_states
from .pair_barrier_validation import atomic_json, atomic_npz

CLASSES = ['natural', 'gaussian', 'shuffled']
LAYER_COUNTS = {'llama3': 32, 'olmo3': 32, 'qwen3': 36}
SEED = 20260910


def auc(scores):
    """Natural is positive, higher is natural, half credit for exact ties."""
    return [float(((scores[0, :, None] > negative[None, :]) +
                   0.5 * (scores[0, :, None] == negative[None, :])).mean())
            for negative in scores[1:]]


def model_seed(key):
    return SEED + 1000 * (list(MODELS).index(key) + 1)


def load_pools(key, data_root, native_layers, output):
    """Validate the native-stage contract without opening any old score arrays."""
    source = native_layers / f'{key}.npz'
    # Guard the directory even before reading: these inputs are already hashed
    # and shared with other experiments. Also reject symlink/hardlink aliases.
    if output.resolve() in (native_layers.resolve(),
                            (data_root / 'norm_growth_classifier_layers').resolve(),
                            (data_root / 'great_circle_synthetic_grid').resolve(),
                            (data_root / 'norm_growth_classifier').resolve()):
        raise ValueError('Classifier scores must not overwrite native pools or upstream inputs')
    for suffix in ('.npz', '.json'):
        destination = output / f'{key}{suffix}'
        if destination.exists() and destination.samefile(source.with_suffix(suffix)):
            raise ValueError('Classifier output aliases a native input')
    info = json.loads(source.with_suffix('.json').read_text())
    ref_path = data_root / 'great_circle_synthetic_grid' / f'{key}.npz'
    ref_info = json.loads(ref_path.with_suffix('.json').read_text())
    eval_path = data_root / 'norm_growth_classifier' / f'{key}.json'
    eval_info = json.loads(eval_path.read_text())
    assert info['model'] == ref_info['model'] == eval_info['model'] == MODELS[key]
    assert info['snapshot'] == ref_info['snapshot'] == eval_info['snapshot']
    assert info['dataset_sha256'] == ref_info['dataset_sha256'] == eval_info['dataset_sha256'] == DATASET_SHA256
    assert info['probe_token_id'] == ref_info['probe_token_id'] == eval_info['probe_token_id']
    assert info['reference_sha256'] == digest(ref_path)
    assert info['reference_records'] == ref_info['records']
    assert info['records'] == eval_info['records']
    assert info['seed'] == eval_info['seed'] == SEED
    assert info['model_seed'] == eval_info['model_seed'] == model_seed(key)
    assert ref_info['seed'] == 20260909 and ref_info['max_tokens'] == 256
    assert info['reference_samples'] == ref_info['reference_samples'] == 256
    assert info['samples_per_class'] == eval_info['samples_per_class'] == 128
    assert info['dtype'] == 'float32'
    fit, heldout = info['reference_records'], info['records']
    assert len(fit) == 256 and len(heldout) == 128
    fit_rows = {r['document_row'] for r in fit}
    test_rows = {r['document_row'] for r in heldout}
    assert len(fit_rows) == 256 and len(test_rows) == 128 and fit_rows.isdisjoint(test_rows)
    assert all(0 < r['position'] < len(r['input_ids']) for r in fit + heldout)
    with np.load(source, allow_pickle=False) as data:
        layers, reference, evaluation = (data[k].copy() for k in ('layers', 'reference', 'evaluation'))
    np.testing.assert_array_equal(layers, np.arange(1, LAYER_COUNTS[key] + 1))
    assert reference.ndim == evaluation.ndim == 3
    assert reference.shape[:2] == (len(layers), 256)
    assert evaluation.shape == (len(layers), 128, reference.shape[-1])
    for pool in (reference, evaluation):
        assert pool.dtype == np.float32 and np.isfinite(pool).all()
        assert (np.linalg.norm(pool, axis=-1) > 0).all()
    provenance = dict(path=str(source.resolve()), sha256=digest(source),
                      metadata_sha256=digest(source.with_suffix('.json')),
                      reference_metadata_sha256=digest(ref_path.with_suffix('.json')),
                      evaluation_metadata_sha256=digest(eval_path))
    return layers, reference, evaluation, info, provenance


def score_states(forward, base, states, batch):
    """Original batched score and Qwen-safe single/batch validation protocol."""
    import torch
    if batch < 1:
        raise ValueError('batch must be positive')
    samples = states.shape[1]
    scores = np.empty((3, samples), np.float32)
    error = relative_error = score_error = 0.0
    elementwise_close = True
    for ci, vectors_np in enumerate(states):
        for start in range(0, samples, batch):
            vectors = torch.tensor(vectors_np[start:start + batch], device=base.device)
            inputs = base.expand(len(vectors), -1, -1).clone()
            inputs[:, 1] = vectors
            output = forward(inputs)
            single = forward(inputs[:1])
            error = max(error, float((single - output[:1]).abs().max()))
            relative = float(((single - output[:1]).norm(dim=-1) / single.norm(dim=-1)).max())
            score_relative = float(((single[:, 1].norm(dim=-1) - output[:1, 1].norm(dim=-1)).abs() /
                                    single[:, 1].norm(dim=-1)).max())
            relative_error = max(relative_error, relative)
            score_error = max(score_error, score_relative)
            elementwise_close &= bool(torch.allclose(single, output[:1], rtol=2e-4, atol=2e-4))
            assert relative < 2e-5 and score_relative < 2e-5, (ci, relative, score_relative)
            scores[ci, start:start + len(vectors)] = (
                output[:, 1].norm(dim=-1) / vectors.norm(dim=-1)).cpu().numpy()
    tests = dict(batch_max_abs_error=error, batch_max_relative_l2_error=relative_error,
                 batch_max_relative_score_error=score_error, batch_elementwise_allclose=elementwise_close)
    if not elementwise_close:
        # Componentwise cancellation at the unnormalized Qwen final block is
        # diagnostic, not a changed score. Replay every vector independently.
        unbatched = np.empty_like(scores)
        diagnostics = []
        for ci, name in enumerate(CLASSES):
            max_abs = max_component_relative = max_l2_relative = 0.0
            failed_vectors = 0
            for start in range(0, samples, batch):
                vectors = torch.tensor(states[ci, start:start + batch], device=base.device)
                inputs = base.expand(len(vectors), -1, -1).clone()
                inputs[:, 1] = vectors
                batched = forward(inputs)
                for vi in range(len(vectors)):
                    single = forward(inputs[vi:vi + 1])
                    delta = (single - batched[vi:vi + 1]).abs()
                    max_abs = max(max_abs, float(delta.max()))
                    max_component_relative = max(max_component_relative,
                        float((delta / single.abs().clamp_min(1e-12)).max()))
                    max_l2_relative = max(max_l2_relative,
                        float((delta.norm(dim=-1) / single.norm(dim=-1)).max()))
                    failed_vectors += int(not torch.allclose(single, batched[vi:vi + 1], rtol=2e-4, atol=2e-4))
                    unbatched[ci, start + vi] = float(single[0, 1].norm() / vectors[vi].norm())
            delta = np.abs(unbatched[ci] - scores[ci])
            diagnostics.append(dict(name=name, vectors=samples,
                failed_legacy_elementwise_vectors=failed_vectors, max_abs_output_error=max_abs,
                max_component_relative_output_error=max_component_relative,
                component_relative_denominator_floor=1e-12, max_vector_relative_l2_error=max_l2_relative,
                max_abs_score_error=float(delta.max()),
                max_relative_score_error=float((delta / unbatched[ci]).max())))
        auc_delta = np.array(auc(unbatched)) - auc(scores)
        tests['independent_unbatched_replay'] = dict(classes=diagnostics, scores=unbatched.tolist(),
            batched_auc=auc(scores), unbatched_auc=auc(unbatched), auc_difference=auc_delta.tolist(),
            legacy_elementwise_check_passed=False, auc_difference_tolerance=2 / samples**2)
        assert max(d['max_vector_relative_l2_error'] for d in diagnostics) < 2e-5
        assert max(d['max_relative_score_error'] for d in diagnostics) < 2e-5
        assert np.max(np.abs(auc_delta)) <= 2 / samples**2
    assert np.isfinite(scores).all() and (scores > 0).all()
    return scores, tests


def compute(a):
    import torch
    import transformers
    if min(a.batch, a.threads) < 1:
        raise ValueError('batch and threads must be positive')
    native = a.native_layers or a.data_root / 'norm_growth_classifier_layers'
    output = a.output or a.data_root / 'norm_growth_classifier_scores'
    layers, reference, evaluation, info, provenance = load_pools(a.model, a.data_root, native, output)
    torch.set_num_threads(a.threads)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(info['snapshot'])
    prefix_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    assert prefix_id == info['probe_token_id']
    assert all(r['input_ids'][r['position']] not in tokenizer.all_special_ids
               for r in info['reference_records'] + info['records'])
    model = transformers.AutoModel.from_pretrained(info['snapshot'], dtype=torch.float32,
        device_map=a.device, attn_implementation='eager').eval()
    model.requires_grad_(False)
    assert len(model.layers) == len(layers) and model.config.hidden_size == reference.shape[-1]
    bases, keywords, ordinary = {}, {}, {}
    def pre_hook(number):
        def capture(module, positional, keyword):
            bases[number] = positional[0].clone()
            keywords[number] = keyword.copy()
        return capture
    def out_hook(number):
        def capture(module, inputs, out):
            ordinary[number] = (out[0] if isinstance(out, tuple) else out).clone()
        return capture
    def forward(number, inputs):
        out = model.layers[number - 1](inputs, **keywords[number])
        return out[0] if isinstance(out, tuple) else out
    scores = np.empty((len(layers), 3, 128), np.float32)
    states = np.empty((len(layers), 3, 128, reference.shape[-1]), np.float32)
    means, factors, statistics, direct_errors, batch_tests = [], [], [], [], []
    with torch.inference_mode():
        handles = []
        try:
            for number in layers.tolist():
                block = model.layers[number - 1]
                handles.extend([block.register_forward_pre_hook(pre_hook(number), with_kwargs=True),
                                block.register_forward_hook(out_hook(number))])
            model(torch.tensor([[prefix_id, prefix_id]], device=a.device), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        for li, number in enumerate(layers.tolist()):
            direct_error = float((forward(number, bases[number]) - ordinary.pop(number)).abs().max())
            assert direct_error < 1e-5
            direct_errors.append(direct_error)
            # Deliberately RESET the RNG at every layer, with the HELDOUT model
            # seed + 1 (not the fit-pool seed + 1 and not seed + layer).
            generated, construction, fitted = synthetic_states(reference[li], 128, model_seed(a.model) + 1)
            states[li] = np.stack([evaluation[li], generated[1], generated[2]])
            means.append(construction['reference_mean'])
            factors.append(construction['covariance_factor'])
            statistics.append(fitted)
            scores[li], diagnostics = score_states(lambda x: forward(number, x), bases[number], states[li], a.batch)
            batch_tests.append(diagnostics)
            print(f'{a.model}: layer {number} scores complete; AUC {auc(scores[li])}', flush=True)
    tests = dict(disjoint_natural_test_documents=True, positive_nonspecial_positions=True,
        fresh_native_pools_reused=True, finite_positive_scores=True,
        direct_block_max_abs_error_by_layer=direct_errors,
        prefix_full_sequence_max_abs_error_by_layer=info['prefix_full_sequence_max_abs_error_by_layer'],
        batch_equivalence_tolerance=dict(relative_l2=2e-5, relative_score=2e-5,
                                        elementwise_diagnostic=dict(rtol=2e-4, atol=2e-4)))
    for name in ('batch_max_abs_error', 'batch_max_relative_l2_error',
                 'batch_max_relative_score_error', 'batch_elementwise_allclose'):
        tests[name + '_by_layer'] = [t[name] for t in batch_tests]
    tests['independent_unbatched_replay'] = {str(number): t['independent_unbatched_replay']
        for number, t in zip(layers.tolist(), batch_tests) if 'independent_unbatched_replay' in t}
    metadata = dict(model=MODELS[a.model], snapshot=info['snapshot'], layers=layers.tolist(),
        classes=CLASSES, samples_per_class=128, reference_samples=256, seed=SEED, model_seed=model_seed(a.model),
        records=info['records'], reference_records=info['reference_records'],
        reference_document_rows=sorted(r['document_row'] for r in info['reference_records']),
        dataset=info['dataset'], dataset_sha256=info['dataset_sha256'], native_pools=provenance,
        fitted_statistics=statistics, score='raw single-block output norm / input norm at the unperturbed state',
        positive_class='natural', score_direction='higher means natural; fixed before evaluation',
        context='fixed [BOS/EOS, injected state] at position 1; no final model normalization',
        probe_token_id=prefix_id, sampling_policy=SAMPLING_POLICY,
        synthetic_rng='same model_seed + 1 at each layer; fresh draws coupled across layers',
        device=a.device, dtype='float32', attention_implementation='eager', threads=a.threads, batch=a.batch,
        tests=tests, auc_negative_classes=CLASSES[1:], auc_by_layer=[auc(s) for s in scores],
        torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
        hostname=platform.node(), slurm_job=os.getenv('SLURM_JOB_ID'), script_sha256=digest(__file__),
        synthetic_helper_sha256=digest(Path(__file__).with_name('great_circle_synthetic_grid.py')))
    assert digest(native / f'{a.model}.npz') == provenance['sha256']
    assert digest(native / f'{a.model}.json') == provenance['metadata_sha256']
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'{a.model}.npz'
    atomic_npz(path, layers=layers, classes=CLASSES, scores=scores, states=states,
        reference=reference, evaluation=evaluation, reference_mean=np.stack(means),
        covariance_factor=np.stack(factors), gaussian_coefficients=construction['gaussian_coefficients'],
        shuffled_permutations=construction['shuffled_permutations'])
    atomic_json(path.with_suffix('.json'), metadata)
    print(f'Saved {path}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--native-layers', type=Path, help='Fresh prepare_native output; read only')
    p.add_argument('--output', type=Path, help='Default: data-root/norm_growth_classifier_scores')
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    compute(p.parse_args())
