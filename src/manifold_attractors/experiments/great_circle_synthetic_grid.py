"""Midpoint single-block great circles through natural and fitted synthetic states.

Compute only; see CORE.md for commands, estimator definitions and context.
Published selected-block plots use --block 21/21/27 with regenerated native pools.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
from .great_circle_norm_grid import MODELS, SAMPLING_POLICY, candidate_positions, digest
from ..core.inputs import dataset_path, model_snapshot

BLOCKS = {'llama3': 16, 'olmo3': 16, 'qwen3': 18}
CLASSES = ['natural', 'gaussian', 'shuffled', 'norm-matched']


def synthetic_states(pool, samples, seed):
    """Fit ddof=1 moments in float64; retain full low-rank covariance factor."""
    rng = np.random.default_rng(seed)
    pool = pool.astype(np.float64)
    n, d = pool.shape
    mean = pool.mean(0)
    factor = (pool - mean) / np.sqrt(n - 1)
    coefficients = rng.standard_normal((samples, n))
    gaussian = mean + coefficients @ factor
    # Independently permute each coordinate across ALL reference rows.
    permutations = np.stack([rng.permutation(n) for _ in range(d)], axis=1)
    shuffled_pool = np.take_along_axis(pool, permutations, axis=0)
    assert np.array_equal(np.sort(shuffled_pool, axis=0), np.sort(pool, axis=0))
    assert np.allclose(shuffled_pool.mean(0), mean, atol=1e-12)
    assert np.allclose(shuffled_pool.var(0, ddof=1), pool.var(0, ddof=1))
    # Check factor construction against explicit covariance on fixed coordinates.
    coords = np.linspace(0, d-1, min(d, 32)).astype(int)
    assert np.allclose(factor[:, coords].T @ factor[:, coords],
                       np.cov(pool[:, coords], rowvar=False, ddof=1), atol=1e-12)
    assert np.allclose(gaussian - mean, coefficients @ factor)
    norms = np.linalg.norm(pool, axis=1)
    radii = rng.normal(norms.mean(), norms.std(ddof=1), samples)
    rejected = 0
    while np.any(radii <= 0):
        bad = radii <= 0
        rejected += int(bad.sum())
        radii[bad] = rng.normal(norms.mean(), norms.std(ddof=1), bad.sum())
    directions = rng.standard_normal((samples, d))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    isotropic = directions * radii[:, None]
    assert np.allclose(np.linalg.norm(isotropic, axis=1), radii)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1)
    states = np.stack([pool[:samples], gaussian, shuffled_pool[:samples], isotropic]).astype(np.float32)
    assert np.isfinite(states).all() and (np.linalg.norm(states, axis=-1) > 0).all()
    arrays = dict(reference_mean=mean, covariance_factor=factor,
                  gaussian_coefficients=coefficients, shuffled_permutations=permutations,
                  shuffled_pool=shuffled_pool.astype(np.float32), isotropic_directions=directions,
                  isotropic_radii=radii)
    stats = dict(norm_mean=float(norms.mean()), norm_std_ddof1=float(norms.std(ddof=1)),
                 nonpositive_radius_rejections=rejected, covariance_rank_bound=min(n-1, d),
                 covariance_ddof=1, synthetic_seed=seed)
    return states, arrays, stats


def compute(a):
    import torch
    import transformers
    import pyarrow.parquet as pq
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cached_pool = None
    cache_provenance = None
    selected_block = a.block
    if a.reference_cache:
        cache_path = a.reference_cache
        if cache_path.is_dir():
            cache_path = cache_path / f'{a.model}.npz'
        cache_info = json.loads(cache_path.with_suffix('.json').read_text())
        original_path = Path(cache_info['reference_path'])
        original_info = json.loads(original_path.with_suffix('.json').read_text())
        assert a.output.resolve() != original_path.parent.resolve(), 'Do not overwrite frozen reference files'
        assert a.output.resolve() != cache_path.parent.resolve(), 'Do not overwrite classifier cache files'
        assert digest(original_path) == cache_info['reference_sha256']
        assert cache_info['reference_records'] == original_info['records']
        assert original_info['seed'] == a.seed and original_info['reference_samples'] == a.reference
        assert original_info['probe_samples'] == a.samples and original_info['max_tokens'] == a.max_tokens
        with np.load(cache_path) as data:
            layer_ids = data['layers']
            selection = None
            if selected_block is None:
                selected_block = BLOCKS[a.model]
            indices_in_cache = np.flatnonzero(layer_ids == selected_block)
            assert len(indices_in_cache) == 1
            cached_pool = data['reference'][int(indices_in_cache[0])].copy()
        assert cached_pool.dtype == np.float32 and cached_pool.shape[0] == a.reference
        assert np.isfinite(cached_pool).all()
        assert cache_info['snapshot'] == original_info['snapshot']
        cache_provenance = dict(path=str(cache_path), sha256=digest(cache_path),
            metadata_sha256=digest(cache_path.with_suffix('.json')),
            original_midpoint_path=str(original_path), original_midpoint_sha256=digest(original_path),
            original_midpoint_metadata_sha256=digest(original_path.with_suffix('.json')),
            records_exactly_match_original=True, selection=selection)
        a.dataset = original_info['dataset']
    dataset = dataset_path(a.dataset)
    docs = pq.read_table(dataset, columns=['text'])['text'].to_pylist()
    indices = np.random.default_rng(a.seed).choice(len(docs), a.reference, replace=False)
    model_seed = a.seed + 1000 * (list(MODELS).index(a.model) + 1)
    rng = np.random.default_rng(model_seed)
    snapshot = (original_info['snapshot'] if cached_pool is not None else
                model_snapshot(a.model, a.revision))
    tok = transformers.AutoTokenizer.from_pretrained(snapshot)
    model = transformers.AutoModel.from_pretrained(snapshot, dtype=torch.float32,
        device_map=a.device, attn_implementation='eager').eval()
    model.requires_grad_(False)
    block_number = selected_block if selected_block is not None else BLOCKS[a.model]
    assert 1 <= block_number <= len(model.layers)
    block = model.layers[block_number-1]
    captured = {}
    def capture(m, args, kwargs):
        captured['base'] = args[0].clone()
        captured['kwargs'] = kwargs.copy()
    probe_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    ids = torch.tensor([[probe_id, probe_id]], device=a.device)
    pool, records = [], []
    with torch.inference_mode():
        h = block.register_forward_pre_hook(capture, with_kwargs=True)
        model(ids, use_cache=False)
        h.remove()
        base, kw = captured['base'], captured['kwargs']
        reference = []
        h = block.register_forward_hook(lambda m, i, o: reference.append(o[0] if isinstance(o, tuple) else o))
        model(ids, use_cache=False)
        h.remove()
        def forward(x):
            out = block(x, **kw)
            return out[0] if isinstance(out, tuple) else out
        direct = forward(base)
        direct_error = float((direct-reference[0]).abs().max())
        assert direct_error < 1e-5
        if cached_pool is not None:
            records = cache_info['reference_records']
            assert digest(dataset) == original_info['dataset_sha256'] == cache_info['dataset_sha256']
            assert probe_id == original_info['probe_token_id'] == cache_info['probe_token_id']
            for record in records:
                position = record['position']
                assert 0 < position < len(record['input_ids'])
                assert record['input_ids'][position] not in tok.all_special_ids
                assert hashlib.sha256(docs[record['document_row']].encode()).hexdigest() == record['text_sha256']
            pool = cached_pool
        for si, di in enumerate(indices if cached_pool is None else []):
            enc = tok(docs[int(di)], truncation=True, max_length=a.max_tokens,
                      return_tensors='pt').to(a.device)
            candidates = candidate_positions(enc.input_ids[0].tolist(), tok.all_special_ids)
            if not candidates:
                raise ValueError(f'No eligible noninitial ordinary token in document {di}')
            pos = int(rng.choice(candidates))
            assert pos > 0 and int(enc.input_ids[0, pos]) not in tok.all_special_ids
            def capture_natural(m, args):
                pool.append(args[0][0, pos].cpu().numpy().copy())
            h = block.register_forward_pre_hook(capture_natural)
            model(**enc, use_cache=False)
            h.remove()
            records.append(dict(document_row=int(di), text_sha256=hashlib.sha256(docs[int(di)].encode()).hexdigest(),
                position=pos, input_ids=enc.input_ids[0].tolist(), token=tok.decode([int(enc.input_ids[0, pos])])))
            print(f'{a.model}: reference {si+1}/{a.reference}', flush=True)
        assert all(record['position'] > 0 for record in records)
        pool = np.stack(pool) if cached_pool is None else cached_pool
        states, arrays, stats = synthetic_states(pool, a.samples, model_seed+1)
        theta = np.arange(-200, 201, dtype=np.float32)
        angles = torch.tensor(np.deg2rad(theta), device=a.device)
        ratios = np.empty((4, a.samples, len(theta)), np.float32)
        tangents = np.empty_like(states)
        # Shared Gaussian draws across classes, projected relative to each state.
        tangent_draws = np.random.default_rng(model_seed+2).standard_normal((a.samples, pool.shape[1])).astype(np.float32)
        max_batch_error = 0.
        for ci, name in enumerate(CLASSES):
            for si, state in enumerate(states[ci]):
                x = torch.tensor(state, device=a.device)
                xhat = x / x.norm()
                u = torch.tensor(tangent_draws[si], device=a.device)
                u -= (u @ xhat) * xhat
                u /= u.norm()
                assert abs(float(u @ xhat)) < 1e-5
                circle = x.norm() * (angles.cos()[:, None]*xhat + angles.sin()[:, None]*u)
                assert torch.allclose(circle.norm(dim=-1), x.norm().expand(len(theta)), rtol=1e-5)
                tangents[ci, si] = u.cpu().numpy()
                for start in range(0, len(theta), a.batch):
                    grid = circle[start:start+a.batch]
                    inp = base.expand(len(grid), -1, -1).clone()
                    inp[:, 1] = grid
                    out = forward(inp)
                    if si == 0 and start == 0:
                        single = forward(inp[:1])
                        err = float((single-out[:1]).abs().max())
                        max_batch_error = max(max_batch_error, err)
                        assert torch.allclose(single, out[:1], rtol=2e-4, atol=2e-4), err
                    ratios[ci, si, start:start+len(grid)] = (out[:, 1].norm(dim=-1)/grid.norm(dim=-1)).cpu().numpy()
            print(f'{a.model}: {name} complete', flush=True)
    assert np.isfinite(ratios).all() and (ratios > 0).all()
    periodic_error = float(np.abs(ratios[..., :41]-ratios[..., 360:]).max())
    assert periodic_error < 1e-3
    provenance = dict(model=MODELS[a.model], snapshot=snapshot, dataset=str(dataset), dataset_sha256=digest(dataset),
        dataset_rows=len(docs), seed=a.seed, model_seed=model_seed, tangent_seed=model_seed+2,
        reference_samples=a.reference, probe_samples=a.samples, natural_probe_rows=list(range(a.samples)),
        max_tokens=a.max_tokens, batch=a.batch, dtype='float32', block_1based=block_number, classes=CLASSES,
        context='fixed [BOS/EOS, injected state], raw single-block residual; natural extraction in document context',
        probe_token_id=probe_id, records=records, fitted_statistics=stats,
        sampling_policy=SAMPLING_POLICY,
        dependence='Natural probes are first reference rows; all controls fit same pool. Shuffled first rows sampled without replacement. Shared preprojection tangent draws across classes.',
        tests=dict(input_positions_positive=True, direct_block_max_abs_error=direct_error, injected_batch_max_abs_error=max_batch_error,
                   periodic_max_abs_error=periodic_error, synthetic_checks='coordinate multisets/moments, covariance factor submatrix, Gaussian reconstruction, isotropic unit directions/radii; all passed'),
        torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
        device=a.device, threads=a.threads, attention_implementation='eager',
        gpu=torch.cuda.get_device_name() if a.device == 'cuda' else None,
        cpu=platform.processor(), hostname=platform.node(), slurm_job=os.getenv('SLURM_JOB_ID'),
        script_sha256=digest(__file__), helper_sha256=digest(Path(__file__).with_name('great_circle_norm_grid.py')))
    if cache_provenance is not None:
        provenance['reference_cache'] = cache_provenance
        provenance['selection'] = cache_provenance['selection']
        assert digest(original_path) == cache_provenance['original_midpoint_sha256']
    a.output.mkdir(parents=True, exist_ok=True)
    output_path = a.output / f'{a.model}.npz'
    temporary = output_path.with_name(f'{a.model}.{os.getpid()}.tmp.npz')
    np.savez_compressed(temporary, theta=theta, block=block_number, classes=CLASSES,
        ratios=ratios, reference=pool, states=states, tangents=tangents, tangent_draws=tangent_draws, **arrays)
    temporary.replace(output_path)
    temporary_json = output_path.with_name(f'{a.model}.{os.getpid()}.tmp.json')
    temporary_json.write_text(json.dumps(provenance, indent=2)+'\n')
    temporary_json.replace(output_path.with_suffix('.json'))
    print(f'Saved {a.output / a.model}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, default='llama3')
    p.add_argument('--output', type=Path, default=Path('data/great_circle_synthetic_grid'))
    p.add_argument('--block', type=int, help='one-based block; default retains original midpoint')
    p.add_argument('--reference-cache', type=Path, help='all-layer classifier NPZ or directory; replay its reference states without document forwards')
    p.add_argument('--dataset')
    p.add_argument('--revision', help='Override the pinned paper revision')
    p.add_argument('--reference', type=int, default=256)
    p.add_argument('--samples', type=int, default=20)
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--seed', type=int, default=20260909)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--threads', type=int, default=4, help='PyTorch CPU intra-op threads')
    args = p.parse_args()
    if not 1 <= args.samples <= args.reference or args.reference < 2 or args.batch < 1 or args.threads < 1:
        p.error('Require 1 <= samples <= reference, reference >= 2, batch >= 1, threads >= 1')
    if args.block is not None and args.block < 1:
        p.error('--block must be one-based and positive')
    if args.block is not None or args.reference_cache:
        original_output = Path('data/great_circle_synthetic_grid')
        if args.output.resolve() == original_output.resolve():
            args.output = original_output.with_name('great_circle_synthetic_grid_best_layers')
    compute(args)
