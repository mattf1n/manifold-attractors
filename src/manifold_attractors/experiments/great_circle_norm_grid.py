"""Generate seeded single-block great-circle norm data on a SLURM GPU.

Render separately with paper/scripts/fig/great_circle_norm_grid.py.

Natural states: one uniformly drawn nonspecial token among the first max_tokens
of each uniformly sampled document from the pinned NeelNanda/pile-10k subset.
The historical norm figures allow position zero when nonspecial. The newer
exclude-first policy remains explicit and is the helper default for other experiments.
Probe context: [BOS, injected token], or EOS when BOS is absent, as in
 gen_ray_profiles.py. Capture each block's unmodified probe input/kwargs, then
run that block alone, replacing position 1. Read raw residual output, including
for the last block (no final model RMSNorm). Block numbers are 1-based.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np

from ..core.inputs import MODELS, dataset_path, model_snapshot


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SAMPLING_POLICY = 'Uniform nonspecial token within max_tokens, excluding model-input position 0 only; no norm filtering'


HISTORICAL_NORM_POLICY = 'Uniform nonspecial token within max_tokens; position 0 eligible when nonspecial; historical norm-figure protocol'


def candidate_positions(input_ids, special_ids, *, include_position_zero=False):
    """Keep other experiments' positive-position policy unless explicitly overridden."""
    return [i for i, token in enumerate(input_ids)
            if (i > 0 or include_position_zero) and token not in special_ids]


def compute_natural_context(a):
    """Matched circles with original document prefix K/V and target RoPE position.

    The cache contains only preceding native keys/values for each direct block
    call. A new target key/value is computed for every intervention. We compare
    this efficient one-query map to whole-sequence model injections at three
    angles for every source document and every displayed block.
    """
    import torch
    import transformers
    from transformers.cache_utils import DynamicCache

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_path = a.source_cache
    if source_path.is_dir():
        source_path = source_path / f'{a.model}.npz'
    source = json.loads(source_path.with_suffix('.json').read_text())
    assert source['model'] == MODELS[a.model]
    assert a.output.resolve() != source_path.parent.resolve(), 'Never overwrite matched BOS source'
    with np.load(source_path) as data:
        theta, layers = data['theta'].copy(), data['layers'].copy()
        natural, tangents = data['natural'].copy(), data['tangents'].copy()
        bos_ratios = data['ratios'].copy()
    assert natural.shape[:2] == (8, 20) and bos_ratios.shape == (8, 20, 401)
    assert np.array_equal(theta, np.arange(-200, 201, dtype=np.float32))
    records = source['records']
    assert len(records) == 20 and len(layers) == 8
    tokenizer = transformers.AutoTokenizer.from_pretrained(source['snapshot'])
    for record in records:
        pos = record['position']
        assert 0 <= pos < len(record['input_ids'])
        assert record['input_ids'][pos] not in tokenizer.all_special_ids
    model = transformers.AutoModel.from_pretrained(source['snapshot'], dtype=torch.float32,
        device_map=a.device, attn_implementation='eager').eval()
    model.requires_grad_(False)
    ratios = np.empty_like(bos_ratios)
    native_ratios = np.empty((8, 20), np.float32)
    checks = []
    angles = torch.tensor(np.deg2rad(theta), device=a.device)
    with torch.inference_mode():
        for si, record in enumerate(records):
            position = record['position']
            ids = torch.tensor([record['input_ids']], device=a.device)
            inputs_by_layer, kwargs_by_layer, native_outputs = {}, {}, {}
            def capture_input(number):
                def hook(module, positional, keyword):
                    inputs_by_layer[number] = positional[0].clone()
                    kwargs_by_layer[number] = keyword.copy()
                return hook
            def capture_output(number):
                def hook(module, inputs, output):
                    output = output[0] if isinstance(output, tuple) else output
                    native_outputs[number] = output[:, position].clone()
                return hook
            handles = []
            for number in layers:
                block = model.layers[number - 1]
                handles += [block.register_forward_pre_hook(capture_input(number), with_kwargs=True),
                            block.register_forward_hook(capture_output(number))]
            baseline = model(ids, use_cache=True)
            for handle in handles:
                handle.remove()
            # Retain only original preceding K/V, never the source target's K/V.
            prefix_kv = [(layer.keys[:, :, :position], layer.values[:, :, :position])
                         for layer in baseline.past_key_values.layers]
            for li, number in enumerate(layers):
                block = model.layers[number - 1]
                original_input = inputs_by_layer[number]
                original_kwargs = kwargs_by_layer[number]
                x = torch.tensor(natural[li, si], device=a.device)
                tangent = torch.tensor(tangents[li, si], device=a.device)
                source_error = float((original_input[0, position] - x).abs().max())
                assert torch.allclose(original_input[0, position], x, rtol=5e-4, atol=5e-4)
                xhat = x / x.norm()
                tangent_dot = abs(float(tangent @ xhat))
                assert tangent_dot < 1e-5
                circle = x.norm() * (angles.cos()[:, None] * xhat + angles.sin()[:, None] * tangent)
                circle_norm_error = float(((circle.norm(dim=-1) - x.norm()).abs() / x.norm()).max())
                assert circle_norm_error < 1e-5

                def one_query(vectors):
                    batch = len(vectors)
                    # Every cache object is new: interventions cannot accumulate
                    # and the original native prefix remains immutable.
                    cache = DynamicCache()
                    key, value = prefix_kv[number - 1]
                    cache.update(key.expand(batch, -1, -1, -1),
                                 value.expand(batch, -1, -1, -1), int(number - 1))
                    keyword = original_kwargs.copy()
                    keyword['past_key_values'] = cache
                    keyword['use_cache'] = True
                    if keyword.get('attention_mask') is not None:
                        keyword['attention_mask'] = keyword['attention_mask'][..., position:position + 1, :position + 1]
                    if keyword.get('position_ids') is not None:
                        keyword['position_ids'] = keyword['position_ids'][..., position:position + 1]
                    if keyword.get('cache_position') is not None:
                        keyword['cache_position'] = keyword['cache_position'][position:position + 1]
                    keyword['position_embeddings'] = tuple(value[:, position:position + 1]
                        for value in keyword['position_embeddings'])
                    output = block(vectors[:, None], **keyword)
                    return (output[0] if isinstance(output, tuple) else output)[:, 0]

                batch_error = batch_relative = batch_score_error = 0.0
                for start in range(0, len(theta), a.batch):
                    vectors = circle[start:start + a.batch]
                    output = one_query(vectors)
                    single = one_query(vectors[:1])
                    batch_error = max(batch_error, float((single - output[:1]).abs().max()))
                    relative = float(((single - output[:1]).norm(dim=-1) / single.norm(dim=-1)).max())
                    score_error = float(((single.norm(dim=-1) - output[:1].norm(dim=-1)).abs() / single.norm(dim=-1)).max())
                    batch_relative = max(batch_relative, relative)
                    batch_score_error = max(batch_score_error, score_error)
                    assert relative < 5e-5 and score_error < 2e-5
                    ratios[li, si, start:start + len(vectors)] = (output.norm(dim=-1) / vectors.norm(dim=-1)).cpu().numpy()
                native_ratios[li, si] = float(native_outputs[number].norm() / original_input[0, position].norm())
                assert np.isclose(ratios[li, si, 200], native_ratios[li, si], rtol=2e-5, atol=2e-5)

                # Whole-prefix direct block, three matched angles. No K/V cache.
                angle_indices = [110, 200, 290]  # -90, 0, +90 degrees
                vectors = circle[angle_indices]
                direct_inputs = original_input.expand(3, -1, -1).clone()
                direct_inputs[:, position] = vectors
                direct_kwargs = original_kwargs.copy()
                direct_kwargs['past_key_values'] = None
                direct_kwargs['use_cache'] = False
                direct = block(direct_inputs, **direct_kwargs)
                direct = (direct[0] if isinstance(direct, tuple) else direct)[:, position]
                fast = one_query(vectors)
                direct_relative = float(((direct - fast).norm(dim=-1) / direct.norm(dim=-1)).max())
                assert direct_relative < 5e-5

                # Actual ordinary-model full-sequence injection: every block,
                # every sample, all three angles, with original absolute RoPE.
                observed = []
                def inject(module, positional):
                    modified = positional[0].clone()
                    modified[:, position] = vectors
                    return (modified,) + positional[1:]
                def observe(module, inputs, output):
                    output = output[0] if isinstance(output, tuple) else output
                    observed.append(output[:, position].clone())
                h1 = block.register_forward_pre_hook(inject)
                h2 = block.register_forward_hook(observe)
                model(ids.expand(3, -1), use_cache=False)
                h1.remove()
                h2.remove()
                ordinary = observed[0]
                full_relative = float(((ordinary - fast).norm(dim=-1) / ordinary.norm(dim=-1)).max())
                full_score_relative = float(((ordinary.norm(dim=-1) - fast.norm(dim=-1)).abs() / ordinary.norm(dim=-1)).max())
                assert full_relative < 5e-5 and full_score_relative < 2e-5
                checks.append(dict(layer=int(number), sample=si, position=position,
                    frozen_anchor_max_abs_error=source_error, tangent_dot=tangent_dot,
                    circle_relative_norm_error=circle_norm_error,
                    batch_max_abs_error=batch_error, batch_relative_l2_error=batch_relative,
                    batch_relative_score_error=batch_score_error,
                    whole_prefix_max_abs_error=float((direct - fast).abs().max()),
                    whole_prefix_relative_l2_error=direct_relative,
                    full_model_max_abs_error=float((ordinary - fast).abs().max()),
                    full_model_relative_l2_error=full_relative,
                    full_model_relative_score_error=full_score_relative,
                    full_model_elementwise_allclose=bool(torch.allclose(ordinary, fast, rtol=2e-4, atol=2e-4)),
                    native_center_score_abs_error=float(abs(ratios[li, si, 200] - native_ratios[li, si]))))
            print(f'{a.model}: matched full context sample {si + 1}/20', flush=True)
    assert np.isfinite(ratios).all() and (ratios > 0).all()
    periodic = float(np.max(np.abs(ratios[..., :41] - ratios[..., 360:])))
    assert periodic < 1e-3
    info = source.copy()
    info.update(context='Original complete causal document prefix held at native block inputs; injected target at original token position and RoPE; raw single-block output, no final norm.',
        source_cache=str(source_path), samples=20, batch=a.batch, device=a.device,
        hostname=platform.node(), slurm_job=os.getenv('SLURM_JOB_ID'),
        torch=torch.__version__, transformers=transformers.__version__,
        position_zero_samples=sum(record['position'] == 0 for record in records),
        matched_source_anchors_and_tangents=True,
        implementation='One target query with immutable original preceding K/V; fresh cache per intervention batch.',
        checks=checks, max_periodic_absolute_error=periodic,
        tests='Every sample and block: native theta0, batch equivalence, whole-prefix direct and actual full-sequence model injection at -90/0/+90 degrees.',
        validation_tolerances=dict(vector_relative_l2=5e-5, score_relative=2e-5),
        bos_comparison=dict(mean_absolute_ratio_difference=float(np.mean(np.abs(ratios - bos_ratios))),
            center_mean_by_layer=ratios[:, :, 200].mean(axis=1).tolist(),
            bos_center_mean_by_layer=bos_ratios[:, :, 200].mean(axis=1).tolist(),
            mean_absolute_difference_by_layer=np.abs(ratios - bos_ratios).mean(axis=(1, 2)).tolist()))
    info.pop('script_sha256', None)
    a.output.mkdir(parents=True, exist_ok=True)
    destination = a.output / f'{a.model}.npz'
    temporary = a.output / f'{a.model}.{os.getpid()}.tmp.npz'
    np.savez_compressed(temporary, theta=theta, layers=layers, ratios=ratios,
                        natural=natural, tangents=tangents, native_ratios=native_ratios)
    temporary.replace(destination)
    temporary = a.output / f'{a.model}.{os.getpid()}.tmp.json'
    temporary.write_text(json.dumps(info, indent=2) + '\n')
    temporary.replace(destination.with_suffix('.json'))
    print(f'Saved {destination}', flush=True)


def compute(a):
    import torch
    import transformers
    import pyarrow.parquet as pq
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(a.threads)
    dataset = dataset_path(a.dataset)
    docs = pq.read_table(dataset, columns=['text'])['text'].to_pylist()
    rng = np.random.default_rng(a.seed)
    historical = getattr(a, 'sampling_policy', 'historical-all-nonspecial') == 'historical-all-nonspecial'
    indices = rng.choice(len(docs), size=a.samples, replace=False)
    snapshot = model_snapshot(a.model, a.revision)
    tok = transformers.AutoTokenizer.from_pretrained(snapshot)
    model = transformers.AutoModel.from_pretrained(snapshot, dtype=torch.float32,
        device_map=a.device, attn_implementation='eager').eval()
    model.requires_grad_(False)
    layers = np.rint(np.linspace(0, len(model.layers)-1, 8)).astype(int)
    theta = np.arange(-200, 201, a.step, dtype=np.float32)
    assert 0 in theta and -180 in theta and 180 in theta
    angles = torch.tensor(np.deg2rad(theta), device=a.device)
    probe_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    ids = torch.tensor([[probe_id, probe_id]], device=a.device)
    captured = {}
    def capture(li):
        def hook(module, args, kwargs):
            captured[li] = (args[0].detach().clone(), kwargs.copy())
        return hook
    handles = [model.layers[int(li)].register_forward_pre_hook(capture(int(li)), with_kwargs=True)
               for li in layers]
    with torch.inference_mode():
        model(ids, use_cache=False)
    for h in handles:
        h.remove()
    natural = np.empty((8, a.samples, model.config.hidden_size), np.float32)
    tangents = np.empty_like(natural)
    ratios = np.empty((8, a.samples, len(theta)), np.float32)
    checks = []
    records = []
    with torch.inference_mode():
        # Verify direct block calls match the normal model traversal at both positions.
        for li in layers:
            base, kw = captured[int(li)]
            reference = []
            h = model.layers[int(li)].register_forward_hook(
                lambda m, i, o: reference.append((o[0] if isinstance(o, tuple) else o).clone()))
            model(ids, use_cache=False)
            h.remove()
            y = model.layers[int(li)](base, **kw)
            y = y[0] if isinstance(y, tuple) else y
            err = float((y-reference[0]).abs().max())
            assert err < 1e-5, err
            batched = model.layers[int(li)](base.expand(3, -1, -1).clone(), **kw)
            batched = batched[0] if isinstance(batched, tuple) else batched
            batch_err = float((batched-y.expand_as(batched)).abs().max())
            assert torch.allclose(batched, y.expand_as(batched), rtol=2e-4, atol=2e-4), batch_err
            checks.append({'layer': int(li+1), 'direct_block_max_abs_error': err,
                           'batch_broadcast_max_abs_error': batch_err})
        for si, di in enumerate(indices):
            enc = tok(docs[int(di)], truncation=True, max_length=a.max_tokens,
                      return_tensors='pt').to(a.device)
            candidates = candidate_positions(enc.input_ids[0].tolist(), tok.all_special_ids,
                                             include_position_zero=historical)
            if not candidates:
                raise ValueError(f'No eligible ordinary tokens in sampled document {di}')
            pos = int(rng.choice(candidates))
            assert (historical or pos > 0) and int(enc.input_ids[0, pos]) not in tok.all_special_ids
            states = {}
            def nat_hook(li):
                def hook(m, args):
                    states[li] = args[0][0, pos].float().clone()
                return hook
            hs = [model.layers[int(li)].register_forward_pre_hook(nat_hook(int(li))) for li in layers]
            model(**enc, use_cache=False)
            for h in hs:
                h.remove()
            records.append({'document_row': int(di), 'text_sha256': hashlib.sha256(docs[int(di)].encode()).hexdigest(),
                            'position': pos, 'input_ids': enc.input_ids[0].tolist(),
                            'token': tok.decode([int(enc.input_ids[0, pos])])})
            for j, li in enumerate(layers):
                x = states[int(li)]
                xhat = x / x.norm()
                u = torch.tensor(rng.standard_normal(x.numel()), device=a.device, dtype=torch.float32)
                u -= (u @ xhat) * xhat
                u /= u.norm()
                assert abs(float(u @ xhat)) < 1e-5
                circle = x.norm() * (angles.cos()[:, None]*xhat + angles.sin()[:, None]*u)
                assert torch.allclose(circle.norm(dim=-1), x.norm().expand(len(theta)), rtol=1e-5)
                natural[j, si] = x.cpu().numpy()
                tangents[j, si] = u.cpu().numpy()
                base, kw = captured[int(li)]
                for start in range(0, len(theta), a.batch):
                    grid = circle[start:start+a.batch]
                    inp = base.expand(len(grid), -1, -1).clone()
                    inp[:, 1] = grid
                    out = model.layers[int(li)](inp, **kw)
                    out = out[0] if isinstance(out, tuple) else out
                    ratios[j, si, start:start+len(grid)] = (out[:, 1].norm(dim=-1)/grid.norm(dim=-1)).cpu().numpy()
            print(f'{a.model}: sample {si+1}/{a.samples}', flush=True)
    assert historical or all(record['position'] > 0 for record in records)
    assert np.isfinite(ratios).all() and (ratios > 0).all()
    # Exact same circle points at angles separated by 360 degrees.
    periodic_error = max(float(np.max(np.abs(ratios[..., k]-ratios[..., np.where(theta == t+360)[0][0]])))
                         for k, t in enumerate(theta) if t+360 in theta)
    assert periodic_error < 1e-3, periodic_error
    provenance = dict(model=MODELS[a.model], snapshot=snapshot, dataset=str(dataset),
        dataset_sha256=digest(dataset), dataset_rows=len(docs), seed=a.seed, samples=a.samples,
        max_tokens=a.max_tokens, step=a.step, batch=a.batch, dtype='float32',
        context='fixed [BOS, injected state], EOS fallback; raw single-block residual output',
        probe_token_id=probe_id, layers_1based=(layers+1).tolist(), records=records,
        sampling_policy=HISTORICAL_NORM_POLICY if historical else SAMPLING_POLICY,
        sampling_mode='historical-all-nonspecial' if historical else 'exclude-first',
        input_positions_positive=all(record['position'] > 0 for record in records),
        checks=checks, max_periodic_absolute_error=periodic_error,
        torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
        gpu=torch.cuda.get_device_name() if a.device == 'cuda' else None, hostname=platform.node(), slurm_job=os.getenv('SLURM_JOB_ID'),
        script_sha256=digest(__file__))
    a.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output / f'{a.model}.npz', theta=theta, layers=layers+1,
                        ratios=ratios, natural=natural, tangents=tangents)
    (a.output / f'{a.model}.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(f'Saved {a.output / a.model}; periodic max error {periodic_error:g}', flush=True)



if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['compute'], nargs='?', default='compute')
    p.add_argument('--model', choices=MODELS, default='llama3')
    p.add_argument('--output', type=Path, default=Path('data/great_circle_norm_grid'))
    p.add_argument('--context', choices=['bos', 'natural'], default='bos')
    p.add_argument('--source-cache', type=Path, default=Path('data/great_circle_norm_grid'))
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='device for matched natural-context replay')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--dataset')
    p.add_argument('--revision', help='Override the pinned paper model revision')
    p.add_argument('--samples', type=int, default=20)
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--seed', type=int, default=20260908)
    p.add_argument('--sampling-policy', choices=['historical-all-nonspecial', 'exclude-first'],
                   default='historical-all-nonspecial', help='Historical norm figures include every nonspecial position')
    p.add_argument('--step', type=int, default=1)
    p.add_argument('--batch', type=int, default=128)
    args = p.parse_args()
    if args.context == 'natural':
        if args.output == Path('data/great_circle_norm_grid'):
            args.output = args.output.with_name('great_circle_norm_grid_context')
        compute_natural_context(args)
    else:
        compute(args)
