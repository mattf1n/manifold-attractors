"""Original synthetic trajectory computation; freshly prepared reference required."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from ..experiments.great_circle_synthetic_grid import synthetic_states
CLASS_INDEX = {'gaussian': 1, 'shuffled': 2, 'norm-matched': 3}

def compute(args, model=None):
    import torch
    import transformers

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metadata = json.loads(args.reference.with_suffix('.json').read_text())
    records = metadata['reference_records']
    assert len(records) == 256 and all(r['position'] > 0 for r in records)
    blocks = np.arange(13, args.end_block + 1)
    count = len(blocks)
    with np.load(args.reference) as data:
        index = np.flatnonzero(data['layers'] == 13)
        assert len(index) == 1
        reference = data['reference'][index[0]].copy()
    model_seed = args.seed + 1000
    generated, _, statistics = synthetic_states(reference, 20, model_seed + 1)
    initial = generated[CLASS_INDEX[args.class_name]]
    tangent_draws = np.random.default_rng(model_seed + 2).standard_normal(initial.shape).astype(np.float32)
    tokenizer = transformers.AutoTokenizer.from_pretrained(metadata['snapshot'])
    assert all(r['input_ids'][r['position']] not in tokenizer.all_special_ids for r in records)
    assert tokenizer.bos_token_id == 128000
    if model is None:
        model = transformers.AutoModel.from_pretrained(metadata['snapshot'], dtype=torch.float32,
            device_map=args.device, attn_implementation='eager').eval()
        model.requires_grad_(False)
    bases, keywords, ordinary = {}, {}, {}
    def pre_hook(number):
        def capture(module, positional, keyword):
            bases[number] = positional[0].clone()
            keywords[number] = keyword.copy()
        return capture
    def out_hook(number):
        def capture(module, inputs, output):
            ordinary[number] = (output[0] if isinstance(output, tuple) else output).clone()
        return capture
    def forward(number, inputs):
        output = model.layers[number - 1](inputs, **keywords[number])
        return output[0] if isinstance(output, tuple) else output

    theta = np.arange(-200, 201, dtype=np.float32)
    ratios = np.empty((count, 20, 401), np.float32)
    anchors = np.empty((count + 1, 20, initial.shape[1]), np.float32)
    prefixes = np.empty_like(anchors)
    tangents = np.empty((count, 20, initial.shape[1]), np.float32)
    chain_ratios = np.empty((count, 20), np.float32)
    tests = dict(direct_block_max_abs=[], batch_max_abs=[], center_score_max_abs=[],
                 tangent_max_abs_dot=[], circle_max_relative_norm_error=[])
    with torch.inference_mode():
        handles = []
        for number in blocks:
            block = model.layers[number - 1]
            handles += [block.register_forward_pre_hook(pre_hook(number), with_kwargs=True),
                        block.register_forward_hook(out_hook(number))]
        probe_ids = torch.tensor([[128000, 128000]], device=args.device)
        model(probe_ids, use_cache=False)
        for handle in handles:
            handle.remove()
        for number in blocks:
            error = float((forward(number, bases[number]) - ordinary[number]).abs().max())
            tests['direct_block_max_abs'].append(error)
            assert error < 1e-5
        current = bases[13].expand(20, -1, -1).clone()
        current[:, 1] = torch.tensor(initial, device=args.device)
        anchors[0] = current[:, 1].cpu().numpy()
        prefixes[0] = current[:, 0].cpu().numpy()
        angles = torch.tensor(np.deg2rad(theta), device=args.device)
        ambient = torch.tensor(tangent_draws, device=args.device)
        for li, number in enumerate(blocks):
            advanced = forward(number, current)
            chain_ratios[li] = (advanced[:, 1].norm(dim=-1) / current[:, 1].norm(dim=-1)).cpu().numpy()
            batch_error = dot_error = norm_error = 0.0
            for si in range(20):
                x = current[si, 1]
                radius = x.norm()
                xhat = x / radius
                tangent = ambient[si] - (ambient[si] @ xhat) * xhat
                tangent /= tangent.norm()
                dot_error = max(dot_error, abs(float(tangent @ xhat)))
                circle = radius * (angles.cos()[:, None] * xhat + angles.sin()[:, None] * tangent)
                norm_error = max(norm_error, float(((circle.norm(dim=-1) - radius).abs() / radius).max()))
                tangents[li, si] = tangent.cpu().numpy()
                for start in range(0, 401, args.batch):
                    vectors = circle[start:start + args.batch]
                    inputs = current[si:si + 1].expand(len(vectors), -1, -1).clone()
                    inputs[:, 1] = vectors
                    output = forward(number, inputs)
                    if si < 2:
                        single = forward(number, inputs[:1])
                        batch_error = max(batch_error, float((single - output[:1]).abs().max()))
                        assert torch.allclose(single, output[:1], rtol=2e-4, atol=2e-4)
                    ratios[li, si, start:start + len(vectors)] = (output[:, 1].norm(dim=-1) / vectors.norm(dim=-1)).cpu().numpy()
            center_error = float(np.max(np.abs(ratios[li, :, 200] - chain_ratios[li])))
            assert np.allclose(ratios[li, :, 200], chain_ratios[li], rtol=2e-5, atol=2e-5)
            assert dot_error < 1e-5 and norm_error < 1e-5
            tests['batch_max_abs'].append(batch_error)
            tests['center_score_max_abs'].append(center_error)
            tests['tangent_max_abs_dot'].append(dot_error)
            tests['circle_max_relative_norm_error'].append(norm_error)
            current = advanced
            anchors[li + 1] = current[:, 1].cpu().numpy()
            prefixes[li + 1] = current[:, 0].cpu().numpy()
            print(f'{args.class_name} block {number}: propagated anchors and re-probed 20 circles', flush=True)

        # Independent ordinary-model replay with injection only at block 13.
        # Every raw block output must match the direct sequential chain.
        full_errors = np.zeros(count)
        for si in (0, 1):
            def inject(module, positional):
                inputs = positional[0].clone()
                inputs[:, 1] = torch.tensor(initial[si], device=args.device)
                return (inputs,) + positional[1:]
            handle = model.layers[12].register_forward_pre_hook(inject)
            hooks = [model.layers[number - 1].register_forward_hook(out_hook(number)) for number in blocks]
            model(probe_ids, use_cache=False)
            for li, number in enumerate(blocks):
                expected = torch.tensor(np.stack([prefixes[li + 1, si], anchors[li + 1, si]])[None], device=args.device)
                error = float((ordinary[number] - expected).abs().max())
                full_errors[li] = max(full_errors[li], error)
                assert torch.allclose(ordinary[number], expected, rtol=2e-4, atol=2e-4)
            handle.remove()
            for hook in hooks:
                hook.remove()
        tests['full_model_injected_chain_max_abs'] = full_errors.tolist()
        circle_full_errors = []
        for li, number in enumerate(blocks):
            anchor = torch.tensor(anchors[li, 0], device=args.device)
            tangent = torch.tensor(tangents[li, 0], device=args.device)
            vector = anchor.norm() * tangent  # independent +90-degree circle point
            def inject_start(module, positional):
                inputs = positional[0].clone()
                inputs[:, 1] = torch.tensor(initial[0], device=args.device)
                return (inputs,) + positional[1:]
            def inject_circle(module, positional):
                inputs = positional[0].clone()
                inputs[:, 1] = vector
                return (inputs,) + positional[1:]
            start_hook = model.layers[12].register_forward_pre_hook(inject_start)
            circle_hook = model.layers[number - 1].register_forward_pre_hook(inject_circle)
            output_hook = model.layers[number - 1].register_forward_hook(out_hook(number))
            model(probe_ids, use_cache=False)
            start_hook.remove()
            circle_hook.remove()
            output_hook.remove()
            inputs = torch.tensor(np.stack([prefixes[li, 0], anchors[li, 0]])[None], device=args.device)
            inputs[:, 1] = vector
            direct = forward(number, inputs)
            error = float((ordinary[number] - direct).abs().max())
            assert torch.allclose(ordinary[number], direct, rtol=2e-4, atol=2e-4)
            circle_full_errors.append(error)
        tests['full_model_circle_point_max_abs'] = circle_full_errors
    periodic_error = float(np.max(np.abs(ratios[..., :41] - ratios[..., 360:])))
    assert periodic_error < 1e-3
    assert np.isfinite(ratios).all() and (ratios > 0).all() and np.isfinite(anchors).all()
    tests['periodic_max_abs'] = periodic_error
    if args.compare_overlap:
        overlap = {}
        with np.load(args.compare_overlap) as previous:
            assert np.array_equal(previous['blocks'], blocks[:len(previous['blocks'])])
            for name, values in dict(ratios=ratios, propagated_anchors=anchors,
                    propagated_prefixes=prefixes, tangents=tangents, chain_ratios=chain_ratios).items():
                old = previous[name]
                new = values[:len(old)]
                assert np.allclose(new, old, rtol=2e-5, atol=2e-5), name
                overlap[name] = dict(max_abs=float(np.max(np.abs(new - old))),
                                     bitwise_equal=bool(np.array_equal(new, old)))
            assert np.array_equal(initial, previous['initial_states'])
            assert np.array_equal(tangent_draws, previous['tangent_draws'])
        tests['previous_run_overlap'] = overlap
    info = dict(model='meta-llama/Llama-3.1-8B', snapshot=metadata['snapshot'],
        seed=args.seed, model_seed=model_seed, synthetic_seed=model_seed + 1, tangent_seed=model_seed + 2,
        blocks=blocks.tolist(), samples=20, reference_samples=256, class_name=args.class_name,
        context=f'Fixed two-position BOS prefix; target injected at input block 13, both positions evolve through blocks 13..{args.end_block}; no final normalization.',
        interpretation='Each panel re-probes a fresh diagnostic great circle around the current propagated theta=0 anchor. Only central anchors propagate; diagnostic circle outputs are discarded. Ambient tangent draws are shared across layers.',
        reference='Cached input-block-13 reference states; positive nonspecial sampled token positions.',
        construction={
            'norm-matched': 'Original Gaussian radius from reference norm mean/std (ddof=1), rejection of nonpositive radii, random isotropic directions; no refitting after block13.',
            'gaussian': 'Reference mean and full empirical covariance factor (ddof=1); 20 Gaussian draws at input block13, no refitting or resampling later.',
            'shuffled': 'Independent per-coordinate permutations across all 256 reference rows, then first20 shuffled vectors at input block13; no refitting or resampling later.'}[args.class_name],
        negative_radius_rejections=statistics['nonpositive_radius_rejections'],
        dtype='float32', attention_implementation='eager', device=args.device, tests=tests)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / 'llama3.npz'
    temporary = args.output / f'llama3.{os.getpid()}.tmp.npz'
    np.savez_compressed(temporary, blocks=blocks, theta=theta, ratios=ratios,
        propagated_anchors=anchors, propagated_prefixes=prefixes, initial_states=initial,
        reference=reference, tangents=tangents, tangent_draws=tangent_draws, chain_ratios=chain_ratios)
    temporary.replace(destination)
    temporary = args.output / f'llama3.{os.getpid()}.tmp.json'
    temporary.write_text(json.dumps(info, indent=2) + '\n')
    temporary.replace(destination.with_suffix('.json'))
    return model

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--end-block',type=int,default=28)
    parser.add_argument('--seed',type=int,default=20260909)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--batch',type=int,default=128)
    parser.add_argument('--classes',nargs='+',choices=CLASS_INDEX,default=list(CLASS_INDEX))
    a=parser.parse_args()
    if not 13<=a.end_block<=32: parser.error('--end-block must be in 13..32')
    a.compare_overlap=None
    model=None
    a.output.mkdir(parents=True,exist_ok=True)
    for name in a.classes:
        run=argparse.Namespace(**vars(a));run.class_name=name
        run.output=a.output/name
        model=compute(run,model)
        # Paper renderer uses flat filenames, not the original class subfolders.
        stem='llama3' if name=='norm-matched' else name
        import shutil
        for suffix in ['npz','json']:
            shutil.copyfile(run.output/f'llama3.{suffix}',a.output/f'{stem}.{suffix}')

if __name__=='__main__':main()
