"""Midpoint natural/synthetic raw speed using EXACT existing norm-grid states/tangents.

No refitting or resampling. Source reference positions and nonspecial IDs are
rechecked. Compute periodic central differences of output VECTORS, per radian,
not norm derivatives or radius-normalized stretch. Input speed equals radius.
First two samples/class additionally use half-degree differences. Store integer
output vectors for independent derivative reproduction. See CORE.md for
standalone commands; hardware allocation and rendering are separate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
from .great_circle_norm_grid import MODELS, digest
from .great_circle_speed_grid import vector_speed
from .great_circle_synthetic_grid import BLOCKS, CLASSES


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def compute(a):
    import torch
    import transformers

    source_npz = a.source / f'{a.model}.npz'
    source_json = a.source / f'{a.model}.json'
    source = json.loads(source_json.read_text())
    source_hashes = dict(npz=digest(source_npz), json=digest(source_json))
    with np.load(source_npz) as data:
        states, tangents = data['states'].copy(), data['tangents'].copy()
        assert data['classes'].tolist() == CLASSES
        assert int(data['block']) == BLOCKS[a.model]
        assert np.array_equal(states[0], data['reference'][:20])
    assert states.shape == tangents.shape and states.ndim == 3
    assert states.shape[:2] == (4, 20)
    assert source['model'] == MODELS[a.model] and source['classes'] == CLASSES
    assert source['tests']['input_positions_positive'] and source['max_tokens'] == 256
    a.output.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = a.output / f'{a.model}.npz', a.output / f'{a.model}.json'
    if npz_path.exists() and json_path.exists():
        old = json.loads(json_path.read_text())
        assert old['source_sha256'] == source_hashes and old['script_sha256'] == digest(__file__)
        print(f'Already complete: {npz_path}', flush=True)
        return
    torch.set_num_threads(a.threads)
    torch.manual_seed(source['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    snapshot = source['snapshot']
    tok = transformers.AutoTokenizer.from_pretrained(snapshot)
    assert all(r['position'] > 0 and r['input_ids'][r['position']] not in tok.all_special_ids
               for r in source['records'])
    model = transformers.AutoModel.from_pretrained(snapshot, dtype=torch.float32,
        device_map=a.device, attn_implementation='eager').eval()
    model.requires_grad_(False)
    block_number = BLOCKS[a.model]
    block = model.layers[block_number-1]
    probe_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    assert probe_id == source['probe_token_id']
    captured = {}

    def capture(module, args, kwargs):
        captured['base'], captured['kwargs'] = args[0].clone(), kwargs.copy()

    def capture_output(module, args, out):
        captured['reference'] = (out[0] if isinstance(out, tuple) else out).clone()

    def forward(inp):
        out = block(inp, **captured['kwargs'])
        return out[0] if isinstance(out, tuple) else out

    theta = np.arange(-200, 201)
    speeds = np.empty((4, 20, 401), np.float32)
    half_speeds = np.empty((4, 2, 401), np.float32)
    output_vectors = np.empty((4, 20, 360, states.shape[-1]), np.float32)
    convergence, checks = [], []
    with torch.inference_mode():
        hs = [block.register_forward_pre_hook(capture, with_kwargs=True),
              block.register_forward_hook(capture_output)]
        model(torch.tensor([[probe_id, probe_id]], device=a.device), use_cache=False)
        for h in hs:
            h.remove()
        direct = forward(captured['base'])
        direct_error = float((direct-captured['reference']).abs().max())
        assert direct_error < 1e-5
        batched = forward(captured['base'].expand(3, -1, -1).clone())
        assert torch.allclose(batched, direct.expand_as(batched), rtol=2e-4, atol=2e-4)
        base_batch_error = float((batched-direct.expand_as(batched)).abs().max())
        for ci, name in enumerate(CLASSES):
            for si in range(20):
                x = torch.tensor(states[ci, si], device=a.device)
                u = torch.tensor(tangents[ci, si], device=a.device)
                radius, xhat = x.norm(), x / x.norm()
                assert abs(float(u @ xhat)) < 1e-5 and abs(float(u.norm())-1) < 1e-5

                def evaluate(degrees):
                    rad = np.deg2rad(np.asarray(degrees, dtype=np.float64))
                    cos = torch.tensor(np.cos(rad), device=a.device, dtype=torch.float32)
                    sin = torch.tensor(np.sin(rad), device=a.device, dtype=torch.float32)
                    circle = radius * (cos[:, None]*xhat + sin[:, None]*u)
                    assert torch.allclose(circle.norm(dim=-1), radius.expand(len(rad)), rtol=1e-5)
                    outputs = []
                    for start in range(0, len(rad), a.batch):
                        grid = circle[start:start+a.batch]
                        inp = captured['base'].expand(len(grid), -1, -1).clone()
                        inp[:, 1] = grid
                        outputs.append(forward(inp)[:, 1].cpu().numpy())
                    return np.concatenate(outputs)

                vectors = evaluate(np.arange(360))
                assert np.isfinite(vectors).all()
                output_vectors[ci, si] = vectors
                speed = vector_speed(vectors.astype(np.float64), 1)
                speeds[ci, si] = speed[theta % 360]
                if si < 2:
                    fine = np.empty((720, x.numel()), np.float64)
                    fine[::2], fine[1::2] = vectors, evaluate(np.arange(360)+.5)
                    fine_speed = vector_speed(fine, .5)[::2]
                    half_speeds[ci, si] = fine_speed[theta % 360]
                    relative = np.abs(speed-fine_speed) / np.maximum(fine_speed, 1e-12)
                    convergence.append(dict(class_name=name, sample=si,
                        relative_l2=float(np.linalg.norm(speed-fine_speed)/np.linalg.norm(fine_speed)),
                        relative_median=float(np.median(relative)), relative_p99=float(np.quantile(relative, .99)),
                        relative_max=float(relative.max())))
                    repeated = evaluate([0, 90, 180, 360, 450, 540])
                    assert np.allclose(repeated[:3], repeated[3:], rtol=2e-4, atol=2e-4)
                    single = evaluate([90])[0]
                    assert np.allclose(single, vectors[90], rtol=2e-4, atol=2e-4)
                    checks.append(dict(class_name=name, sample=si,
                        periodic_output_max_abs=float(np.max(np.abs(repeated[:3]-repeated[3:]))),
                        injected_batch_max_abs=float(np.max(np.abs(single-vectors[90])))))
            print(f'{a.model}: {name} complete', flush=True)
    assert np.isfinite(speeds).all() and (speeds > 0).all()
    assert np.array_equal(speeds[..., :41], speeds[..., 360:])
    assert max(c['relative_l2'] for c in convergence) < .05, convergence
    assert max(c['relative_p99'] for c in convergence) < .1, convergence
    assert digest(source_npz) == source_hashes['npz'] and digest(source_json) == source_hashes['json']
    model_files = {}
    for path in sorted(Path(snapshot).iterdir()):
        if path.is_file():
            small = path.stat().st_size < 100_000_000
            model_files[path.name] = dict(bytes=path.stat().st_size,
                sha256=digest(path) if small else path.resolve().name,
                hash_source='computed' if small else 'HF content-addressed blob')
    provenance = dict(model=MODELS[a.model], snapshot=snapshot, model_revision=Path(snapshot).name,
        model_files=model_files, source_npz=str(source_npz), source_json=str(source_json),
        source_sha256=source_hashes, source_states_sha256=array_hash(states), source_tangents_sha256=array_hash(tangents),
        source_sampling_policy=source['sampling_policy'], records=source['records'],
        dataset_sha256=source['dataset_sha256'], natural_probe_rows=source['natural_probe_rows'],
        fitted_statistics=source['fitted_statistics'], dependence=source['dependence'],
        resampling=False, refitting=False, block_1based=block_number, classes=CLASSES, samples_per_class=20,
        probe_token_id=probe_id, context=source['context'], final_normalization=False,
        derivative='Periodic output-VECTOR central difference; 1 degree divided by radians; not radius normalized',
        convergence=convergence, half_step_degrees=.5, half_step_samples_per_class=2,
        checks=checks, direct_block_max_abs_error=direct_error, base_batch_max_abs_error=base_batch_error,
        input_positions_positive=True, special_tokens_excluded=True, seed=source['seed'],
        dtype='float32', device=a.device, threads=a.threads, batch=a.batch, attention_implementation='eager',
        torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
        hostname=platform.node(), slurm_job=os.getenv('SLURM_JOB_ID'), script_sha256=digest(__file__),
        speed_helper_sha256=digest(Path(__file__).with_name('great_circle_speed_grid.py')))
    temporary = npz_path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, theta=theta, block=block_number, classes=CLASSES, states=states,
        tangents=tangents, speeds=speeds, half_step_speeds=half_speeds,
        output_vector_angles=np.arange(360), output_vectors=output_vectors)
    temporary.replace(npz_path)
    temporary = json_path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(provenance, indent=2)+'\n')
    temporary.replace(json_path)
    print(f'Saved {npz_path}; worst half-step relative L2 {max(c["relative_l2"] for c in convergence):.4g}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--source', type=Path, default=Path('data/great_circle_synthetic_grid'))
    p.add_argument('--output', type=Path, default=Path('data/great_circle_synthetic_speed_grid'))
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch', type=int, default=128)
    compute(p.parse_args())
