"""Raw single-block great-circle speed, in residual units per radian.

Use original-document natural states, then a fixed BOS/EOS prefix probe.
Periodic vector central differences (NOT differences of norms) use 1 degree;
the first two samples per block are also evaluated with a half-degree step.
Rendering lives in paper/scripts/fig/great_circle_speed_grid.py.
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


def vector_speed(outputs, step_degrees):
    derivative = (np.roll(outputs, -1, axis=0) - np.roll(outputs, 1, axis=0)) / (2 * np.deg2rad(step_degrees))
    return np.linalg.norm(derivative, axis=-1)


def compute(a):
    import torch
    import transformers
    import pyarrow.parquet as pq

    a.output.mkdir(parents=True, exist_ok=True)
    npz_path = a.output / f'{a.model}.npz'
    json_path = a.output / f'{a.model}.json'
    if npz_path.exists() and json_path.exists():
        saved = json.loads(json_path.read_text())
        if saved['script_sha256'] != digest(__file__):
            raise ValueError('Existing completed output has a different generator; choose a new output directory')
        print(f'Already complete: {npz_path}', flush=True)
        return
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = dataset_path(a.dataset)
    docs = pq.read_table(dataset, columns=['text'])['text'].to_pylist()
    rng = np.random.default_rng(a.seed)
    indices = rng.choice(len(docs), size=a.samples, replace=False)
    snapshot = model_snapshot(a.model, a.revision)
    tok = transformers.AutoTokenizer.from_pretrained(snapshot)
    model = transformers.AutoModel.from_pretrained(snapshot, dtype=torch.float32,
        device_map=a.device, attn_implementation='eager').eval()
    model.requires_grad_(False)
    layers = np.rint(np.linspace(0, len(model.layers)-1, 8)).astype(int)
    theta = np.arange(-200, 201)
    periodic_indices = theta % 360
    probe_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    assert probe_id is not None
    ids = torch.tensor([[probe_id, probe_id]], device=a.device)
    captured, reference = {}, {}

    def capture(li):
        def hook(module, args, kwargs):
            captured[li] = (args[0].detach().clone(), kwargs.copy())
        return hook

    def output_hook(li):
        def hook(module, args, out):
            reference[li] = (out[0] if isinstance(out, tuple) else out).clone()
        return hook

    hs = []
    for li in layers:
        hs.append(model.layers[int(li)].register_forward_pre_hook(capture(int(li)), with_kwargs=True))
        hs.append(model.layers[int(li)].register_forward_hook(output_hook(int(li))))
    with torch.inference_mode():
        model(ids, use_cache=False)
    for h in hs:
        h.remove()
    natural = np.empty((8, a.samples, model.config.hidden_size), np.float32)
    tangents = np.empty_like(natural)
    speeds = np.empty((8, a.samples, len(theta)), np.float32)
    half_speeds = np.empty((8, min(2, a.samples), len(theta)), np.float32)
    checks, records, convergence = [], [], []

    def block_output(li, inputs):
        out = model.layers[li](inputs, **captured[li][1])
        return out[0] if isinstance(out, tuple) else out

    with torch.inference_mode():
        for li in map(int, layers):
            base = captured[li][0]
            y = block_output(li, base)
            err = float((y-reference[li]).abs().max())
            batched = block_output(li, base.expand(3, -1, -1).clone())
            batch_err = float((batched-y.expand_as(batched)).abs().max())
            assert err < 1e-5
            assert torch.allclose(batched, y.expand_as(batched), rtol=2e-4, atol=2e-4)
            checks.append(dict(block=li+1, direct_max_abs=err, batch_max_abs=batch_err))
        for si, di in enumerate(indices):
            enc = tok(docs[int(di)], truncation=True, max_length=a.max_tokens,
                      return_tensors='pt').to(a.device)
            candidates = candidate_positions(enc.input_ids[0].tolist(), tok.all_special_ids)
            if not candidates:
                raise ValueError(f'No eligible token in document {di}')
            pos = int(rng.choice(candidates))
            states = {}

            def natural_hook(li):
                def hook(module, args):
                    states[li] = args[0][0, pos].float().clone()
                return hook

            hs = [model.layers[int(li)].register_forward_pre_hook(natural_hook(int(li))) for li in layers]
            model(**enc, use_cache=False)
            for h in hs:
                h.remove()
            assert pos > 0 and int(enc.input_ids[0, pos]) not in tok.all_special_ids
            records.append(dict(document_row=int(di), text_sha256=hashlib.sha256(docs[int(di)].encode()).hexdigest(),
                                position=pos, input_ids=enc.input_ids[0].tolist(),
                                token=tok.decode([int(enc.input_ids[0, pos])])))
            for j, li in enumerate(map(int, layers)):
                x = states[li]
                radius = x.norm()
                xhat = x / radius
                u = torch.tensor(rng.standard_normal(x.numel()), device=a.device, dtype=torch.float32)
                u -= (u @ xhat) * xhat
                u /= u.norm()
                assert abs(float(u @ xhat)) < 1e-5
                natural[j, si], tangents[j, si] = x.cpu().numpy(), u.cpu().numpy()

                def evaluate(degrees):
                    # Float64 trig before casting improves periodic/seam accuracy.
                    rad = np.deg2rad(degrees.astype(np.float64))
                    cos = torch.tensor(np.cos(rad), device=a.device, dtype=torch.float32)
                    sin = torch.tensor(np.sin(rad), device=a.device, dtype=torch.float32)
                    circle = radius * (cos[:, None]*xhat + sin[:, None]*u)
                    assert torch.allclose(circle.norm(dim=-1), radius.expand(len(degrees)), rtol=1e-5)
                    outputs = []
                    for start in range(0, len(circle), a.batch):
                        grid = circle[start:start+a.batch]
                        inp = captured[li][0].expand(len(grid), -1, -1).clone()
                        inp[:, 1] = grid
                        outputs.append(block_output(li, inp)[:, 1].cpu().numpy())
                    return np.concatenate(outputs)

                vectors = evaluate(np.arange(360))
                assert np.isfinite(vectors).all()
                speed = vector_speed(vectors.astype(np.float64), 1)
                speeds[j, si] = speed[periodic_indices]
                if si < 2:
                    # Interleave new half-angle outputs with the existing integer grid.
                    fine = np.empty((720, x.numel()), np.float64)
                    fine[::2] = vectors
                    fine[1::2] = evaluate(np.arange(360) + 0.5)
                    fine_speed = vector_speed(fine, 0.5)[::2]
                    half_speeds[j, si] = fine_speed[periodic_indices]
                    rel = np.abs(speed-fine_speed) / np.maximum(fine_speed, 1e-12)
                    convergence.append(dict(block=li+1, sample=si, relative_median=float(np.median(rel)),
                                            relative_p99=float(np.quantile(rel, .99)), relative_max=float(rel.max()),
                                            relative_l2=float(np.linalg.norm(speed-fine_speed)/np.linalg.norm(fine_speed))))
                    # Independently execute repeated points, rather than just testing indexing.
                    repeat = evaluate(np.array([0, 90, 180, 360, 450, 540]))
                    periodic = float(np.max(np.abs(repeat[:3]-repeat[3:])))
                    assert np.allclose(repeat[:3], repeat[3:], rtol=2e-4, atol=2e-4)
                    single = evaluate(np.array([90]))[0]
                    assert np.allclose(single, vectors[90], rtol=2e-4, atol=2e-4)
                    convergence[-1].update(output_periodic_max_abs=periodic,
                                           injected_batch_max_abs=float(np.max(np.abs(single-vectors[90]))))
            print(f'{a.model}: sample {si+1}/{a.samples}', flush=True)
    assert np.isfinite(speeds).all() and (speeds > 0).all()
    # A failure prevents publication; inspect convergence before changing resolution.
    assert max(c['relative_l2'] for c in convergence) < .05, convergence
    assert max(c['relative_p99'] for c in convergence) < .1, convergence
    files = {}
    for path in sorted(Path(snapshot).iterdir()):
        if path.is_file():
            # HF weight blob names are content hashes; hash small config/tokenizer files directly.
            files[path.name] = dict(bytes=path.stat().st_size, blob=path.resolve().name,
                                   sha256=digest(path) if path.stat().st_size < 100_000_000 else path.resolve().name,
                                   hash_source='computed' if path.stat().st_size < 100_000_000 else 'HF content-addressed blob')
    provenance = dict(model=MODELS[a.model], snapshot=snapshot, model_revision=Path(snapshot).name,
        model_files=files, dataset=str(dataset), dataset_sha256=digest(dataset), dataset_rows=len(docs),
        seed=a.seed, samples=a.samples, max_tokens=a.max_tokens, batch=a.batch, dtype='float32',
        derivative='periodic central difference of output VECTORS, 1-degree step divided by radians; no radius normalization',
        convergence_step_degrees=.5, convergence_samples_per_block=min(2, a.samples), convergence=convergence,
        context='fixed [BOS, injected state], EOS fallback; raw single-block residual output; no final norm',
        probe_token_id=probe_id, layers_1based=(layers+1).tolist(), records=records,
        sampling_policy=SAMPLING_POLICY, input_positions_positive=True, checks=checks,
        device=a.device, threads=a.threads, torch=torch.__version__, transformers=transformers.__version__,
        numpy=np.__version__, hostname=platform.node(), slurm_job=os.getenv('SLURM_JOB_ID'),
        script_sha256=digest(__file__), sampling_helper_sha256=digest(Path(__file__).with_name('great_circle_norm_grid.py')))
    temp = npz_path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, theta=theta, layers=layers+1, speeds=speeds,
                        natural=natural, tangents=tangents, half_step_speeds=half_speeds)
    temp.replace(npz_path)
    temp_json = json_path.with_suffix('.tmp.json')
    temp_json.write_text(json.dumps(provenance, indent=2)+'\n')
    temp_json.replace(json_path)
    print(f'Saved {npz_path}; max convergence relative L2={max(c["relative_l2"] for c in convergence):.4g}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--output', type=Path, default=Path('data/great_circle_speed_grid'))
    p.add_argument('--dataset')
    p.add_argument('--revision', help='Override the pinned paper revision')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--samples', type=int, default=20)
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--seed', type=int, default=20260910)
    p.add_argument('--batch', type=int, default=128)
    compute(p.parse_args())
