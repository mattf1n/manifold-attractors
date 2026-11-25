"""Full-D outputs of the paper's single-activation sphere probe (bubble single sample 0), stored as a per-layer low-rank basis, in FP64.

Same probe as probe_sphere.py (the paper's bubble protocol: Llama-3.1-8B, x = hidden_states[15] of "University" in
data/bubble_sources.json records[0], tangents e1, e2 from torch.manual_seed(4242) + bubble_coords.tangent_axes_np, the paper's
4000-point Fibonacci lattice X_k = |x| (n_k1 e1 + n_k2 e2 + n_k3 xhat), injected as the last position of a [BOS, BOS] context with the
paper's fast KV-cache forward bubble_forward.make_forward, inject_layer = 15), but the model runs in float64 and every layer's raw
residual output F_l(X_k) (layers 15..LAST, 15 = the injected sphere) is kept in full D = 4096.
Per layer l the outputs Y (4000 x D) are written as Y ~= C B with B (r x D) orthonormal: B = Gram-Schmidt of [mean direction, top
centered principal directions], r = 1 + the fewest principal directions capturing >= TARGET of the centered variance AND leaving every
point's residual (from the centered PCA, an upper bound) <= RESID * |F_l(x)| (the activation's norm at that layer). C = Y B^T.
Any 3-D linear projection of the surface onto the span of output points can be computed from C (inner products are preserved).
Also: the paper's avg_image_frame (a1, a2, a3) with frame point x, to check against assets/probe-sphere.js (FP32 run).
Output: out/probe_face/probe_face.npz, probe_face.json.
"""
import json, os, platform, subprocess
from pathlib import Path
import numpy as np

OUT = Path.home() / 'thesis-slides/out/probe_face'
LAST, TARGET, RESID = 25, 0.999, 0.005   # rank: centered variance >= TARGET and every point's residual <= RESID * |F_l(x)|

def main():
    import torch, transformers
    from manifold_attractors.core.inputs import model_snapshot
    from manifold_attractors.maps import bubble_coords as b
    from manifold_attractors.maps.bubble_forward import make_forward
    from manifold_attractors.maps.bubble_geometry import avg_image_frame
    torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK', '8')))
    src = model_snapshot('llama3')
    tok = transformers.AutoTokenizer.from_pretrained(src)
    model = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.float64, device_map='cpu',
                                                              attn_implementation='eager').eval()
    model.requires_grad_(False)
    model.model.layers = torch.nn.ModuleList(list(model.model.layers[:LAST]))   # blocks 1..LAST (as bubble_run truncates to READ)
    rec = json.loads((Path.home() / 'manifold-attractors/data/bubble_sources.json').read_text())['records'][0]
    assert rec['token'] == 'University'
    with torch.inference_mode():
        x = model(torch.tensor([rec['input_ids']]), output_hidden_states=True,
                  use_cache=False).hidden_states[b.INJECT][0, rec['position']].double().numpy()
    D = x.shape[0]
    torch.manual_seed(4242 + 0); raw = torch.randn(D, 2).numpy()          # bubble_run.compute, single sample i = 0
    e1, e2 = (v.astype(np.float64) for v in b.tangent_axes_np(x, raw[:, 0], raw[:, 1]))
    R = float(np.linalg.norm(x)); basis3 = np.stack([e1, e2, x / R])
    fib = b.FIB.astype(np.float64); n = len(fib)
    ep = 1e-3 * R + 1e-6
    frame = np.stack([x, x + ep * e1, x - ep * e1, x + ep * e2, x - ep * e2])
    H = torch.tensor(np.concatenate([frame, R * (fib @ basis3)]), dtype=torch.float64)
    bos = tok.bos_token_id or tok.eos_token_id
    layers = list(range(b.INJECT, LAST + 1))
    F = make_forward(model, torch.tensor([[bos, bos]]), b.INJECT, b.INJECT, fast=True)
    outs = {l: [] for l in layers}
    bs = 256
    for s in range(0, len(H), bs):
        cap = {}
        hooks = [model.model.layers[l - 1].register_forward_hook(
                    lambda m, i, o, l=l: cap.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[:, -1].double().clone()))
                 for l in layers[1:]]
        try:
            with torch.inference_mode():
                F(H[s:s + bs])
        finally:
            for h in hooks: h.remove()
        cap[b.INJECT] = H[s:s + bs].clone()   # the injected points themselves (see probe_layers.py on hidden_states[15])
        for l in layers: outs[l].append(cap[l].numpy())
        print(f'batch {s // bs + 1}/{-(-len(H) // bs)}', flush=True)
    result, info_layers = {}, []
    for i, l in enumerate(layers):
        Y = np.concatenate(outs[l]); outs[l] = None
        FP, FD, Y = Y[:1], Y[1:5][None], Y[5:]
        a1, a2, a3 = avg_image_frame(FP, FD, ep)
        A = np.stack([a1, a2, a3], 1)
        mu = Y.mean(0); Z = Y - mu; act_norm = float(np.linalg.norm(FP[0]))
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        var = S ** 2; frac = np.cumsum(var) / var.sum()
        k_var = int(np.searchsorted(frac, TARGET) + 1) if var.sum() > 0 else 0
        res2 = (Z ** 2).sum(1)[:, None] - np.cumsum((U * S) ** 2, axis=1)    # per point, centered residual^2 after j+1 components
        ok = np.sqrt(np.clip(res2, 0, None)).max(0) <= RESID * act_norm
        k_res = int(np.argmax(ok) + 1) if ok.any() else len(S)
        k = max(k_var, k_res) if var.sum() > 1e-20 * act_norm ** 2 * len(Y) else 0
        Bm = np.concatenate([mu[None] / np.linalg.norm(mu), Vt[:k]])
        Q, _ = np.linalg.qr(Bm.T); B = Q.T                       # orthonormal rows spanning [mean direction, top-k PCs]
        C = Y @ B.T
        rabs = np.linalg.norm(Y - C @ B, axis=1)
        resid = rabs / np.linalg.norm(Y, axis=1)
        energy = float((C ** 2).sum() / (Y ** 2).sum())
        centered = float(frac[k - 1]) if k else 1.0
        result[f'C{l}'] = C; result[f'proj{l}'] = Y @ A
        info_layers.append(dict(layer=l, rank=int(B.shape[0]), centered_variance=centered, energy=energy,
                                max_point_rel_residual=float(resid.max()), max_residual_over_act_norm=float(rabs.max() / act_norm),
                                median_residual_over_act_norm=float(np.median(rabs) / act_norm), act_norm=act_norm))
        print(info_layers[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'probe_face.npz', layers=np.array(layers), **result)
    commit = subprocess.run(['git', '-C', str(Path.home() / 'manifold-attractors'), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    (OUT / 'probe_face.json').write_text(json.dumps(dict(model='meta-llama/Llama-3.1-8B', snapshot=src, dtype='float64', commit=commit,
        record=rec, R=R, ep=ep, target=TARGET, layers=info_layers, points=n, torch=torch.__version__,
        transformers=transformers.__version__, host=platform.node(), slurm_job=os.getenv('SLURM_JOB_ID')), indent=1) + '\n')
    print('saved', flush=True)


if __name__ == '__main__':
    main()
