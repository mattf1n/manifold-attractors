"""Layer-by-layer evolution of the paper's single-activation sphere probe, restricted to one great circle.

Reuses the paper's bubble protocol (src/manifold_attractors/maps/bubble_run.py + bubble_coords.py, Fig. "Projections of
2D spheres"): Llama-3.1-8B in float32, natural activation = bubble single sample 0 (hidden_states[15] of the token
"University", records[0] of data/bubble_sources.json), tangents e1, e2 from torch.manual_seed(4242) and
bubble_coords.tangent_axes_np, injection at the last position of a [BOS, BOS] context with the paper's fast
KV-cache forward (bubble_forward.make_forward, inject_layer=15).
Probe: the great circle gamma(theta) = R (cos theta xhat + sin theta e1), R = |x| (the bubble sphere's meridian through
the activation along e1). Every decoder layer's raw residual output (no final norm) is captured for layers 15..32.
Projection at layer l: the paper's avg_image_frame with the single frame point x: a1 = unit F_l(x),
a2 = unit(J_l e1) Gram-Schmidt against a1 (central difference, ep = 1e-3 R + 1e-6). Coordinates measured from the origin.
Output: out/probe_layers.npz (theta, layers, xy (L, N, 2), act (L, 2), norms) and probe_layers.json (provenance).
"""
import json, os, platform, subprocess, sys
from pathlib import Path
import numpy as np, torch, transformers
from manifold_attractors.core.inputs import model_snapshot
from manifold_attractors.maps import bubble_coords as b
from manifold_attractors.maps.bubble_forward import make_forward
from manifold_attractors.maps.bubble_geometry import avg_image_frame

N = 720
OUT = Path.home() / 'thesis-slides/out'
torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK', '8')))
src = model_snapshot('llama3')
tok = transformers.AutoTokenizer.from_pretrained(src)
model = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32, device_map='cpu',
                                                          attn_implementation='eager').eval()
model.requires_grad_(False)
rec = json.loads((Path.home() / 'manifold-attractors/data/bubble_sources.json').read_text())['records'][0]
assert rec['token'] == 'University'
with torch.inference_mode():
    x = model(torch.tensor([rec['input_ids']]), output_hidden_states=True,
              use_cache=False).hidden_states[b.INJECT][0, rec['position']].float().numpy()
D = x.shape[0]
torch.manual_seed(4242 + 0); raw = torch.randn(D, 2).numpy()          # bubble_run.compute, single sample i = 0
e1, e2 = b.tangent_axes_np(x, raw[:, 0], raw[:, 1])
R = float(np.linalg.norm(x)); xhat = x / R
theta = np.arange(N) * 2 * np.pi / N
circle = R * (np.cos(theta)[:, None] * xhat + np.sin(theta)[:, None] * e1)
ep = 1e-3 * R + 1e-6
sten = np.stack([x + ep * e1, x - ep * e1, x + ep * e2, x - ep * e2])
X = torch.tensor(np.concatenate([circle, x[None], sten]), dtype=torch.float32)

bos = tok.bos_token_id or tok.eos_token_id
ctx = torch.tensor([[bos, bos]])
layers = list(range(b.INJECT, len(model.model.layers) + 1))        # residual index: 15 = injected input, 32 = last block output
def run(F, H, bs=128):
    outs = {l: [] for l in layers}
    for s in range(0, len(H), bs):
        cap = {}
        hooks = [model.model.layers[l - 1].register_forward_hook(
                    lambda m, i, o, l=l: cap.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[:, -1].float().clone()))
                 for l in layers[1:]]
        try:
            with torch.inference_mode():
                F(H[s:s + bs])
            # The residual at the injection layer IS the injected points. (hidden_states[15] as returned by this transformers version
            # is recorded before make_forward's replacement hook runs, so it is not used.)
            cap[b.INJECT] = H[s:s + bs].float().clone()
        finally:
            for h in hooks: h.remove()
        for l in layers: outs[l].append(cap[l])
        print(f'batch {s // bs + 1}/{-(-len(H) // bs)}', flush=True)
    return {l: torch.cat(v).numpy() for l, v in outs.items()}
fast = make_forward(model, ctx, b.INJECT, b.INJECT, fast=True)
Y = run(fast, X)
# Audit: the fast KV-cache forward against the full re-forward reference on a few circle points.
full = make_forward(model, ctx, b.INJECT, b.INJECT, fast=False)
idx = [0, 180, 360, 540]
Z = run(full, X[idx])
audit = max(float(np.max(np.abs(Y[l][idx] - Z[l]) / (np.abs(Z[l]).max() + 1e-9))) for l in layers)
print('fast/full max rel diff', audit, flush=True)
assert audit < 1e-3
xy, act, norms, frames = [], [], [], []
for l in layers:
    out = Y[l]
    FS, FP, FD = out[:N], out[N:N + 1], out[N + 1:][None]
    a1, a2, a3 = avg_image_frame(FP, FD, ep)
    A = np.stack([a1, a2], 1)
    xy.append(FS @ A); act.append(FP[0] @ A); norms.append(float(np.linalg.norm(FP[0])))
    frames.append(float(np.linalg.norm(FS - (FS @ A) @ A.T, axis=1).mean() / np.linalg.norm(FS, axis=1).mean()))
xy, act = np.array(xy), np.array(act)
assert np.isfinite(xy).all()
assert np.linalg.norm(Y[b.INJECT + 1][:N] - Y[b.INJECT + 1][0], axis=1).max() > 1e-2 * R   # later layers see the injected circle
assert np.allclose(xy[0], R * np.stack([np.cos(theta), np.sin(theta)], 1), atol=1e-2 * R)
OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT / 'probe_layers.npz', theta=theta, layers=np.array(layers), xy=xy, act=act, norms=np.array(norms))
commit = subprocess.run(['git', '-C', str(Path.home() / 'manifold-attractors'), 'rev-parse', 'HEAD'],
                        capture_output=True, text=True).stdout.strip()
(OUT / 'probe_layers.json').write_text(json.dumps(dict(
    model='meta-llama/Llama-3.1-8B', snapshot=src, dtype='float32', commit=commit, record=rec, inject=b.INJECT,
    layers=layers, points=N, R=R, ep=ep, act_norms=norms, out_of_plane_fraction=frames, fast_full_audit=audit,
    torch=torch.__version__, transformers=transformers.__version__, host=platform.node(),
    slurm_job=os.getenv('SLURM_JOB_ID')), indent=1) + '\n')
print('saved', OUT / 'probe_layers.npz', flush=True)
