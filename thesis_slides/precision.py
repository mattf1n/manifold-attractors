"""FP32 variant of the released refusal probe (imported by basins_speed.py when DTYPE=float32).
Follows the paper's own FP32 code paths (experiments/great_circle_speed_grid.py and steering/common.py load the model in
float32 with Transformers' default math). Overrides, without editing ~/manifold-attractors:
- the model loads in float32 instead of float64, and the FP64-only RMSNorm/softmax patch (enable_float64_math) is skipped;
- ContextProbe.forward feeds float32 points (it hard-codes float64) and returns float64 numpy for the finite differences;
- Backend.probe's self-check (probe vs full forward) uses an FP32 tolerance instead of atol=1e-9.
Arithmetic otherwise identical (same layers, prefix replay, hooks, centered differences)."""
import numpy as np, torch, transformers
import manifold_attractors.maps.refusal as refusal
from manifold_attractors.maps.probes import ContextProbe

DTYPE = torch.float32
_from = transformers.AutoModelForCausalLM.from_pretrained.__func__
def from_pretrained(cls, path, *args, **kwargs):
    if kwargs.get('dtype') == torch.float64: kwargs['dtype'] = DTYPE
    return _from(cls, path, *args, **kwargs)
transformers.AutoModelForCausalLM.from_pretrained = classmethod(from_pretrained)
refusal.enable_float64_math = lambda model, key: dict(precision='float32 (Transformers default math)')

def forward(self, points):
    out = []
    with torch.inference_mode():
        for offset in range(0, len(points), self.microbatch):
            h = torch.as_tensor(np.asarray(points[offset:offset + self.microbatch]), dtype=DTYPE, device=self.device)[:, None, :]
            for i, l in enumerate(self.layers, self.start):
                h = l(h, **self.kwargs[i]); h = h[0] if isinstance(h, tuple) else h
            out.append(h[:, 0].double().cpu().numpy())
    return np.concatenate(out)
ContextProbe.forward = forward

def probe(self, ids, coefficients):
    _, captured = self.call(ids, coefficients, True)
    p = ContextProbe(self.layers[12:16], captured, start=12)
    p.point = p.point.astype(np.float64)
    points = np.stack([p.point, p.point + self.direction.cpu().numpy()])
    references = []
    for point in points:
        def inject(module, args):
            h = args[0].clone(); h[:, -1] = torch.as_tensor(point, device=h.device, dtype=h.dtype)
            return (h, *args[1:])
        def read(module, args, value):
            value = value[0] if isinstance(value, tuple) else value
            references.append(value[0, -1].double().cpu().numpy())
        hooks = [self.layers[12].register_forward_pre_hook(inject), self.layers[15].register_forward_hook(read)]
        try: self.call(ids, coefficients)
        finally:
            for hook in hooks: hook.remove()
    ref = np.array(references)
    np.testing.assert_allclose(p.forward(points), ref, rtol=0, atol=1e-4 * np.abs(ref).max())   # FP32: relative 1e-4 of the largest value
    return p
refusal.Backend.probe = probe
