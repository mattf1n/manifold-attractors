"""Evolving two-position fixed BOS/EOS probe, raw output or full-model argmax."""
import numpy as np
from .probes import enable_float64_math

class FixedProbe:
    def __init__(self, snapshot, key='llama3', start=15, depth=4, device='cuda', batch=128):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device, self.batch, self.start, self.depth = device, batch, start, depth
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot)
        self.model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.float64,
            device_map=device, attn_implementation='eager').eval().requires_grad_(False)
        enable_float64_math(self.model, key)
        self.layers = self.model.model.layers
        self.captured = {}
        prefix = self.tokenizer.bos_token_id
        if prefix is None: prefix = self.tokenizer.eos_token_id
        self.ids = torch.tensor([[prefix, prefix]], device=device)
        def capture(index):
            def hook(module, args, kwargs): self.captured[index] = (args[0].clone(), kwargs.copy())
            return hook
        hooks = [layer.register_forward_pre_hook(capture(i), with_kwargs=True)
                 for i, layer in enumerate(self.layers) if i >= start]
        try:
            with torch.inference_mode(): self.model(self.ids, use_cache=False)
        finally:
            for h in hooks: h.remove()

    def forward(self, points, full=False):
        import torch
        outputs = []
        with torch.inference_mode():
            for offset in range(0, len(points), self.batch):
                batch = torch.as_tensor(points[offset:offset+self.batch], dtype=torch.float64, device=self.device)
                hidden = self.captured[self.start][0].expand(len(batch), -1, -1).clone()
                hidden[:, 1] = batch
                end = len(self.layers) if full else self.start + self.depth
                for index in range(self.start, end):
                    value = self.layers[index](hidden, **self.captured[index][1])
                    hidden = value[0] if isinstance(value, tuple) else value
                value = hidden[:, 1]
                if full: value = self.model.lm_head(self.model.model.norm(value)).argmax(-1)
                outputs.append(value.cpu().numpy())
        return np.concatenate(outputs)

    def predict(self, points): return self.forward(points, full=True)

    def audit(self, point):
        import torch
        observed = []
        def inject(module, args):
            h = args[0].clone(); h[:, 1] = torch.as_tensor(point, device=self.device)
            return (h, *args[1:])
        def read(module, args, value):
            value = value[0] if isinstance(value, tuple) else value
            observed.append(value[0, 1].cpu().numpy())
        hooks = [self.layers[self.start].register_forward_pre_hook(inject),
                 self.layers[self.start+self.depth-1].register_forward_hook(read)]
        try:
            with torch.inference_mode(): reference = self.model(self.ids, use_cache=False).logits[0,1].argmax().item()
        finally:
            for h in hooks: h.remove()
        actual = self.forward(np.asarray(point)[None])[0]
        np.testing.assert_allclose(actual, observed[0], atol=1e-9, rtol=0)
        assert int(self.predict(np.asarray(point)[None])[0]) == reference
        return float(np.max(abs(actual-observed[0])))
