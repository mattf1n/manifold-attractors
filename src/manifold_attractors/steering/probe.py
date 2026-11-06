"""Original evolving two-BOS FP64 four-block/full-decoder probe.

Position 0 (BOS) is causal-independent of the steered position 1, so its
keys/values are cached per block and only position 1 is propagated.
"""
import numpy as np
from ..experiments.pair_barrier_validation import enable_float64_math

class Probe:
    """Shared FP64 two-position probe: four-block vectors or full-output tokens."""

    def __init__(self, snapshot, start=15, device="cuda"):
        import torch
        import transformers

        self.device = device
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            snapshot,
            dtype=torch.float64,
            device_map=device,
            attn_implementation="eager",
        ).eval()
        self.model.requires_grad_(False)
        self.precision = enable_float64_math(self.model, "llama3")
        self.start = start
        prefix = self.tokenizer.bos_token_id
        if prefix is None:
            prefix = self.tokenizer.eos_token_id
        self.input_ids = torch.tensor([[prefix, prefix]], device=device)
        self.captured = {}

        def capture(index):
            def hook(module, inputs, kwargs):
                self.captured[index] = (inputs[0].clone(), kwargs.copy())

            return hook

        with torch.inference_mode():
            handles = [
                self.model.model.layers[index].register_forward_pre_hook(
                    capture(index), with_kwargs=True
                )
                for index in range(start, len(self.model.model.layers))
            ]
            reference = (
                self.model(self.input_ids, use_cache=False).logits[0, 1].argmax().item()
            )
            for handle in handles:
                handle.remove()
        self._prefix = self._cache_prefix()
        point = self.captured[start][0][0, 1].cpu().numpy()[None]
        assert int(self.predict(point)[0]) == reference
        self._check_single_position(point[0])

    def _cache_prefix(self):
        """Keys/values of the BOS at position 0, which never sees position 1 (causal)."""
        import torch
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        prefix = {}
        with torch.inference_mode():
            for index in range(self.start, len(self.model.model.layers)):
                layer = self.model.model.layers[index]
                attention = layer.self_attn
                hidden, kwargs = self.captured[index]
                mask = kwargs.get("attention_mask")
                # Position 1 must attend to both positions with no additive bias.
                if mask is not None:
                    row = mask[..., 1, :2]
                    assert bool(row.all() if row.dtype == torch.bool else (row == 0).all())
                cos, sin = kwargs["position_embeddings"]
                x = layer.input_layernorm(hidden[:, :1])
                shape = (1, 1, -1, attention.head_dim)
                key = attention.k_proj(x).view(shape).transpose(1, 2)
                value = attention.v_proj(x).view(shape).transpose(1, 2)
                _, key = apply_rotary_pos_emb(key, key, cos[:, :1], sin[:, :1])
                prefix[index] = (key, value, cos[:, 1:], sin[:, 1:])
        return prefix

    def _layer(self, index, hidden):
        """One decoder block for position 1 only, given (B, 1, d) states."""
        import torch
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

        layer = self.model.model.layers[index]
        attention = layer.self_attn
        key0, value0, cos, sin = self._prefix[index]
        batch = len(hidden)
        x = layer.input_layernorm(hidden)
        shape = (batch, 1, -1, attention.head_dim)
        query = attention.q_proj(x).view(shape).transpose(1, 2)
        key = attention.k_proj(x).view(shape).transpose(1, 2)
        value = attention.v_proj(x).view(shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        key = torch.cat([key0.expand(batch, -1, -1, -1), key], dim=2)
        value = torch.cat([value0.expand(batch, -1, -1, -1), value], dim=2)
        key = repeat_kv(key, attention.num_key_value_groups)
        value = repeat_kv(value, attention.num_key_value_groups)
        weights = torch.matmul(query, key.transpose(2, 3)) * attention.scaling
        weights = torch.softmax(weights, dim=-1, dtype=torch.float64)
        output = torch.matmul(weights, value).transpose(1, 2).reshape(batch, 1, -1)
        hidden = hidden + attention.o_proj(output)
        return hidden + layer.mlp(layer.post_attention_layernorm(hidden))

    def _run(self, batch, end, depths=()):
        """Propagate (B, d) position-1 states from self.start up to block `end`."""
        hidden = batch[:, None]
        kept = {}
        for index in range(self.start, end):
            hidden = self._layer(index, hidden)
            if index - self.start + 1 in depths:
                kept[index - self.start + 1] = hidden[:, 0]
        return hidden[:, 0], kept

    def _run_two_positions(self, batch, end):
        """Original evolving two-BOS propagation; reference for the cached path."""
        hidden = self.captured[self.start][0].expand(len(batch), -1, -1).clone()
        hidden[:, 1] = batch
        for index in range(self.start, end):
            value = self.model.model.layers[index](hidden, **self.captured[index][1])
            hidden = value[0] if isinstance(value, tuple) else value
        return hidden[:, 1]

    def _check_single_position(self, point):
        import torch

        random = np.random.default_rng(0)
        scale = np.linalg.norm(point)
        points = point + random.normal(size=(4, len(point))) * scale / np.sqrt(len(point))
        batch = torch.as_tensor(np.vstack([point, points]), dtype=torch.float64,
                                device=self.device)
        end = len(self.model.model.layers)
        with torch.inference_mode():
            fast = self._run(batch, end)[0]
            slow = self._run_two_positions(batch, end)
        error = float(((fast - slow).abs().max() / slow.abs().max()).item())
        assert error < 1e-12, error
        self.single_position_relative_error = error

    def forward(self, points, full=False, category_matrix=None):
        """Return vectors/argmax, or (category masses, argmax) when a matrix is supplied."""
        import torch

        assert category_matrix is None or full
        outputs = []
        predictions = []
        end = len(self.model.model.layers) if full else self.start + 4
        with torch.inference_mode():
            for offset in range(0, len(points), 128):
                batch = torch.as_tensor(
                    points[offset : offset + 128], dtype=torch.float64, device=self.device
                )
                value = self._run(batch, end)[0]
                if full:
                    logits = self.model.lm_head(self.model.model.norm(value))
                    if category_matrix is None:
                        value = logits.argmax(-1)
                    else:
                        assert logits.dtype == category_matrix.dtype == torch.float64
                        predictions.append(logits.argmax(-1).cpu().numpy())
                        value = (
                            torch.softmax(logits, dim=-1, dtype=torch.float64)
                            @ category_matrix
                        )
                outputs.append(value.cpu().numpy())
        values = np.concatenate(outputs)
        if category_matrix is not None:
            return values, np.concatenate(predictions)
        return values

    def vectors_at_depths(self, points, depths):
        """Evaluate shared prefixes once, retaining the requested block outputs."""
        import torch

        depths = tuple(sorted(set(depths)))
        assert depths and depths[0] > 0
        assert self.start + depths[-1] <= len(self.model.model.layers)
        outputs = {depth: [] for depth in depths}
        with torch.inference_mode():
            for offset in range(0, len(points), 128):
                batch = torch.as_tensor(
                    points[offset : offset + 128], dtype=torch.float64, device=self.device
                )
                _, kept = self._run(batch, self.start + depths[-1], depths)
                for depth, value in kept.items():
                    outputs[depth].append(value.cpu().numpy())
        return {depth: np.concatenate(values) for depth, values in outputs.items()}

    def predict(self, points):
        return self.forward(points, full=True)

    def probability_mass(self, points, category_matrix):
        """Temperature-one full-vocabulary probability mass for disjoint categories."""
        return self.forward(points, full=True, category_matrix=category_matrix)

    def audit(self, point, depth=4):
        import torch

        captured = []

        def inject(module, inputs):
            hidden = inputs[0].clone()
            hidden[:, 1] = torch.as_tensor(point, device=self.device, dtype=torch.float64)
            return (hidden, *inputs[1:])

        def read(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured.append(value[0, 1].cpu().numpy().copy())

        with torch.inference_mode():
            hooks = [
                self.model.model.layers[self.start].register_forward_pre_hook(inject),
                self.model.model.layers[self.start + depth - 1].register_forward_hook(
                    read
                ),
            ]
            reference = (
                self.model(self.input_ids, use_cache=False).logits[0, 1].argmax().item()
            )
            for hook in hooks:
                hook.remove()
        value = self.vectors_at_depths(np.asarray(point)[None], [depth])[depth][0]
        error = float(np.max(np.abs(value - captured[0])))
        assert error < 1e-9
        assert int(self.predict(np.asarray(point)[None])[0]) == reference
        return error
