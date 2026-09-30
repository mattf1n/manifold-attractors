"""Original evolving two-BOS FP64 four-block/full-decoder probe."""
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
        point = self.captured[start][0][0, 1].cpu().numpy()[None]
        assert int(self.predict(point)[0]) == reference

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
                hidden = self.captured[self.start][0].expand(len(batch), -1, -1).clone()
                hidden[:, 1] = batch
                for index in range(self.start, end):
                    value = self.model.model.layers[index](
                        hidden, **self.captured[index][1]
                    )
                    hidden = value[0] if isinstance(value, tuple) else value
                value = hidden[:, 1]
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
                hidden = self.captured[self.start][0].expand(len(batch), -1, -1).clone()
                hidden[:, 1] = batch
                for depth in range(1, depths[-1] + 1):
                    layer = self.start + depth - 1
                    value = self.model.model.layers[layer](
                        hidden, **self.captured[layer][1]
                    )
                    hidden = value[0] if isinstance(value, tuple) else value
                    if depth in outputs:
                        outputs[depth].append(hidden[:, 1].cpu().numpy())
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
