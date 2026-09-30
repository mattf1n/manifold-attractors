"""Regenerate native pools needed by selected-block plots and trajectories.

Reference records come from this repository's freshly generated midpoint
synthetic grid. Heldout records use the original classifier sampling algorithm.
--records-only performs no model forwards. All-layer extraction reproduces the
causal-prefix computation in paper_scripts/norm_growth_classifier.py; it does
not compute unrelated classifier scores or copy any precomputed activations.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from ..core.inputs import MODELS, digest, dataset_path
from ..core.records import heldout_records
from .pair_barrier_validation import atomic_json, atomic_npz


def compute(a):
    import pyarrow.parquet as pq
    import transformers
    source = a.data_root / 'great_circle_synthetic_grid' / f'{a.model}.npz'
    meta = json.loads(source.with_suffix('.json').read_text())
    dataset = dataset_path(meta['dataset'])
    documents = pq.read_table(dataset, columns=['text'])['text'].to_pylist()
    tok = transformers.AutoTokenizer.from_pretrained(meta['snapshot'])
    records = heldout_records(documents, tok, a.model, meta['records'])
    evaluation = dict(model=MODELS[a.model], snapshot=meta['snapshot'],
                      dataset=str(dataset), dataset_sha256=digest(dataset), records=records,
                      seed=20260910, model_seed=20260910 + 1000 * (list(MODELS).index(a.model) + 1),
                      probe_token_id=meta['probe_token_id'], samples_per_class=128,
                      purpose='Sampling metadata only; no classifier scores are needed by these figures')
    evaluation_path = a.data_root / 'norm_growth_classifier' / f'{a.model}.json'
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(evaluation_path, evaluation)
    if a.records_only:
        return
    import torch
    torch.set_num_threads(a.threads)
    torch.manual_seed(20260910)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = transformers.AutoModel.from_pretrained(meta['snapshot'], dtype=torch.float32,
        device_map=a.device, attn_implementation='eager').eval()
    model.requires_grad_(False)
    layers = np.arange(1, len(model.layers) + 1)
    all_records = meta['records'] + records
    natural = np.empty((len(layers), len(all_records), model.config.hidden_size), np.float32)
    prefix_errors = np.zeros(len(layers))
    with torch.inference_mode():
        for index, record in enumerate(all_records):
            position = record['position']
            captured = {}
            def capture(li):
                def hook(module, inputs):
                    captured[li] = inputs[0][0, position].cpu().numpy().copy()
                return hook
            handles = [model.layers[int(number)-1].register_forward_pre_hook(capture(li))
                       for li, number in enumerate(layers)]
            try:
                model(torch.tensor([record['input_ids'][:position+1]], device=a.device), use_cache=False)
                for li in range(len(layers)):
                    natural[li, index] = captured[li]
                if index in (0, 1, 256, 257):
                    model(torch.tensor([record['input_ids']], device=a.device), use_cache=False)
                    for li in range(len(layers)):
                        prefix_errors[li] = max(prefix_errors[li], float(np.max(np.abs(natural[li, index]-captured[li]))))
                        np.testing.assert_allclose(natural[li, index], captured[li], rtol=5e-4, atol=5e-4)
            finally:
                for h in handles:
                    h.remove()
            print(f'{a.model}: native input extraction {index+1}/{len(all_records)}', flush=True)
    output = a.output or a.data_root / 'norm_growth_classifier_layers'
    output.mkdir(parents=True, exist_ok=True)
    atomic_npz(output / f'{a.model}.npz', layers=layers, reference=natural[:, :256], evaluation=natural[:, 256:])
    info = dict(evaluation, reference_records=meta['records'], reference_path=str(source.resolve()),
                reference_sha256=digest(source), reference_samples=256, dtype='float32',
                device=a.device, torch=torch.__version__, transformers=transformers.__version__,
                prefix_full_sequence_max_abs_error_by_layer=prefix_errors.tolist())
    atomic_json(output / f'{a.model}.json', info)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path)
    p.add_argument('--records-only', action='store_true')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--threads', type=int, default=4)
    compute(p.parse_args())
