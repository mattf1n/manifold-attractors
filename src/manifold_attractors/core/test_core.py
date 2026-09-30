"""CPU-only regressions; no archived data, downloads, GPUs or project imports."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from .inputs import dataset_path, REVISIONS
from .records import heldout_records, pair_exclusions
from .pair_barrier_geometry import analyze, downhill_minimum, speed_from_vectors
from ..experiments.great_circle_norm_grid import candidate_positions
from ..experiments.great_circle_speed_grid import vector_speed
from ..experiments.great_circle_synthetic_grid import synthetic_states
from ..experiments.pair_barrier_validation import auc, bootstrap_auc


class CoreTests(unittest.TestCase):
    def test_sampling(self):
        self.assertEqual(candidate_positions([10, 11, 12], []), [1, 2])
        self.assertEqual(candidate_positions([0, 11, 12, 0], [0]), [1, 2])
        self.assertEqual(candidate_positions([10, 0, 11, 0, 12], [0]), [2, 4])
        for ids in ([], [10], [0, 0]):
            self.assertEqual(candidate_positions(ids, [0]), [])

    def test_synthetic_reproducibility_and_moments(self):
        pool = np.random.default_rng(3).normal(size=(32, 17)).astype(np.float32)
        states, arrays, stats = synthetic_states(pool, 20, 17)
        again, arrays2, _ = synthetic_states(pool, 20, 17)
        np.testing.assert_array_equal(states, again)
        np.testing.assert_array_equal(states[0], pool[:20])
        np.testing.assert_array_equal(np.sort(arrays['shuffled_pool'], axis=0), np.sort(pool, axis=0))
        np.testing.assert_allclose(arrays['covariance_factor'].T @ arrays['covariance_factor'], np.cov(pool, rowvar=False))
        np.testing.assert_allclose(np.linalg.norm(states[3], axis=-1), arrays['isotropic_radii'], rtol=1e-6)
        self.assertEqual(stats['covariance_ddof'], 1)
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], arrays2[key])

    def test_speed_is_raw_vector_derivative(self):
        for radius in [1., 17.]:
            angle = np.deg2rad(np.arange(360))
            vectors = radius*np.stack([np.cos(angle), np.sin(angle)], axis=1)
            expected = radius*np.sin(np.deg2rad(1))/np.deg2rad(1)
            np.testing.assert_allclose(vector_speed(vectors, 1), expected, rtol=1e-12)
            np.testing.assert_array_equal(vector_speed(vectors, 1), speed_from_vectors(vectors, 1, 1))

    def test_second_order_convergence(self):
        errors = []
        for step in [1., .5]:
            theta = np.deg2rad(np.arange(0, 360, step))
            vectors = np.stack([3*np.cos(theta), 7*np.sin(theta)], axis=1)
            exact = np.sqrt((3*np.sin(theta))**2+(7*np.cos(theta))**2)
            errors.append(np.linalg.norm(vector_speed(vectors, step)-exact)/np.linalg.norm(exact))
        self.assertAlmostEqual(errors[0]/errors[1], 4., places=4)

    def test_downhill_barrier(self):
        angle = np.deg2rad(np.arange(360))
        speed = 2-np.cos(2*angle)
        result = analyze(speed, 1., 180.)
        self.assertTrue(result['minimum_peak_minimum'])
        self.assertEqual(result['peak_over_higher_minimum'], 3.)
        self.assertTrue(result['strict_single_peak_w'])
        result = analyze(speed, 1., 10.)
        self.assertFalse(result['minimum_peak_minimum'])
        _, _, path = downhill_minimum(speed, 1., 32.4)
        self.assertTrue(np.all(np.diff(path['speeds']) <= 0))

    def test_auc_and_bootstrap(self):
        self.assertEqual(auc(np.ones(3), np.ones(4)), .5)
        self.assertEqual(auc(np.ones(3), np.zeros(4)), 1.)
        self.assertEqual(bootstrap_auc(np.ones(3), np.zeros(4), 7, draws=20), [1., 1.])

    def test_exclusion_union(self):
        docs = [f'doc {i}' for i in range(10000)]
        rows, hashes = pair_exclusions(docs)
        fit = np.random.default_rng(20260909).choice(10000, 256, replace=False)
        evaluation = np.random.default_rng(20260910).permutation(sorted(set(range(10000))-set(fit)))[:128]
        expected = set(fit) | set(evaluation)
        for seed in [20260908, 20260910]:
            expected.update(np.random.default_rng(seed).choice(10000, 20, replace=False))
        self.assertEqual(rows, expected)
        self.assertEqual(hashes, {hashlib.sha256(docs[i].encode()).hexdigest() for i in expected})

    def test_heldout_order_and_token_rng(self):
        class Tokenizer:
            all_special_ids = [0]
            def __call__(self, text, **kwargs): return {'input_ids': [0, 4, 5, 6]}
            def decode(self, ids): return str(ids[0])
        docs = [f'doc {i}' for i in range(1000)]
        fit = [{'document_row': i} for i in range(256)]
        records = heldout_records(docs, Tokenizer(), 'llama3', fit)
        expected = np.random.default_rng(20260910).permutation(list(range(256, 1000)))[:128]
        np.testing.assert_array_equal([r['document_row'] for r in records], expected)
        expected_pos = np.random.default_rng(20261910).choice([1, 2, 3], 128)
        np.testing.assert_array_equal([r['position'] for r in records], expected_pos)

    def test_pinned_inputs_reject_wrong_data(self):
        self.assertTrue(all(len(revision) == 40 for revision in REVISIONS.values()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'wrong.parquet'
            path.write_bytes(b'not the public input')
            with self.assertRaises(ValueError):
                dataset_path(path)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers')
                     and importlib.util.find_spec('pyarrow'), 'compute dependencies not installed')
class TinyModelTests(unittest.TestCase):
    def test_circle_generators_and_matched_context(self):
        """Real HF eager block/cache APIs on a tiny random CPU Llama, no weights."""
        import torch
        import transformers
        import pyarrow as pa
        import pyarrow.parquet as pq
        from ..experiments import great_circle_norm_grid as norm
        from ..experiments import great_circle_speed_grid as speed
        from ..experiments import great_circle_synthetic_grid as synthetic
        from ..experiments import great_circle_synthetic_speed_grid as synthetic_speed
        from ..experiments import prepare_native, norm_growth_trajectory_starts as trajectory
        from ..experiments import pair_barrier_validation as pair
        torch.manual_seed(0)
        config = transformers.LlamaConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
            num_hidden_layers=32, num_attention_heads=2, num_key_value_heads=2,
            bos_token_id=1, eos_token_id=2, attn_implementation='eager')
        model = transformers.LlamaModel(config).eval()
        class Tokenizer:
            bos_token_id = 1
            eos_token_id = 2
            all_special_ids = [1, 2]
            def __call__(self, text, return_tensors=None, **kwargs):
                row = int(text.split()[-1])
                ids = [1] + [3 + (row + offset) % 29 for offset in range(7)]
                if return_tensors:
                    return transformers.BatchEncoding({'input_ids': torch.tensor([ids]),
                                                       'attention_mask': torch.ones((1, len(ids)), dtype=torch.long)})
                return {'input_ids': ids}
            def decode(self, ids): return str(ids[0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root/'input.parquet'
            pq.write_table(pa.table({'text': [f'doc {i}' for i in range(1000)]}), dataset)
            snapshot = root/'snapshot'; snapshot.mkdir()
            common = dict(model='llama3', device='cpu', threads=1, batch=128, revision=None,
                          dataset=str(dataset), samples=20, max_tokens=256)
            with patch.object(transformers.AutoModel, 'from_pretrained',
                              side_effect=lambda *args, **kwargs: model.to(dtype=kwargs['dtype'])), \
                 patch.object(transformers.AutoTokenizer, 'from_pretrained', return_value=Tokenizer()), \
                 patch.object(norm, 'dataset_path', return_value=dataset), \
                 patch.object(norm, 'model_snapshot', return_value=str(snapshot)), \
                 patch.object(speed, 'dataset_path', return_value=dataset), \
                 patch.object(speed, 'model_snapshot', return_value=str(snapshot)), \
                 patch.object(synthetic, 'dataset_path', return_value=dataset), \
                 patch.object(synthetic, 'model_snapshot', return_value=str(snapshot)), \
                 patch.object(prepare_native, 'dataset_path', return_value=dataset):
                norm.compute(argparse.Namespace(**common, output=root/'norm', seed=20260908, step=1))
                norm.compute_natural_context(argparse.Namespace(**common, output=root/'context',
                    source_cache=root/'norm', seed=20260908))
                speed.compute(argparse.Namespace(**common, output=root/'speed', seed=20260910))
                synthetic.compute(argparse.Namespace(**common, output=root/'synthetic', seed=20260909,
                    reference=256, block=None, reference_cache=None))
                synthetic_speed.compute(argparse.Namespace(model='llama3', device='cpu', threads=1,
                    batch=128, source=root/'synthetic', output=root/'synthetic_speed'))
            for group, quantity, shape in [('norm', 'ratios', (8, 20, 401)),
                                          ('context', 'ratios', (8, 20, 401)),
                                          ('speed', 'speeds', (8, 20, 401)),
                                          ('synthetic', 'ratios', (4, 20, 401)),
                                          ('synthetic_speed', 'speeds', (4, 20, 401))]:
                with np.load(root/group/'llama3.npz') as d:
                    self.assertEqual(d[quantity].shape, shape)
                    self.assertTrue(np.isfinite(d[quantity]).all())
                self.assertEqual(json.loads((root/group/'llama3.json').read_text())['model'],
                                 'meta-llama/Llama-3.1-8B')
            # Reuse only this test's freshly generated midpoint pool, not any
            # archive. Exercise both trajectory stages and float32->64 audit.
            (root/'synthetic').rename(root/'great_circle_synthetic_grid')
            with patch.object(transformers.AutoModel, 'from_pretrained',
                              side_effect=lambda *args, **kwargs: model.to(dtype=kwargs['dtype'])), \
                 patch.object(transformers.AutoTokenizer, 'from_pretrained', return_value=Tokenizer()), \
                 patch.object(prepare_native, 'dataset_path', return_value=dataset):
                prepare_native.compute(argparse.Namespace(model='llama3', data_root=root,
                    output=None, records_only=False, device='cpu', threads=1))
                args = argparse.Namespace(model='llama3', data_root=root, device='cpu', threads=1,
                    seed=20260912, batch=128, cache=root/'cache',
                    output=root/'norm_growth_trajectory_starts', native_layers=root/'norm_growth_classifier_layers')
                trajectory.compute(args)
                args.output = root/'norm_growth_trajectory_starts_all_layers'
                trajectory.compute_all_layers(args)
                with np.load(args.output/'llama3.npz') as d:
                    self.assertEqual(d['ratios'].shape, (3, 32, 128, 32))
                    self.assertTrue(np.isnan(d['ratios'][:, 1, :, 0]).all())
                pair_args = argparse.Namespace(validate_model='llama3', data_root=root,
                    output=root/'pair32', precision='float32', audit_validation=None,
                    pairs=2, seed=20260915, device='cpu', threads=1, batch=128)
                pair.run_validation(pair_args)
                pair_args.output=root/'pair64'; pair_args.precision='float64'
                pair_args.audit_validation=root/'pair32'
                from transformers.models.llama import modeling_llama
                with patch.object(modeling_llama, 'eager_attention_forward', modeling_llama.eager_attention_forward):
                    pair.run_validation(pair_args)
                pair_args.paper_export=root/'pair_export'
                with patch.object(pair, 'MODELS', ['llama3']):
                    pair.collect_validation(pair_args)
                with np.load(root/'pair_export/llama3.npz') as d:
                    self.assertEqual(d['speeds'].shape, (3, 2, 401))
                self.assertTrue(json.loads((root/'pair_export/summary.json').read_text())['heldout'])


if __name__ == '__main__':
    unittest.main()
