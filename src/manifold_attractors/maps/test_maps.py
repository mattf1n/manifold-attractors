"""CPU-only mathematical and input-contract tests; no weights/downloads."""
import json
import unittest
from unittest.mock import patch
import numpy as np
from .geometry import surface_area_map, extend_centroid_surface_map, regions, tangent, frame_from_point
from .refusal import linear_map, next_minimum, TEMPLATE
from .language_speed import speed_from_vectors
from .bubble_geometry import fib_lattice, avg_image_frame
from .inputs import load, public_corpus

class MapsTest(unittest.TestCase):
    def test_surface_identity_and_scale(self):
        triple=np.array([[1.,.4,0.],[1.,0.,0.],[1.,0.,.4]])
        arrays,checks=surface_area_map(triple,lambda x:3*x,grid_size=9,center_kind='centroid')
        delta=np.deg2rad(.05)
        np.testing.assert_allclose(arrays['area_ratio'],9*(np.sin(delta)/delta)**2,atol=1e-10)
        np.testing.assert_allclose(arrays['basis']@arrays['basis'].T,np.eye(3),atol=1e-12)
        self.assertLess(checks['convergence_max'],1e-5)

    def test_full_sphere_seam_and_poles(self):
        triple=np.array([[1.,.4,0.],[1.,0.,0.],[1.,0.,.4]])
        old,_=surface_area_map(triple,lambda x:x,grid_size=145,center_kind='centroid')
        arrays,checks=extend_centroid_surface_map(old,lambda x:2*x,recompute_all=True)
        self.assertEqual(arrays['area_ratio'].shape,(145,289))
        np.testing.assert_allclose(arrays['area_ratio'],4,atol=2e-6)
        self.assertFalse(checks['old_hemisphere_bit_exact'])
        self.assertEqual(checks['new_points'],145*289)
        self.assertLess(checks['seam_relative_max'],1e-10)

    def test_affine_area(self):
        h=np.array([1.,2.,3.]);d=np.array([2.,0.,0.]);e=np.array([0.,1.,0.])
        arrays,check=linear_map(lambda x:x*np.array([3.,4.,5.]),h,d,e,np.linspace(-4,8,5),np.linspace(-3,3,3),batch=4)
        np.testing.assert_allclose(arrays['area_ratio'],12,atol=1e-9)
        self.assertLess(check['convergence_max'],1e-8)

    def test_random_frame_and_tangent(self):
        h=np.array([1.,2.,3.,4.]);d=np.array([3.,4.,-1.,0.])
        f=frame_from_point(h,d,20261003)
        np.testing.assert_allclose(f['basis']@f['basis'].T,np.eye(3),atol=1e-12)
        self.assertLess(abs(f['basis'][2]@d),1e-12)
        u,_=tangent(d,h);self.assertLess(abs(u@h),1e-12)

    def test_regions_cover_periodic_cells(self):
        ids=np.array([1,1,2,2,3,3,1,1])
        r=regions(ids,45.)
        self.assertEqual(sum(x['samples'] for x in r),8)
        self.assertEqual(sum(x['end']-x['start'] for x in r),360)
        self.assertEqual(len(regions(np.ones(8,int),45)),1)

    def test_speed_units(self):
        angle=np.deg2rad(np.arange(720)/2)
        v=np.stack([2*np.cos(angle),2*np.sin(angle)],axis=1)
        np.testing.assert_allclose(speed_from_vectors(v,.5),2*np.sin(np.deg2rad(.5))/np.deg2rad(.5),atol=1e-12)

    def test_fibonacci_and_output_frame(self):
        f=fib_lattice(4000)
        self.assertEqual(f.dtype,np.float32)
        np.testing.assert_allclose(np.linalg.norm(f,axis=1),1,atol=1e-7)
        fp=np.array([[2.,0.,0.]])
        fd=np.array([[[2.,.01,0.],[2.,-.01,0.],[2.,0.,.01],[2.,0.,-.01]]])
        frame=np.stack(avg_image_frame(fp,fd,.01))
        np.testing.assert_allclose(frame,np.eye(3),atol=1e-10)

    def test_fixed_input_counts_and_revisions(self):
        self.assertEqual(len(load('triple')['records']),18)
        self.assertEqual(load('triple')['sample_ids'],[4654,3707,6984,7473,6917,3872,3293,7563,4167,3026,1571,7925,8130,5826,3660,7413,4729,5465])
        self.assertEqual([r['language'] for r in load('language_area')['records']],['en','en','zh','zh','el','el'])
        self.assertEqual(load('language_speed_chinese')['indices'],[409,397,357,449])
        for name in ['triple','language_area','language_speed','language_speed_chinese']:
            data=load(name);self.assertEqual(len(data['revision']),40)
            for r in data['records']:
                self.assertGreater(r['position'],0)
                self.assertLess(r['position'],len(r['input_ids']))
        self.assertTrue(TEMPLATE.startswith('"<|begin_of_text|>'))

    def test_public_speed_corpus_rejects_changed_data(self):
        rows=[{'text':'changed passage '*30}]*200
        with patch('datasets.load_dataset',return_value=iter(rows)):
            with self.assertRaisesRegex(ValueError,'no substitution'):
                public_corpus('speed','en')

    def test_tiny_llama_fixed_and_conditional_replay(self):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        from transformers.models.llama import modeling_llama
        # The FP64 audit patches a module-level HF hook; do not leak it into
        # subsequent native-FP32 tests in the same interpreter.
        original_attention = modeling_llama.eager_attention_forward
        self.addCleanup(setattr, modeling_llama, 'eager_attention_forward', original_attention)
        from .fixed_probe import FixedProbe
        from .probes import ContextProbe
        config=LlamaConfig(vocab_size=16,hidden_size=8,intermediate_size=16,num_hidden_layers=4,
                           num_attention_heads=2,num_key_value_heads=2)
        config._attn_implementation='eager'
        torch.manual_seed(3)
        model=LlamaForCausalLM(config).double().eval().requires_grad_(False)
        class Tokenizer:
            bos_token_id=1
            eos_token_id=2
        with patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()), patch(
                'transformers.AutoModelForCausalLM.from_pretrained',return_value=model):
            probe=FixedProbe('unused',start=0,depth=4,device='cpu',batch=2)
        point=np.arange(8,dtype=float)/8
        self.assertLess(probe.audit(point),1e-9)
        context=ContextProbe(probe.layers,probe.captured,start=0)
        points=np.stack([point,point+.2])
        np.testing.assert_allclose(context.forward(points),probe.forward(points),atol=1e-9,rtol=0)

    def test_tiny_llama_bubble_cached_forward(self):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        from .bubble_forward import make_forward,forward_batched
        config=LlamaConfig(vocab_size=16,hidden_size=8,intermediate_size=16,num_hidden_layers=4,
                           num_attention_heads=2,num_key_value_heads=2)
        config._attn_implementation='eager'
        torch.manual_seed(5)
        model=LlamaForCausalLM(config).float().eval().requires_grad_(False)
        ids=torch.tensor([[1,1]])
        fast=make_forward(model,ids,1,3,fast=True)
        full=make_forward(model,ids,1,3,fast=False)
        points=torch.randn(3,8)
        torch.testing.assert_close(forward_batched(fast,points,bs=2),full(points),rtol=3e-4,atol=3e-4)

    def test_editorial_labels_are_frozen_and_fail_closed(self):
        from .language_speed import source_labels
        labels=load('language_speed_labels')
        tid,entry=next(iter(labels['tokens'].items()))
        region=dict(token_id=int(tid),text=entry['text'],token_piece=entry['token_piece'])
        self.assertEqual(source_labels('',[[region]],[]),labels)
        with self.assertRaisesRegex(ValueError,'exact figure is blocked'):
            source_labels('',[[dict(region,token_id=-123)]],[])
        with self.assertRaisesRegex(ValueError,'mismatch'):
            source_labels('',[[dict(region,text='not the original')]],[])
        chinese=load('language_speed_labels_chinese')
        self.assertEqual(set(chinese['interlinear_contexts']),{'409','397','357','449'})
        contexts=[]
        for key,annotation in chinese['interlinear_contexts'].items():
            units=annotation['units']
            contexts.append(dict(index=int(key),target_token=next(u['highlight'] for u in units if u.get('highlight')),
                                 prefix_through_target=''.join(u['source'] for u in units)))
        self.assertEqual(source_labels('_chinese',[],contexts),chinese)
        contexts[0]['target_token']='changed'
        with self.assertRaisesRegex(ValueError,'target mismatch'):
            source_labels('_chinese',[],contexts)

    def test_refusal_wrapper_uses_isolated_historical_environment(self):
        from .refusal_extract import command
        cmd=command('/tmp/author','/tmp/output')
        self.assertIn('--no-project',cmd)
        self.assertIn('--isolated',cmd)
        self.assertEqual(cmd[cmd.index('--python')+1],'3.10')
        self.assertIn('--with-requirements',cmd)
        self.assertIn('manifold_attractors.maps.refusal_direction',cmd)

    def test_next_basin_not_first_minimum(self):
        # Derivative of this scalar mapping has minima near .2 and 2.2;
        # at zero the next maximum is 1.2, so select 2.2, not .2.
        def evaluate(x):
            a=x[:,0]
            return (2*a-np.sin(np.pi*(a-.2))/np.pi)[:,None]
        alpha,_=next_minimum(evaluate,np.array([0.]),np.array([1.]))
        self.assertAlmostEqual(alpha,2.2,places=4)

if __name__=='__main__':unittest.main()
