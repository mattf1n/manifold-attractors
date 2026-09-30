"""Extended trajectories: midpoint starts15..20 or --all-layers starts1..depth.

All-layer mode reuses the verified native-layer classifier cache, streams norms,
and preserves midpoint input cohorts without storing quadratic full-D trajectories.

Fresh native-document fit states at each start; fit each cohort once then pass
it continuously through block20. Natural plotted baseline is ONLY the original
BOS-probe trajectory initialized at input block15, NOT fresh native states at
each block. Same records/RNG as original start15 experiment, including its
full-document extraction arithmetic, regenerated without legacy artifacts.

Schema: ratios[3 synthetic classes,6 starts,128 samples,6 output blocks];
valid[start,block]=block>=start. layer_states[...,7 state boundaries,d] has initial
state at boundary start-15 and subsequent block outputs, NaN before that boundary.
Raw per-step ratios, never cumulative ratios or artificial unity anchors.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from .great_circle_norm_grid import MODELS,digest
from .great_circle_synthetic_grid import CLASSES,synthetic_states
from ..core.trajectory_activation_cache import extract_cached
from ..core import trajectory_activation_cache

BLOCKS=np.arange(15,21)


def compute(a):
    import torch
    import transformers
    fit_path=a.data_root/f'great_circle_synthetic_grid/{a.model}.json'
    eval_path=a.data_root/f'norm_growth_classifier/{a.model}.json'
    fit_meta=json.loads(fit_path.read_text()); eval_meta=json.loads(eval_path.read_text())
    fit_records,eval_records=fit_meta['records'],eval_meta['records']
    assert len(fit_records)==256 and len(eval_records)==128
    assert not {r['document_row'] for r in fit_records}.intersection(r['document_row'] for r in eval_records)
    assert fit_meta['snapshot']==eval_meta['snapshot'] and fit_meta['dataset_sha256']==eval_meta['dataset_sha256']
    assert digest(fit_meta['dataset'])==fit_meta['dataset_sha256']
    a.output.mkdir(parents=True,exist_ok=True); output=a.output/f'{a.model}.npz'
    if output.exists() and output.with_suffix('.json').exists():
        old=json.loads(output.with_suffix('.json').read_text())
        assert old['seed']==a.seed and old['start_blocks_1based']==BLOCKS.tolist()
        assert old['snapshot']==fit_meta['snapshot'] and old['fit_records']==fit_records and old['evaluation_records']==eval_records
        print(f'Already complete: {output}',flush=True); return
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    snapshot=fit_meta['snapshot']; tok=transformers.AutoTokenizer.from_pretrained(snapshot)
    assert all(r['position']>0 and r['input_ids'][r['position']] not in tok.all_special_ids for r in fit_records+eval_records)
    model=transformers.AutoModel.from_pretrained(snapshot,dtype=torch.float32,
        device_map=a.device,attn_implementation='eager').eval(); model.requires_grad_(False)
    prefix=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    assert prefix==fit_meta['probe_token_id']==eval_meta['probe_token_id']
    ids=torch.tensor([[prefix,prefix]],device=a.device)
    captured={}; clean_outputs={}; checks=[]
    def pre(li):
        def hook(module,args,kwargs): captured[li]=(args[0].clone(),kwargs.copy())
        return hook
    def post(destination,li):
        def hook(module,args,out): destination[li]=(out[0] if isinstance(out,tuple) else out).clone()
        return hook
    def forward(li,inp):
        out=model.layers[li](inp,**captured[li][1]); return out[0] if isinstance(out,tuple) else out
    with torch.inference_mode():
        hs=[]
        for li in range(14,20):
            hs += [model.layers[li].register_forward_pre_hook(pre(li),with_kwargs=True),
                   model.layers[li].register_forward_hook(post(clean_outputs,li))]
        model(ids,use_cache=False)
        for h in hs:h.remove()
        reference,fit_cache=extract_cached(model,fit_records,list(range(14,20)),a.device,snapshot,
                                           a.cache,f'{a.model} fit')
        native_natural,eval_cache=extract_cached(model,eval_records,list(range(14,20)),a.device,snapshot,
                                                a.cache,f'{a.model} heldout')
        # The published start15 pool was extracted with full documents, whereas
        # starts16..20 used causal truncation. Recompute that arithmetic instead
        # of requiring the old norm_growth_trajectory generated-data cache.
        def full_document_start15(records):
            values=[]
            for record in records:
                def capture_first(module,args):
                    values.append(args[0][0,record['position']].cpu().numpy().copy())
                handle=model.layers[14].register_forward_pre_hook(capture_first)
                ids_full=torch.tensor([record['input_ids']],device=a.device)
                try:
                    model(ids_full,attention_mask=torch.ones_like(ids_full),use_cache=False)
                finally:
                    handle.remove()
            return np.stack(values)
        for pool,records in [(reference,fit_records),(native_natural,eval_records)]:
            full=full_document_start15(records)
            np.testing.assert_allclose(pool[0],full,rtol=2e-4,atol=2e-4)
            pool[0]=full
        def propagate(initial,start,name):
            start_li=14+start; reference_outputs={}
            def inject(module,args):
                replacement=args[0].clone(); replacement[:,1]=torch.tensor(initial[0],device=a.device)
                return (replacement,*args[1:])
            hs=[model.layers[start_li].register_forward_pre_hook(inject)]
            hs += [model.layers[li].register_forward_hook(post(reference_outputs,li)) for li in range(start_li,20)]
            model(ids,use_cache=False)
            for h in hs:h.remove()
            states=np.empty((128,7-start,model.config.hidden_size),np.float32); states[:,0]=initial
            for offset in range(0,128,a.batch):
                points=torch.tensor(initial[offset:offset+a.batch],device=a.device)
                inp=captured[start_li][0].expand(len(points),-1,-1).clone(); inp[:,1]=points
                for index,li in enumerate(range(start_li,20),start=1):
                    out=forward(li,inp); single=forward(li,inp[:1])
                    assert torch.allclose(single,out[:1],rtol=2e-4,atol=2e-4)
                    assert torch.allclose(out[:,0],clean_outputs[li][:,0].expand(len(points),-1),rtol=2e-4,atol=2e-4)
                    error=None
                    if offset==0:
                        error=float((single-reference_outputs[li]).abs().max())
                        assert torch.allclose(single,reference_outputs[li],rtol=2e-4,atol=2e-4)
                    checks.append(dict(class_name=name,start_block=15+start,block=li+1,batch_start=offset,
                                       batch_max_abs=float((single-out[:1]).abs().max()),full_model_injection_max_abs=error))
                    states[offset:offset+len(points),index]=out[:,1].cpu().numpy(); inp=out
            return states
        natural_states=propagate(native_natural[0],0,'natural-baseline')
        all_states=np.full((3,6,128,7,model.config.hidden_size),np.nan,np.float32)
        initial_states=np.empty((3,6,128,model.config.hidden_size),np.float32)
        fitted=[]; model_seed=a.seed+1000*(list(MODELS).index(a.model)+1)
        for start in range(6):
            cohort_seed=model_seed+10000*start
            generated,_,statistics=synthetic_states(reference[start],128,cohort_seed)
            fitted.append(dict(start_block=15+start,**statistics))
            for ci,name in enumerate(CLASSES[1:]):
                initial_states[ci,start]=generated[ci+1]
                trajectory=propagate(initial_states[ci,start],start,name)
                all_states[ci,start,:,start:]=trajectory
            print(f'{a.model}: start block{15+start} all3 synthetic cohorts complete',flush=True)
    norms=np.linalg.norm(all_states.astype(np.float64),axis=-1)
    ratios=norms[...,1:]/norms[...,:-1]
    natural_norms=np.linalg.norm(natural_states.astype(np.float64),axis=-1)
    natural_ratios=natural_norms[:,1:]/natural_norms[:,:-1]
    valid=np.arange(6)[None,:]>=np.arange(6)[:,None]
    for start in range(6):
        assert np.isnan(ratios[:,start,:,:start]).all()
        assert np.isfinite(ratios[:,start,:,start:]).all() and (ratios[:,start,:,start:]>0).all()
        assert np.array_equal(all_states[:,start,:,start],initial_states[:,start])
        np.testing.assert_allclose(np.cumprod(ratios[:,start,:,start:],axis=-1),
            norms[:,start,:,start+1:]/norms[:,start,:,start:start+1],rtol=1e-12,atol=1e-12)
    provenance=dict(model=MODELS[a.model],snapshot=snapshot,source_fit_metadata_sha256=digest(fit_path),
        source_eval_metadata_sha256=digest(eval_path),dataset_sha256=fit_meta['dataset_sha256'],
        fit_records=fit_records,evaluation_records=eval_records,reference_samples=256,samples_per_cohort=128,
        classes=CLASSES[1:],start_blocks_1based=BLOCKS.tolist(),output_blocks_1based=BLOCKS.tolist(),
        initialization='input of own one-based start block; fit once there; no reinitialization during propagation',
        quantity='per-block output/input norm ratio, NOT cumulative',
        natural_baseline='single continuous BOS-probe trajectory initialized input block15; NOT native-document states at later blocks',
        fit_context='fresh native-document input states for each start from identical256 fit docs; heldout128 docs captured separately',
        probe_context='initial fixed [BOS/EOS,injected target]; advance BOTH positions through20, raw residual no final norm',
        start15_extraction='full original documents with attention_mask ones; recomputed without legacy cache',
        acceleration='causal token truncation, stop at input block20, content-addressed native activation cache',
        activation_cache=dict(fit=fit_cache,evaluation=eval_cache),
        activation_cache_helper_sha256=digest(trajectory_activation_cache.__file__),
        seed=a.seed,model_seed=model_seed,start_seed_rule='model_seed+10000*(start_block-15)',fitted_statistics=fitted,
        schema='ratios[synthetic_class,start,sample,block]; layer_states[synthetic_class,start,sample,boundary0..6,hidden]; NaN before own initialization; valid[start,block]',
        checks=checks,tests=dict(disjoint_documents=True,positions_positive=True,special_tokens_excluded=True,
            no_midtrajectory_resampling=True,perstep_cumulative_identity=True,full_model_injection=True,batch_equivalence=True),
        torch=torch.__version__,transformers=transformers.__version__,dtype='float32 model; float64 norms',
        device=a.device,batch=a.batch,slurm_job=os.getenv('SLURM_JOB_ID'),script_sha256=digest(__file__),
        synthetic_helper_sha256=digest(Path(__file__).with_name('great_circle_synthetic_grid.py')))
    temp=output.with_suffix('.tmp.npz')
    np.savez_compressed(temp,blocks=BLOCKS,starts=BLOCKS,classes=CLASSES[1:],valid=valid,reference=reference,
        native_natural=native_natural,initial_states=initial_states,layer_states=all_states,norms=norms,ratios=ratios,
        natural_states=natural_states,natural_norms=natural_norms,natural_ratios=natural_ratios)
    temp.replace(output);temp=output.with_suffix('.tmp.json');temp.write_text(json.dumps(provenance,indent=2)+'\n');temp.replace(output.with_suffix('.json'))
    print(f'Saved {output}',flush=True)



def compute_all_layers(a):
    """Every native-input cohort, streamed scalar norms; the midpoint mode stays unchanged."""
    import torch
    import transformers
    from .pair_barrier_validation import atomic_npz, atomic_json

    key=a.model
    source=a.native_layers/f'{key}.npz'
    meta=json.loads(source.with_suffix('.json').read_text())
    old_path=a.data_root/f'norm_growth_trajectory_starts/{key}.npz'
    old_meta=json.loads(old_path.with_suffix('.json').read_text())
    fit_meta=json.loads((a.data_root/f'great_circle_synthetic_grid/{key}.json').read_text())
    eval_meta=json.loads((a.data_root/f'norm_growth_classifier/{key}.json').read_text())
    assert meta['snapshot']==old_meta['snapshot']==fit_meta['snapshot']==eval_meta['snapshot']
    assert meta['reference_records']==old_meta['fit_records']==fit_meta['records']
    assert meta['records']==old_meta['evaluation_records']==eval_meta['records']
    assert meta['dataset_sha256']==fit_meta['dataset_sha256']==eval_meta['dataset_sha256']
    assert meta['dtype']=='float32' and old_meta['seed']==a.seed
    fit_records,records=meta['reference_records'],meta['records']
    assert len(fit_records)==256 and len(records)==128
    assert not {r['document_row'] for r in fit_records}&{r['document_row'] for r in records}
    with np.load(source) as d:
        blocks=d['layers'].copy();reference=d['reference'].copy();native=d['evaluation'].copy()
    depth=len(blocks);hidden=reference.shape[-1]
    assert np.array_equal(blocks,np.arange(1,depth+1))
    assert reference.shape==(depth,256,hidden) and native.shape==(depth,128,hidden)
    with np.load(old_path) as d:
        old={s:d[s].copy() for s in ['reference','native_natural','initial_states','ratios','norms','starts']}
    overlap_native={}
    for new,previous,name in [(reference,old['reference'],'reference'),(native,old['native_natural'],'evaluation')]:
        delta=new[14:20].astype(np.float64)-previous
        rel=np.linalg.norm(delta,axis=-1)/np.linalg.norm(previous.astype(np.float64),axis=-1)
        # Cross-cache CPU/GPU extraction uses the classifier cache's audited tolerance;
        # runtime propagation of identical states is checked separately below at 2e-5.
        np.testing.assert_allclose(new[14:20],previous,rtol=5e-4,atol=5e-4)
        assert rel.max()<1e-4
        overlap_native[name]=dict(max_abs=float(np.abs(delta).max()),max_relative_l2=float(rel.max()),
                                  extraction_rtol=5e-4,extraction_atol=5e-4)
        # Preserve published midpoint fits exactly; these differ only by FP32 extraction arithmetic.
        new[14:20]=previous
    a.output.mkdir(parents=True,exist_ok=True)
    checkpoint=a.output/'checkpoints'/key;checkpoint.mkdir(parents=True,exist_ok=True)
    model_seed=a.seed+1000*(list(MODELS).index(key)+1)
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tok=transformers.AutoTokenizer.from_pretrained(meta['snapshot'])
    assert all(r['position']>0 and r['input_ids'][r['position']] not in tok.all_special_ids for r in fit_records+records)
    model=transformers.AutoModel.from_pretrained(meta['snapshot'],dtype=torch.float32,
        device_map=a.device,attn_implementation='eager').eval();model.requires_grad_(False)
    assert len(model.layers)==depth and model.config.hidden_size==hidden
    prefix=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    assert prefix==meta['probe_token_id']==fit_meta['probe_token_id']==eval_meta['probe_token_id']
    ids=torch.tensor([[prefix,prefix]],device=a.device)
    captured={};clean={};checks=[];overlap=[];fitted=[]
    def capture(li):
        def hook(module,args,kwargs):captured[li]=(args[0].clone(),kwargs.copy())
        return hook
    def post(dst,li):
        def hook(module,args,out):dst[li]=(out[0] if isinstance(out,tuple) else out).clone()
        return hook
    def compare(x,y,tolerance=2e-5):
        x=x.detach().double();y=y.detach().double()
        rel=float(((x-y).norm(dim=-1)/y.norm(dim=-1).clamp_min(1e-30)).max())
        normrel=float(((x.norm(dim=-1)-y.norm(dim=-1)).abs()/y.norm(dim=-1).clamp_min(1e-30)).max())
        assert rel<tolerance and normrel<tolerance,(rel,normrel,tolerance)
        return dict(max_abs=float((x-y).abs().max()),relative_l2=rel,relative_norm=normrel,
                    elementwise_allclose=bool(torch.allclose(x,y,rtol=2e-4,atol=2e-4)))
    with torch.inference_mode():
        handles=[]
        for li in range(depth):
            handles.extend([model.layers[li].register_forward_pre_hook(capture(li),with_kwargs=True),
                            model.layers[li].register_forward_hook(post(clean,li))])
        model(ids,use_cache=False)
        for h in handles:h.remove()
        def propagate(initial,start,name):
            path=checkpoint/f'{name}-{start+1:02d}.npz'
            info_path=path.with_suffix('.json')
            if path.exists() and info_path.exists():
                with np.load(path) as d:
                    assert np.array_equal(d['initial'],initial)
                    norms=d['norms'].copy()
                assert norms.shape==(128,depth-start+1)
                checks.extend(json.loads(info_path.read_text()))
                return norms
            full={}
            def inject(module,args):
                replacement=args[0].clone();replacement[:,1]=torch.tensor(initial[0],device=a.device)
                return (replacement,*args[1:])
            handles=[model.layers[start].register_forward_pre_hook(inject)]
            handles.extend(model.layers[li].register_forward_hook(post(full,li)) for li in range(start,depth))
            model(ids,use_cache=False)
            for h in handles:h.remove()
            norms=np.empty((128,depth-start+1),np.float64)
            norms[:,0]=np.linalg.norm(initial.astype(np.float64),axis=-1)
            local_checks=[]
            for offset in range(0,128,a.batch):
                points=torch.tensor(initial[offset:offset+a.batch],device=a.device)
                inp=captured[start][0].expand(len(points),-1,-1).clone();inp[:,1]=points
                for index,li in enumerate(range(start,depth),1):
                    def forward(value):
                        out=model.layers[li](value,**captured[li][1]);return out[0] if isinstance(out,tuple) else out
                    out=forward(inp);single=forward(inp[:1])
                    batch=compare(out[:1],single)
                    prefix_check=compare(out[:,0],clean[li][:,0].expand(len(points),-1))
                    # Independent full-chain FP32 accumulation: retain legacy 2e-4 scale;
                    # single-block batch checks above remain at the tighter 2e-5.
                    injection=compare(out[:1],full[li],tolerance=2e-4) if offset==0 else None
                    local_checks.append(dict(class_name=name,start_block=start+1,block=li+1,batch_start=offset,
                                             batch=batch,prefix=prefix_check,full_injection=injection))
                    norms[offset:offset+len(points),index]=np.linalg.norm(out[:,1].cpu().numpy().astype(np.float64),axis=-1)
                    inp=out
            assert np.isfinite(norms).all() and (norms>0).all()
            atomic_npz(path,initial=initial,norms=norms);atomic_json(info_path,local_checks)
            checks.extend(local_checks)
            return norms
        natural_norms=propagate(native[0],0,'natural-baseline')
        norms=np.full((3,depth,128,depth+1),np.nan,np.float64)
        initial_states=np.empty((3,depth,128,hidden),np.float32)
        for start in range(depth):
            seed=model_seed+10000*(start+1-15)
            generated,_,statistics=synthetic_states(reference[start],128,seed)
            fitted.append(dict(start_block=start+1,seed=seed,**statistics))
            if 14<=start<20:
                previous=old['initial_states'][:,start-14]
                np.testing.assert_allclose(generated[1:],previous,rtol=2e-4,atol=2e-4)
                generated[1:]=previous
            for ci,name in enumerate(CLASSES[1:]):
                initial_states[ci,start]=generated[ci+1]
                trajectory=propagate(initial_states[ci,start],start,name)
                norms[ci,start,:,start:]=trajectory
                if 14<=start<20:
                    fresh=trajectory[:,1:21-start]/trajectory[:,:20-start]
                    previous=old['ratios'][ci,start-14,:,start-14:]
                    assert fresh.shape==previous.shape
                    np.testing.assert_allclose(fresh,previous,rtol=2e-4,atol=2e-4)
                    overlap.append(dict(start_block=start+1,class_name=name,initial_exact=True,
                                        ratios_exact=bool(np.array_equal(fresh,previous)),
                                        max_abs_ratio=float(np.abs(fresh-previous).max()),
                                        max_relative_ratio=float(np.max(np.abs(fresh-previous)/previous))))
            print(f'{key}: start {start+1}/{depth} complete',flush=True)
    ratios=norms[...,1:]/norms[...,:-1]
    natural_ratios=natural_norms[:,1:]/natural_norms[:,:-1]
    valid=blocks[None,:]>=blocks[:,None]
    for start in range(depth):
        assert np.isnan(ratios[:,start,:,:start]).all()
        assert np.isfinite(ratios[:,start,:,start:]).all() and (ratios[:,start,:,start:]>0).all()
        np.testing.assert_array_equal(norms[:,start,:,start],np.linalg.norm(initial_states[:,start].astype(np.float64),axis=-1))
        np.testing.assert_allclose(np.cumprod(ratios[:,start,:,start:],axis=-1),
                                  norms[:,start,:,start+1:]/norms[:,start,:,start:start+1],rtol=1e-12,atol=1e-12)
    assert np.isfinite(natural_ratios).all() and (natural_ratios>0).all()
    np.testing.assert_allclose(np.cumprod(natural_ratios,axis=-1),natural_norms[:,1:]/natural_norms[:,:1],rtol=1e-12,atol=1e-12)
    native_norms=np.linalg.norm(native.astype(np.float64),axis=-1)
    atomic_npz(a.output/f'{key}.npz',blocks=blocks,starts=blocks,classes=CLASSES[1:],valid=valid,
        reference=reference,native_natural=native,initial_states=initial_states,norms=norms,ratios=ratios,
        natural_norms=natural_norms,natural_ratios=natural_ratios,native_input_norms=native_norms,
        native_document_ratio_blocks=blocks[:-1],native_document_ratios=(native_norms[1:]/native_norms[:-1]).T)
    result=dict(model=MODELS[key],snapshot=meta['snapshot'],seed=a.seed,model_seed=model_seed,
        start_seed_rule='model_seed+10000*(start_block-15)',classes=CLASSES[1:],
        start_blocks_1based=blocks.tolist(),output_blocks_1based=blocks.tolist(),reference_samples=256,samples_per_cohort=128,
        fit_records=fit_records,evaluation_records=records,native_cache=str(source),native_cache_method='validated native input states, no new document forward',
        source_dataset_sha256=meta['dataset_sha256'],midpoint_native_arithmetic_differences=overlap_native,
        midpoint_reuse='start15..20 exact original reference/native/initial states; all output ratios recomputed, not spliced',
        original_midpoint_path=str(old_path),overlap_checks=overlap,probe_token_id=prefix,
        natural_baseline='single continuous trajectory initialized at input block1; BOTH fixed BOS/EOS probe positions advance; NOT native document trajectory',
        native_diagnostic='native-document input-norm ratios through penultimate block only, separate from fixed-probe baseline',
        quantity='per-block output/input norm ratio, NOT cumulative; raw outputs, no final norm',
        schema='ratios[class,start,sample,block], norms[class,start,sample,boundary]; NaN before start; full hidden trajectories not stored',
        fitted_statistics=fitted,checks=checks,tests=dict(frozen_records_match=True,disjoint_documents=True,
        nonspecial_positive_positions=True,fit_once_per_start=True,chain_norm_identity=True,finite_positive=True),
        dtype='float32 model; float64 norms',device=a.device,slurm_job=os.getenv('SLURM_JOB_ID'),
        verification_tolerances=dict(single_block_relative_vector_l2=2e-5,single_block_relative_norm=2e-5,
            full_chain_relative_vector_l2=2e-4,full_chain_relative_norm=2e-4,midpoint_ratio_rtol=2e-4,midpoint_ratio_atol=2e-4),
        torch=torch.__version__,transformers=transformers.__version__)
    atomic_json(a.output/f'{key}.json',result)
    print(f'Saved {a.output/key} all layers',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--all-layers',action='store_true',help='Every native-input start through final block; streamed norms, existing classifier-layer cache')
    p.add_argument('--data-root',type=Path,default=Path('data'))
    p.add_argument('--native-layers',type=Path)
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--cache',type=Path)
    p.add_argument('--seed',type=int,default=20260912)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--batch',type=int,default=128)
    a=p.parse_args()
    if a.cache is None:a.cache=a.data_root/'trajectory_activation_cache'
    if a.native_layers is None:a.native_layers=a.data_root/'norm_growth_classifier_layers'
    if a.output is None:a.output=a.data_root/('norm_growth_trajectory_starts_all_layers' if a.all_layers else 'norm_growth_trajectory_starts')
    (compute_all_layers if a.all_layers else compute)(a)
