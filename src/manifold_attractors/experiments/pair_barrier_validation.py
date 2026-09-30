"""Held-out four-block velocity-barrier validation for the paper.

128 disjoint endpoint-pairs/class; original fit; frozen log barrier-strength score.
Generate fresh float32 states, then audit EXACT prepared states in float64 using
--precision float64 --audit-validation <first-run-dir> --output <audit-dir>.
Float64 includes RMSNorm/softmax, avoiding hardcoded HF float32 casts.
Export: --collect-validation --output <audit-dir> --paper-export <paper-data-dir>.
Allocate hardware separately; standalone reproduction commands are in CORE.md.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from ..core.inputs import digest as file_hash
from ..core.records import pair_exclusions
from ..core import pair_barrier_geometry
MODELS=['llama3','olmo3','qwen3']

def auc(positive,negative):
    differences=positive[:,None]-negative[None,:]
    return float(np.mean((differences>0)+.5*(differences==0)))

def atomic_npz(path,**arrays):
    temporary=path.with_name(f'{path.stem}.{os.getpid()}.tmp.npz')
    np.savez_compressed(temporary,**arrays);temporary.replace(path)

def atomic_json(path,value):
    temporary=path.with_name(f'{path.stem}.{os.getpid()}.tmp.json')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def bootstrap_auc(positive,negative,seed,draws=2000):
    rng=np.random.default_rng(seed)
    estimates=[auc(rng.choice(positive,len(positive)),rng.choice(negative,len(negative))) for _ in range(draws)]
    return np.quantile(estimates,[.025,.975]).tolist()

def enable_float64_math(model,key):
    """Avoid HF's unconditional fp32 RMSNorm/softmax casts during precision audits."""
    import importlib
    import types
    import torch
    module_name={'llama3':'llama','olmo3':'olmo3','qwen3':'qwen3'}[key]
    modeling=importlib.import_module(f'transformers.models.{module_name}.modeling_{module_name}')
    count=0
    def norm_forward(self,hidden_states):
        assert hidden_states.dtype==torch.float64
        variance=hidden_states.square().mean(-1,keepdim=True)
        return self.weight*(hidden_states*torch.rsqrt(variance+self.variance_epsilon))
    for module in model.modules():
        if module.__class__.__name__.endswith('RMSNorm') and hasattr(module,'variance_epsilon'):
            module.forward=types.MethodType(norm_forward,module);count+=1
    def attention(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
        assert query.dtype==torch.float64
        keys=modeling.repeat_kv(key,module.num_key_value_groups)
        values=modeling.repeat_kv(value,module.num_key_value_groups)
        weights=torch.matmul(query,keys.transpose(2,3))*scaling
        if attention_mask is not None:weights=weights+attention_mask
        weights=torch.nn.functional.softmax(weights,dim=-1,dtype=query.dtype)
        weights=torch.nn.functional.dropout(weights,p=dropout,training=module.training)
        return torch.matmul(weights,values).transpose(1,2).contiguous(),weights
    modeling.eager_attention_forward=attention
    return dict(rmsnorm_modules=count,softmax_dtype='float64',rope='same fixed probe-position constants')

def run_validation(a):
    """Frozen score on fresh disjoint document-pairs; checkpoint each scored pair.

    Gaussian uses the ORIGINAL full covariance. Each shuffled pair is sampled as
    two distinct reference rows independently per coordinate, the exact two-row
    marginal of an independently shuffled reference pool. Different pairs use
    independent draws, avoiding shared-permutation dependence between pairs.
    Refinement depends ONLY on vector-derivative convergence, never class/AUC.
    """
    import torch
    import transformers
    import pyarrow.parquet as pq
    from .great_circle_norm_grid import MODELS as HF_MODELS,candidate_positions
    from .great_circle_synthetic_grid import BLOCKS
    from ..core.trajectory_activation_cache import extract_cached
    from ..core.pair_barrier_geometry import analyze,speed_from_vectors

    assert a.pairs>=2
    key=a.validate_model
    root=a.output or a.data_root/'pair_barrier_validation_gpu'
    root.mkdir(parents=True,exist_ok=True)
    checkpoint=root/'checkpoints'/key;checkpoint.mkdir(parents=True,exist_ok=True)
    reference_path=a.data_root/f'great_circle_synthetic_grid/{key}.npz'
    reference_meta=reference_path.with_suffix('.json')
    reference_info=json.loads(reference_meta.read_text())
    dataset=Path(reference_info['dataset'])
    assert file_hash(dataset)==reference_info['dataset_sha256']
    documents=pq.read_table(dataset,columns=['text'])['text'].to_pylist()
    excluded_rows,excluded_texts=pair_exclusions(documents)
    exclusion_sources={'algorithm': 'core.records.pair_exclusions: exact shared archived document draws',
                       'helper_sha256': file_hash(Path(__file__).parents[1]/'core/records.py')}
    base_step=.25 if key=='qwen3' else 1.
    contract=dict(version='heldout-downhill-barrier-v1',model=HF_MODELS[key],snapshot=reference_info['snapshot'],
        source_reference_sha256=file_hash(reference_path),source_metadata_sha256=file_hash(reference_meta),
        exclusion_sources=exclusion_sources,pairs=a.pairs,seed=a.seed,block_start=BLOCKS[key],block_count=4,
        classes=['natural','gaussian','shuffled'],score='log(peak/higher downhill minimum), zero if no separating peak',
        initial_step_degrees=base_step,convergence_relative_l2_limit=.05,convergence_p99_limit=.1,max_refinements=3,
        downhill_helper_sha256=file_hash(pair_barrier_geometry.__file__),
        shuffled_sampling='independent two-row marginal of reference-coordinate permutations per pair')
    if a.precision=='float64' or a.audit_validation is not None:
        assert a.audit_validation is not None,'Precision audits must replay frozen validation inputs'
        contract.update(precision=a.precision,audit_source=str(a.audit_validation),
            audit_prepared_sha256=file_hash(a.audit_validation/f'checkpoints/{key}/prepared.npz'))
    contract_hash=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    final=root/f'{key}.json'
    if final.exists() and (root/f'{key}.npz').exists():
        assert json.loads(final.read_text())['contract_sha256']==contract_hash
        print(f'Already complete: {final}',flush=True);return
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tokenizer=transformers.AutoTokenizer.from_pretrained(reference_info['snapshot'])
    compute_dtype=torch.float64 if a.precision=='float64' else torch.float32
    model=transformers.AutoModel.from_pretrained(reference_info['snapshot'],dtype=compute_dtype,
        device_map=a.device,attn_implementation='eager').eval();model.requires_grad_(False)
    precision_math=enable_float64_math(model,key) if a.precision=='float64' else None
    with np.load(reference_path) as d:
        reference=d['reference'].astype(np.float64)
        mean=d['reference_mean'];factor=d['covariance_factor']
    assert reference.shape[0]==256
    np.testing.assert_allclose(mean,reference.mean(0),rtol=0,atol=1e-12)
    np.testing.assert_allclose(factor,(reference-mean)/np.sqrt(255),rtol=0,atol=1e-12)
    model_seed=a.seed+1000*(MODELS.index(key)+1)
    prepared=checkpoint/'prepared.npz';prepared_meta=prepared.with_suffix('.json')
    if a.audit_validation is not None and not (prepared.exists() and prepared_meta.exists()):
        import shutil
        source_prepared=a.audit_validation/f'checkpoints/{key}/prepared.npz'
        original=json.loads(source_prepared.with_suffix('.json').read_text())
        original_result=json.loads((a.audit_validation/f'{key}.json').read_text())
        assert original_result['contract']['seed']==a.seed and original_result['contract']['pairs']==a.pairs
        shutil.copyfile(source_prepared,prepared)
        original.update(contract_sha256=contract_hash,replayed_preparation_sha256=file_hash(source_prepared),
                        audit_note='Exact stored float32 initial states; only forward arithmetic precision changes')
        atomic_json(prepared_meta,original)
    with torch.inference_mode():
        if prepared.exists() and prepared_meta.exists():
            preparation=json.loads(prepared_meta.read_text());assert preparation['contract_sha256']==contract_hash
            with np.load(prepared) as d:states=d['states'];pairs=d['pairs']
        else:
            dataset=Path(reference_info['dataset']);assert file_hash(dataset)==reference_info['dataset_sha256']
            documents=pq.read_table(dataset,columns=['text'])['text'].to_pylist()
            token_rng=np.random.default_rng(model_seed+1)
            records=[];seen_texts=set(excluded_texts)
            for row in np.random.default_rng(a.seed).permutation(len(documents)):
                if int(row) in excluded_rows:continue
                text_hash=hashlib.sha256(documents[int(row)].encode()).hexdigest()
                if text_hash in seen_texts:continue
                ids=tokenizer(documents[int(row)],truncation=True,max_length=256)['input_ids']
                candidates=candidate_positions(ids,tokenizer.all_special_ids)
                if not candidates:continue
                position=int(token_rng.choice(candidates))
                records.append(dict(document_row=int(row),text_sha256=text_hash,position=position,input_ids=ids))
                seen_texts.add(text_hash)
                if len(records)==2*a.pairs:break
            assert len(records)==2*a.pairs
            assert not excluded_rows.intersection(r['document_row'] for r in records)
            assert len({r['document_row'] for r in records})==len(records)
            assert len({r['text_sha256'] for r in records})==len(records)
            natural,cache_info=extract_cached(model,records,[BLOCKS[key]-1],a.device,reference_info['snapshot'],
                                              root/'activation_cache',f'{key} heldout')
            pairs=np.random.default_rng(model_seed+4).permutation(2*a.pairs).reshape(a.pairs,2)
            states=np.empty((3,2*a.pairs,reference.shape[1]),np.float32);states[0]=natural[0]
            coefficients=np.random.default_rng(model_seed+2).standard_normal((2*a.pairs,len(reference)))
            states[1]=mean+coefficients@factor
            shuffle_rng=np.random.default_rng(model_seed+3)
            first=shuffle_rng.integers(len(reference),size=(a.pairs,reference.shape[1]))
            second=shuffle_rng.integers(len(reference)-1,size=first.shape)
            second+=second>=first
            assert np.all(first!=second)
            coordinates=np.arange(reference.shape[1])[None,:]
            states[2,pairs[:,0]]=reference[first,coordinates]
            states[2,pairs[:,1]]=reference[second,coordinates]
            preparation=dict(contract_sha256=contract_hash,records=records,excluded_document_rows=sorted(excluded_rows),
                excluded_text_hashes=sorted(excluded_texts),activation_cache=cache_info,model_seed=model_seed,
                device=a.device,source_reference=str(reference_path))
            atomic_npz(prepared,states=states,pairs=pairs,gaussian_coefficients=coefficients,
                       shuffled_reference_rows=np.stack([first,second],axis=1))
            atomic_json(prepared_meta,preparation)
        assert pairs.shape==(a.pairs,2) and np.array_equal(np.sort(pairs.ravel()),np.arange(2*a.pairs))
        assert states.shape==(3,2*a.pairs,model.config.hidden_size) and np.isfinite(states).all()
        assert all(r['position']>0 and r['input_ids'][r['position']] not in tokenizer.all_special_ids for r in preparation['records'])
        start=BLOCKS[key]-1;end=start+4
        prefix=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        assert prefix==reference_info['probe_token_id']
        probe=torch.tensor([[prefix,prefix]],device=a.device);captured={};reference_output=[]
        def capture(li):
            def hook(module,args,kwargs):captured[li]=(args[0].clone(),kwargs.copy())
            return hook
        def output_hook(module,args,out):reference_output.append((out[0] if isinstance(out,tuple) else out).clone())
        hs=[model.layers[li].register_forward_pre_hook(capture(li),with_kwargs=True) for li in range(start,end)]
        hs.append(model.layers[end-1].register_forward_hook(output_hook))
        model(probe,use_cache=False)
        for h in hs:h.remove()
        def forward(inp):
            for li in range(start,end):
                out=model.layers[li](inp,**captured[li][1]);inp=out[0] if isinstance(out,tuple) else out
            return inp
        baseline=forward(captured[start][0]);reference_error=float((baseline-reference_output[0]).abs().max())
        assert reference_error<1e-5
        results=[];scores=np.empty((3,a.pairs));fine_scores=np.empty_like(scores)
        for ci,name in enumerate(contract['classes']):
            for pi,(ai,bi) in enumerate(pairs):
                path=checkpoint/f'{name}-{pi:04d}.npz';info_path=path.with_suffix('.json')
                if path.exists() and info_path.exists():
                    info=json.loads(info_path.read_text());assert info['contract_sha256']==contract_hash
                    scores[ci,pi],fine_scores[ci,pi]=info['score'],info['half_step_score'];results.append(info)
                    continue
                x_np,b_np=states[ci,ai].astype(np.float64),states[ci,bi].astype(np.float64)
                xhat_np=x_np/np.linalg.norm(x_np);bhat=b_np/np.linalg.norm(b_np)
                tangent=b_np-(b_np@xhat_np)*xhat_np
                assert np.linalg.norm(tangent)>np.linalg.norm(b_np)*1e-8
                tangent/=np.linalg.norm(tangent)
                partner_angle=float(np.rad2deg(np.arccos(np.clip(xhat_np@bhat,-1,1))))
                np.testing.assert_allclose(np.cos(np.deg2rad(partner_angle))*xhat_np+np.sin(np.deg2rad(partner_angle))*tangent,bhat,atol=1e-10,rtol=1e-10)
                x=torch.tensor(states[ci,ai],device=a.device,dtype=compute_dtype);radius=x.norm();xhat=x/radius
                u=torch.tensor(tangent,device=a.device,dtype=compute_dtype)
                assert abs(float(xhat@u))<1e-5
                def evaluate(degrees):
                    rad=np.deg2rad(degrees)
                    cos=torch.tensor(np.cos(rad),device=a.device,dtype=compute_dtype)
                    sin=torch.tensor(np.sin(rad),device=a.device,dtype=compute_dtype)
                    circle=radius*(cos[:,None]*xhat+sin[:,None]*u)
                    assert torch.allclose(circle.norm(dim=-1),radius.expand(len(rad)),rtol=1e-5)
                    outputs=[]
                    for offset in range(0,len(rad),a.batch):
                        points=circle[offset:offset+a.batch]
                        inp=captured[start][0].expand(len(points),-1,-1).clone();inp[:,1]=points
                        outputs.append(forward(inp)[:,1].cpu().numpy())
                    return np.concatenate(outputs)
                h=base_step;grid=np.arange(round(360/h))*h;vectors=evaluate(grid)
                batch_error=float(np.max(np.abs(evaluate(np.array([0.]))[0]-vectors[0])))
                assert np.allclose(evaluate(np.array([0.]))[0],vectors[0],rtol=2e-4,atol=2e-4)
                injected_error=None
                if pi==0:
                    reference_output.clear()
                    def inject(module,args):
                        replacement=args[0].clone();replacement[:,1]=x;return (replacement,*args[1:])
                    hs=[model.layers[start].register_forward_pre_hook(inject),model.layers[end-1].register_forward_hook(output_hook)]
                    model(probe,use_cache=False)
                    for handle in hs:handle.remove()
                    injected_reference=reference_output[0][0,1].cpu().numpy()
                    injected_error=float(np.max(np.abs(injected_reference-vectors[0])))
                    assert np.allclose(injected_reference,vectors[0],rtol=2e-4,atol=2e-4)
                for refinement in range(4):
                    fine=np.empty((2*len(grid),x.numel()),vectors.dtype)
                    fine[::2],fine[1::2]=vectors,evaluate(grid+h/2)
                    speed=speed_from_vectors(vectors.astype(np.float64),h,1)
                    fine_speed=speed_from_vectors(fine.astype(np.float64),h/2,1)
                    relative=np.abs(speed-fine_speed[::2])/np.maximum(fine_speed[::2],1e-12)
                    relative_l2=float(np.linalg.norm(speed-fine_speed[::2])/np.linalg.norm(fine_speed[::2]))
                    p99=float(np.quantile(relative,.99))
                    if relative_l2<.05 and p99<.1:break
                    if refinement==3:raise ValueError(f'{key} {name} pair{pi}: derivative did not converge: {relative_l2}, {p99}')
                    vectors=fine;h/=2;grid=np.arange(len(vectors))*h
                coarse_result=analyze(speed,h,partner_angle);fine_result=analyze(fine_speed,h/2,partner_angle)
                def strength(r):return float(np.log(r['peak_over_higher_minimum'])) if r['minimum_peak_minimum'] else 0.
                score,fine_score=strength(coarse_result),strength(fine_result)
                info=dict(contract_sha256=contract_hash,class_name=name,pair_index=pi,anchor_index=int(ai),partner_index=int(bi),
                    partner_angle=partner_angle,anchor_radius=float(np.linalg.norm(x_np)),partner_radius=float(np.linalg.norm(b_np)),
                    step_degrees=h,refinements=refinement,relative_l2=relative_l2,relative_p99=p99,
                    score=score,half_step_score=fine_score,analysis=coarse_result,half_step_analysis=fine_result,
                    batch_max_abs=batch_error,full_model_injection_max_abs=injected_error,device=a.device)
                atomic_npz(path,theta=grid,speed=speed,half_theta=np.arange(len(fine_speed))*h/2,half_speed=fine_speed)
                atomic_json(info_path,info)
                scores[ci,pi],fine_scores[ci,pi]=score,fine_score;results.append(info)
                if (pi+1)%8==0:print(f'{key}: {name} {pi+1}/{a.pairs} pairs checkpointed',flush=True)
        assert np.isfinite(scores).all() and (scores>=0).all()
        comparisons={}
        for ci,name in enumerate(contract['classes'][1:],start=1):
            comparisons[name]=dict(auc=auc(scores[0],scores[ci]),
                conditional_bootstrap_95=bootstrap_auc(scores[0],scores[ci],model_seed+100+ci),
                half_step_auc=auc(fine_scores[0],fine_scores[ci]))
        result=dict(model=HF_MODELS[key],contract=contract,contract_sha256=contract_hash,classes=contract['classes'],
            scores=scores.tolist(),comparisons=comparisons,samples_per_class=a.pairs,endpoint_documents=2*a.pairs,
            block_start=BLOCKS[key],block_count=4,heldout=True,preparation=preparation,
            score='log(peak/higher downhill minimum), zero if no separating peak',
            direction='higher means natural; frozen before heldout generation',locality_weighting=False,cutoff=None,
            uncertainty='pair bootstrap conditional on original fixed reference fit; no reference-fit uncertainty',
            source_model_snapshot=reference_info['snapshot'],dataset_sha256=reference_info['dataset_sha256'],
            tests=dict(document_and_text_disjointness=True,no_endpoint_reuse_within_model=True,nonspecial_positive_positions=True,
                original_fit_preserved=True,independent_shuffled_pair_marginals=True,checkpoint_per_pair=True,
                full_model_reference_max_abs=reference_error,all_pair_numerical_convergence=True),
            max_refinements=max(r['refinements'] for r in results),
            max_score_half_step_abs=float(np.max(np.abs(scores-fine_scores))),pair_checkpoints=str(checkpoint),
            torch=torch.__version__,transformers=transformers.__version__,device=a.device,
            precision=a.precision,precision_math=precision_math,slurm_job=os.getenv('SLURM_JOB_ID'),
            script_sha256=file_hash(__file__))
        atomic_npz(root/f'{key}.npz',states=states,pairs=pairs,scores=scores,half_step_scores=fine_scores,classes=contract['classes'])
        atomic_json(final,result)
        print(f'{key}: VALIDATION COMPLETE {comparisons}',flush=True)

def collect_validation(a):
    root=a.output or a.data_root/'pair_barrier_validation_gpu'
    results={key:json.loads((root/f'{key}.json').read_text()) for key in MODELS}
    assert all(r['heldout'] for r in results.values())
    summary=dict(score='log(peak/higher downhill minimum), zero if no separating peak',score_kind='barrier-strength',
        direction='higher means natural; frozen before heldout generation',cutoff=None,locality_weighting=False,
        heldout=True,excluded_controls=['norm-matched'],models=results,
        limitation='128 disjoint endpoint-pairs/class/model; bootstrap conditional on original fitted reference; no reference-fit uncertainty')
    for key,result in results.items():
        with np.load(root/f'{key}.npz') as d:
            scores=d['scores'];half=d['half_step_scores']
        result['barrier_counts']=(scores>0).sum(axis=1).tolist()
        result['half_step_barrier_counts']=(half>0).sum(axis=1).tolist()
        result['barrier_proportions']=(scores>0).mean(axis=1).tolist()
        result['incidence_definition']='B>0: distinct downhill minima separated by a higher intervening peak; no amplitude cutoff'
    atomic_json(root/'summary.json',summary)
    if a.paper_export is not None:
        destination=a.paper_export;destination.mkdir(parents=True,exist_ok=True)
        for key,result in results.items():
            n=result['samples_per_class'];classes=result['classes'];theta=np.arange(-200,201)
            displayed=np.empty((len(classes),n,401),np.float64)
            angles=np.empty((len(classes),n));steps=np.empty_like(angles)
            profiles=[];half_profiles=[];offsets=[0];half_offsets=[0]
            for ci,name in enumerate(classes):
                for pi in range(n):
                    path=root/f'checkpoints/{key}/{name}-{pi:04d}.npz'
                    info=json.loads(path.with_suffix('.json').read_text())
                    with np.load(path) as d:
                        speed=d['speed'];fine=d['half_speed'];h=info['step_degrees']
                        displayed[ci,pi]=speed[np.rint((theta%360)/h).astype(int)]
                        profiles.append(speed);half_profiles.append(fine)
                        offsets.append(offsets[-1]+len(speed));half_offsets.append(half_offsets[-1]+len(fine))
                    angles[ci,pi]=info['partner_angle'];steps[ci,pi]=h
            with np.load(root/f'{key}.npz') as d:
                atomic_npz(destination/f'{key}.npz',theta=theta,classes=classes,speeds=displayed,
                    block=result['block_start'],block_end=result['block_start']+result['block_count']-1,
                    block_count=result['block_count'],partner_angles=angles,pair_steps=steps,
                    step_degrees=result['contract']['initial_step_degrees'],states=d['states'],pairs=d['pairs'],
                    scores=d['scores'],half_step_scores=d['half_step_scores'],
                    profile_speeds=np.concatenate(profiles),profile_offsets=offsets,
                    half_profile_speeds=np.concatenate(half_profiles),half_profile_offsets=half_offsets)
            exported=dict(result,export_source=str(root),source_result_sha256=file_hash(root/f'{key}.json'),
                profile_layout='class-major/pair-major flattened periodic speeds; offsets mark profiles; half profiles use half the pair_steps')
            atomic_json(destination/f'{key}.json',exported)
        atomic_json(destination/'summary.json',summary)
    for key,result in results.items():print(key,result['comparisons'],'barriers',result['barrier_counts'],'half-step',result['half_step_barrier_counts'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate-model',choices=MODELS)
    p.add_argument('--collect-validation',action='store_true')
    p.add_argument('--paper-export',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--data-root',type=Path,default=Path('data'))
    p.add_argument('--precision',choices=['float32','float64'],default='float32')
    p.add_argument('--audit-validation',type=Path)
    p.add_argument('--pairs',type=int,default=128)
    p.add_argument('--seed',type=int,default=20260915)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--batch',type=int,default=128)
    a=p.parse_args()
    if a.validate_model:run_validation(a)
    elif a.collect_validation:collect_validation(a)
    else:p.error('Choose --validate-model or --collect-validation')
