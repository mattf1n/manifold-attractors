"""Fresh English/Chinese steering circles and complete sampled token regions."""
import argparse
import json
from pathlib import Path
import numpy as np
from .geometry import regions, tangent
from .inputs import prepare, save, load


def speed_from_vectors(vectors, step):
    derivative = (np.roll(vectors,-1,axis=0)-np.roll(vectors,1,axis=0))/(2*np.deg2rad(step))
    return np.linalg.norm(derivative,axis=-1)


def crop_regions(ids, step, tokenizer):
    output=[]
    for region in regions(ids,step):
        for shift in [-360,0,360]:
            lo=max(-90,region['start']+shift); hi=min(90,region['end']+shift)
            if hi>lo:
                tid=region['token_id']
                output.append(dict(start=lo,end=hi,token_id=tid,text=tokenizer.decode([tid]),
                                   token_piece=tokenizer.convert_ids_to_tokens(tid)))
    output.sort(key=lambda r:r['start'])
    assert output[0]['start']==-90 and output[-1]['end']==90
    assert all(a['end']==b['start'] for a,b in zip(output,output[1:]))
    return output


def source_labels(suffix, all_regions, contexts):
    """Apply frozen editorial inputs only after recomputing regions; never invent labels."""
    labels=load('language_speed_labels'+suffix)
    for row in all_regions:
        for region in row:
            key=str(region['token_id'])
            if key not in labels['tokens']:
                raise ValueError(f'No source editorial label for token {key}; exact figure is blocked')
            entry=labels['tokens'][key]
            if entry['text']!=region['text'] or entry['token_piece']!=region['token_piece']:
                raise ValueError(f'Source editorial label/token mismatch for {key}; exact figure is blocked')
    annotations=labels.get('interlinear_contexts',{})
    if annotations:
        import re
        source={str(c['index']):c for c in contexts}
        if set(annotations)!=set(source):
            raise ValueError('Source interlinear annotations do not match the current selected quartet')
        for key,annotation in annotations.items():
            units=annotation['units']
            highlighted=[u['highlight'] for u in units if u.get('highlight')]
            if highlighted!=[source[key]['target_token']]:
                raise ValueError(f'Interlinear target mismatch for source {key}')
            shown=''.join(u['source'] for u in units)
            normalize=lambda text:re.sub(r'[\W_]+','',text)
            if not normalize(source[key]['prefix_through_target']).endswith(normalize(shown)):
                raise ValueError(f'Interlinear context mismatch for source {key}')
    return labels


def compute(output, device='cuda'):
    from .fixed_probe import FixedProbe
    output=Path(output); meta=json.loads((output/'inputs.json').read_text())
    with np.load(output/'inputs.npz') as z:
        states=z['states'].copy(); en=z['centroid_en'].copy(); zh=z['centroid_zh'].copy()
    direction=(zh-en)/np.linalg.norm(zh-en)
    probe=FixedProbe(meta['snapshot'],device=device)
    for language, subset in [('english',states[:5]),('chinese',states[5:])]:
        speeds=[]; tokens=[]; tangents=[]; all_regions=[]; validations=[]; theta=None
        for index,native in enumerate(subset):
            x=native.astype(np.float64); radius=np.linalg.norm(x); pole=x/radius
            # English originals were evaluated toward EN, then exactly reversed at export.
            sign=-1 if language=='english' else 1
            u,_=tangent(sign*direction,x); tangents.append(sign*u)
            def points(degrees):
                a=np.deg2rad(degrees)
                return radius*(np.cos(a)[:,None]*pole+np.sin(a)[:,None]*u)
            h=1.; vectors=probe.forward(points(np.arange(360)))
            for refinement in range(4):
                fine=np.empty((2*len(vectors),len(x)),dtype=np.float64)
                fine[::2]=vectors
                fine[1::2]=probe.forward(points(np.arange(len(vectors))*h+h/2))
                coarse_speed=speed_from_vectors(vectors,h); fine_speed=speed_from_vectors(fine,h/2)
                error=coarse_speed-fine_speed[::2]
                l2=float(np.linalg.norm(error)/np.linalg.norm(fine_speed[::2]))
                p99=float(np.quantile(abs(error)/np.maximum(fine_speed[::2],1e-12),.99))
                if l2<.05 and p99<.1: break
                if refinement==3: raise ValueError('Derivative failed convergence')
                vectors=fine; h/=2
            step=h/2
            if language=='chinese' and step!=.5:
                raise ValueError('Chinese published protocol requires .5 degree grid')
            ids=probe.predict(points(np.arange(len(fine))*step))
            check=dict(relative_l2=l2,relative_p99=p99,step_degrees=step,full_hook_error=probe.audit(x))
            if language=='chinese':
                quarter_theta=np.arange(1440)/4
                quarter=speed_from_vectors(probe.forward(points(quarter_theta)),.25)
                quarter_ids=probe.predict(points(quarter_theta))
                np.testing.assert_array_equal(ids,quarter_ids[::2])
                err=fine_speed-quarter[::2]
                ql2=float(np.linalg.norm(err)/np.linalg.norm(quarter[::2]))
                qp99=float(np.quantile(abs(err)/np.maximum(quarter[::2],1e-12),.99))
                assert ql2<.05 and qp99<.1
                check.update(quarter_relative_l2=ql2,quarter_relative_p99=qp99)
                np.savez_compressed(output/f'chinese_curve_{index:02d}.npz',quarter_theta=quarter_theta,
                    quarter_speed=quarter,quarter_token_ids=quarter_ids,fine_speed=fine_speed,
                    fine_token_ids=ids,coarse_speed=coarse_speed,
                    source_index=load('language_speed_chinese')['indices'][index])
            else:
                reverse=(-np.arange(len(ids)))%len(ids)
                fine_speed=fine_speed[reverse]; ids=ids[reverse]
            current_theta=np.arange(round(180/step)+1)*step-90
            if theta is not None: np.testing.assert_array_equal(theta,current_theta)
            theta=current_theta; crop=np.rint((theta%360)/step).astype(int)%len(ids)
            speeds.append(fine_speed[crop]); tokens.append(ids[crop]); validations.append(check)
            all_regions.append(crop_regions(ids,step,probe.tokenizer))
        suffix='' if language=='english' else '_chinese'
        source=load('language_speed'+suffix)
        np.savez_compressed(output/f'llama3{suffix}.npz',theta=theta,speeds=speeds,token_ids=tokens,
            states=subset,direction=direction,tangents=tangents,source_indices=source['indices'],
            centroid_en=en,centroid_zh=zh,step_degrees=step)
        save(output/f'regions{suffix}.json',all_regions)
        save(output/f'validation{suffix}.json',dict(examples=validations))
        contexts=[]
        for source_index,record in zip(source['indices'],source['records']):
            position=record['position'];ids=record['input_ids']
            contexts.append(dict(index=source_index,record=record,text=record['text'],
                target_position=position,target_id=ids[position],target_token=probe.tokenizer.decode([ids[position]]),
                prefix_through_target=probe.tokenizer.decode(ids[:position+1])))
        save(output/f'selection{suffix}.json',dict(selected_indices=source['indices'],contexts=contexts))
        save(output/f'glosses{suffix}.json',source_labels(suffix,all_regions,contexts))
    save(output/'provenance.json',dict(model=meta['model'],revision=meta['revision'],blocks=[16,17,18,19],
         context='Fixed two BOS; both positions evolve. NOT source-document decoding.',
         speed='Raw four-block output central difference per radian; no final norm',
         tokens='All remaining blocks, final normalization, vocabulary argmax',fresh_recomputation=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','compute','all']);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    if a.stage in ['prepare','all']:prepare('speed',a.output,a.device)
    if a.stage in ['compute','all']:compute(a.output,a.device)

if __name__=='__main__':main()
