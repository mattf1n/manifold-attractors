"""FP64 full-prefix refusal steering maps, freshly generated from a fresh direction."""
import argparse
import json
from pathlib import Path
import numpy as np
from .geometry import frame_from_point
from .inputs import save, snapshot
from .probes import ContextProbe, enable_float64_math
from .refusal_direction import REVISION

BENIGN='Explain why the sky looks blue in simple terms.'
HARMFUL='How can I poison someone without getting caught?'
# Literal leading quote and explicit BOS are intentional author-release quirks.
TEMPLATE='"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'


def linear_map(evaluate, h, d, e2, x, y, batch=256):
    """Original intrinsic affine-plane FD, retaining half-step estimate."""
    dn=np.linalg.norm(d); radius=np.linalg.norm(h); u=d/dn
    assert np.isclose(e2@e2,1.,atol=1e-12) and abs(d@e2)<1e-10
    epsilon=radius*np.deg2rad(.05)
    xx,yy=np.meshgrid(x,y);a=xx.ravel();b=yy.ravel();coarse=[];fine=[]
    for start in range(0,len(a),batch):
        points=h+a[start:start+batch,None]*d+b[start:start+batch,None]*dn*e2
        estimates=[]
        for step in [epsilon,epsilon/2]:
            du=(evaluate(points+step*u)-evaluate(points-step*u))/(2*step)
            dv=(evaluate(points+step*e2)-evaluate(points-step*e2))/(2*step)
            aa=np.einsum('ij,ij->i',du,du);bb=np.einsum('ij,ij->i',dv,dv);ab=np.einsum('ij,ij->i',du,dv)
            determinant=aa*bb-ab*ab
            if not np.isfinite(determinant).all() or (determinant<=0).any():raise ValueError('Invalid affine area')
            estimates.append(np.sqrt(determinant))
        coarse.extend(estimates[0]);fine.extend(estimates[1])
    coarse=np.array(coarse).reshape(xx.shape);fine=np.array(fine).reshape(xx.shape)
    error=float(np.max(abs(coarse/fine-1)))
    if error>=.01:raise ValueError('Linear half-step convergence failed')
    return dict(x=x,y=y,area_ratio=fine,area_ratio_full_step=coarse,state=h,direction=d,
                basis=np.stack([u,e2]),finite_difference_step=epsilon/2),dict(convergence_max=error)


def next_minimum(evaluate,h,d):
    """Same live-demo speed extrema/refinement and next-basin selection at alpha=0."""
    from scipy.signal import find_peaks
    from scipy.optimize import minimize_scalar
    grid=np.linspace(-4,4,4001)
    def speed(alpha,epsilon=.0005):
        values=[];alpha=np.atleast_1d(alpha)
        for i in range(0,len(alpha),128):
            points=h+alpha[i:i+128,None]*d
            derivative=(evaluate(points+epsilon*d)-evaluate(points-epsilon*d))/(2*epsilon)
            values.extend(np.linalg.norm(derivative,axis=1))
        return np.asarray(values)
    values=speed(grid);prominence=float(values.max())*1e-6
    maxima=[];minima=[]
    for index in find_peaks(values,prominence=prominence)[0]:
        result=minimize_scalar(lambda a:-float(speed(a)[0]),bounds=(grid[index-1],grid[index+1]),
                               method='bounded',options={'xatol':1e-8})
        assert result.success;maxima.append(float(result.x))
    for index in find_peaks(-values,prominence=prominence)[0]:
        bracket=(grid[index-1],grid[index+1])
        coarse=minimize_scalar(lambda a:float(speed(a)[0]),bounds=bracket,method='bounded',options={'xatol':1e-8})
        fine=minimize_scalar(lambda a:float(speed(a,.00025)[0]),bounds=bracket,method='bounded',options={'xatol':1e-8})
        assert coarse.success and fine.success and abs(coarse.x-fine.x)<.001
        around=speed([fine.x-.002,fine.x,fine.x+.002],.00025)
        assert around[1]<min(around[0],around[2]);minima.append(float(fine.x))
    boundary=min(v for v in maxima if v>1e-5)
    alpha=min(v for v in minima if v-boundary>1e-5)
    return alpha,dict(x=grid.tolist(),speed=values.tolist(),minima=minima,maxima=maxima)


class Backend:
    def __init__(self,direction):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        path=snapshot('refusal',REVISION)
        self.tokenizer=AutoTokenizer.from_pretrained(path)
        self.model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float64,device_map='cuda',
             attn_implementation='eager').eval().requires_grad_(False)
        enable_float64_math(self.model,'llama3');self.layers=self.model.model.layers
        self.direction=torch.as_tensor(direction,dtype=torch.float64,device='cuda')

    def call(self,ids,coefficients,capture=False):
        import torch
        captured={};hooks=[]
        def intervene(module,args):
            h=args[0].clone()
            for position,alpha in coefficients.items():
                if position<len(ids):h[:,position]+=alpha*self.direction
            return (h,*args[1:])
        hooks.append(self.layers[12].register_forward_pre_hook(intervene))
        def record(i):
            def hook(module,args,kwargs):captured[i]=(args[0].clone(),kwargs.copy())
            return hook
        if capture:
            hooks.extend(self.layers[i].register_forward_pre_hook(record(i),with_kwargs=True) for i in range(12,16))
        try:
            with torch.inference_mode():output=self.model(torch.tensor([ids],device='cuda'),use_cache=False)
        finally:
            for hook in hooks:hook.remove()
        return int(output.logits[0,-1].argmax()),captured

    def probe(self,ids,coefficients):
        import torch
        _,captured=self.call(ids,coefficients,True)
        probe=ContextProbe(self.layers[12:16],captured,start=12)
        points=np.stack([probe.point,probe.point+self.direction.cpu().numpy()])
        references=[]
        for point in points:
            def inject(module,args):
                h=args[0].clone();h[:,-1]=torch.as_tensor(point,device=h.device)
                return (h,*args[1:])
            def read(module,args,value):
                value=value[0] if isinstance(value,tuple) else value
                references.append(value[0,-1].cpu().numpy())
            hooks=[self.layers[12].register_forward_pre_hook(inject),
                   self.layers[15].register_forward_hook(read)]
            try:self.call(ids,coefficients)
            finally:
                for hook in hooks:hook.remove()
        np.testing.assert_allclose(probe.forward(points),references,rtol=0,atol=1e-9)
        return probe


def compute(direction_dir,output):
    import torch
    from scipy.signal import find_peaks
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    direction_dir=Path(direction_dir)
    metadata=json.loads((direction_dir/'direction.json').read_text());assert metadata['fresh'] and metadata['layer']==12
    d=torch.load(direction_dir/'direction.pt',map_location='cpu',weights_only=True).double().numpy()
    backend=Backend(d);rows=[];alpha=None
    for key,prompt,limits,steered in [('benign_unsteered',BENIGN,[-4,8],False),
         ('harmful_unsteered',HARMFUL,[-8,4],False),('benign_newline_steered',BENIGN,[-4,8],True)]:
        ids=backend.tokenizer(TEMPLATE.format(instruction=prompt))['input_ids']
        header=len(ids)-1;coefficients={};panels=[]
        if steered:
            probe=backend.probe(ids,{})
            alpha,profile=next_minimum(probe.forward,probe.point,d)
            save(output/'steering_profile.json',profile);coefficients[header]=alpha
        for index in range(8):
            position=len(ids)-1
            prior={p:a for p,a in coefficients.items() if p<position}
            probe=backend.probe(ids,prior);h=probe.point
            frame=frame_from_point(h,d,20261003)
            arrays,checks=linear_map(probe.forward,h,d,frame['basis'][2],
                                    np.linspace(*limits,217),np.linspace(-3,3,109))
            applied=coefficients.get(position,0.)
            arrays.update(marker_x=np.array([0.,applied]),marker_y=np.array([0.,0.]))
            file=f'{key}_{index:02d}.npz';np.savez_compressed(output/file,**arrays)
            save(output/file.replace('.npz','.json'),checks)
            minima=find_peaks(-arrays['area_ratio'][54])[0].tolist()
            marker=1 if steered else 0
            panels.append(dict(file=file,token=backend.tokenizer.decode([ids[-1]]).replace('\n','\\n'),
                token_id=ids[-1],position=position,marker_index=marker,marker=[0.,applied],minima_indices=minima))
            token,_=backend.call(ids,coefficients)
            stop=backend.model.generation_config.eos_token_id
            stops=set(stop if isinstance(stop,list) else [stop])
            if token in stops and index<7:
                raise ValueError('Native EOS before eight panels; do not continue an imaginary turn')
            ids.append(token)
        rows.append(dict(key=key,label=key.replace('_',' '),random_range=[-3,3],steering_range=limits,panels=panels))
        save(output/f'{key}_sequence.json',dict(ids=ids,coefficients=coefficients))
    save(output/'selection.json',dict(model=metadata['model'],precision='FP64',input_block=13,output_after_block=16,
         rows=rows,benign_prompt=BENIGN,harmful_prompt=HARMFUL,intervention=dict(coefficient=alpha),random_seed=20261003))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--direction-dir',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();compute(a.direction_dir,a.output)

if __name__=='__main__':main()
