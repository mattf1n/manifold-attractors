"""FP64 patches and conditional prefix replay, preserved from original producers."""
import numpy as np

from ..experiments.pair_barrier_validation import enable_float64_math

class Replay:
    def __init__(self,values):self.values=values
    def update(self,k,v,i,*args,**kwargs):
        import torch
        a,b=self.values[i]
        return torch.cat((a.expand(len(k),-1,-1,-1),k),-2),torch.cat((b.expand(len(v),-1,-1,-1),v),-2)

class Recorder:
    def __init__(self):self.values={}
    def update(self,k,v,i,*args,**kwargs):self.values[i]=(k[...,:-1,:].clone(),v[...,:-1,:].clone());return k,v

class ContextProbe:
    def __init__(self,layers,captured,start=15):
        import torch
        self.layers=layers;self.start=start;self.microbatch=128; self.kwargs={}; recorder=Recorder()
        hidden=captured[start][0].clone();self.device=hidden.device;self.point=hidden[0,-1].cpu().numpy()
        with torch.inference_mode():
            for i,layer in enumerate(layers,start):
                original=captured[i][1].copy();original['past_key_values']=recorder;original['use_cache']=True
                hidden=layer(hidden,**original)
                if isinstance(hidden,tuple):hidden=hidden[0]
                single=original.copy()
                for name in ['position_ids','cache_position']:
                    if single.get(name) is not None:single[name]=single[name][...,-1:]
                if single.get('position_embeddings') is not None:single['position_embeddings']=tuple(v[...,-1:,:] for v in single['position_embeddings'])
                if single.get('attention_mask') is not None:single['attention_mask']=single['attention_mask'][...,-1:,:]
                self.kwargs[i]=single
        self.reference=hidden[0,-1].cpu().numpy();self.cache=Replay(recorder.values)
        for kw in self.kwargs.values():kw['past_key_values']=self.cache
    def forward(self,points):
        import torch
        out=[]
        with torch.inference_mode():
            for offset in range(0,len(points),self.microbatch):
                h=torch.as_tensor(points[offset:offset+self.microbatch],dtype=torch.float64,device=self.device)[:,None,:]
                for i,l in enumerate(self.layers,self.start):
                    h=l(h,**self.kwargs[i]);h=h[0] if isinstance(h,tuple) else h
                out.append(h[:,0].cpu().numpy())
        return np.concatenate(out)
