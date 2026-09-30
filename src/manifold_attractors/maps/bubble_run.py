"""Run fresh historical-v3 and supplementary bubbles, without historical caches."""
import argparse
import random
from pathlib import Path
import numpy as np
from .inputs import load,save


def source_draws(tokenizer,texts,seed,count,excluded=()):
    rng=random.Random(seed);records=[];excluded=set(excluded)
    while len(records)<count:
        index=rng.randrange(len(texts));ids=tokenizer(texts[index])['input_ids']
        if len(ids)<3:continue
        position=rng.randint(1,len(ids)-1)
        if excluded and (index in excluded or ids[position] in tokenizer.all_special_ids):continue
        records.append(dict(sentence_index=index,position=position,input_ids=ids,
            token_id=int(ids[position]),token=tokenizer.decode([ids[position]]).strip(),sentence=texts[index]))
        if excluded:excluded.update(i for i,text in enumerate(texts) if text==texts[index])
    return records


def write_table(path,rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    names=['kind','sample','token','token2','angle_deg','rin','i','fib_x','fib_y','fib_z','x','y','z']
    types=[pa.string(),pa.int32(),pa.string(),pa.string(),pa.float64(),pa.float64(),pa.int32()]+[pa.float64()]*6
    pq.write_table(pa.table({n:pa.array(c,type=t) for n,c,t in zip(names,zip(*rows),types)}),path)


def compute(output,revision=None):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from ..core.inputs import model_snapshot
    from . import bubble_coords as b
    from .bubble_forward import make_forward,forward_batched
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    # Historical v3 named a model but did not record its revision. Pin the same
    # Llama3.1 snapshot as the other paper experiments, report this limitation.
    source=model_snapshot('llama3',revision)
    torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(source);tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(source,dtype=torch.float32,device_map='cuda',attn_implementation='eager').eval()
    model.requires_grad_(False);model.model.layers=torch.nn.ModuleList(list(model.model.layers[:b.READ+1]))
    bos=tokenizer.bos_token_id or tokenizer.eos_token_id
    forward=make_forward(model,torch.tensor([[bos,bos]],device=model.device),b.INJECT,b.READ,fast=True)
    b._M.update(tok=tokenizer,m=model,dev=model.device,fb=lambda x:forward_batched(forward,x,bs=2048))
    texts=load('bubble_sentences')
    records=source_draws(tokenizer,texts,b.SEED,15)
    natural_records=source_draws(tokenizer,texts,b.SEED+100,100)
    excluded={i for i,text in enumerate(texts) if text in {r['sentence'] for r in records}}
    extra=source_draws(tokenizer,texts,20261004,2,excluded)
    save(output/'bubble_sources.json',dict(records=records,synthetic_source_records=natural_records,extra=extra,
        sentence_provenance='Historical FineWeb sentence list; original document IDs/revision/selection script missing',
        snapshot=source,revision_provenance='Original v3 did not pin a model revision'))
    def extract(records):
        result=[]
        with torch.inference_mode():
            for record in records:
                ids=torch.tensor([record['input_ids']],device=model.device)
                h=model(ids,output_hidden_states=True,use_cache=False).hidden_states[15][0,record['position']]
                result.append(h.float().cpu().numpy())
        return np.stack(result)
    embedding=extract(records);natural=extract(natural_records);extra_embedding=extract(extra)
    rng=np.random.default_rng(b.SEED)
    synthetic=np.take_along_axis(natural,rng.random(natural.shape).argsort(axis=0),axis=0)
    b.D=embedding.shape[1];b.rows.clear()
    for i in range(5):
        x=embedding[i];torch.manual_seed(4242+i);raw=torch.randn(b.D,2).numpy()
        e1,e2=b.tangent_axes_np(x,raw[:,0],raw[:,1])
        b.bubble('single',i,x,e1,e2,records[i]['token'],'',float('nan'),[x],[x],[[e1,e2]])
    for i in range(5):
        first=5+2*i
        b.pair_bubble('pair',i,embedding[first],embedding[first+1],records[first]['token'],records[first+1]['token'],9000+i)
    for i in range(5):
        b.pair_bubble('synth_pair',i,synthetic[2*i],synthetic[2*i+1],f'synth{2*i}',f'synth{2*i+1}',11000+i)
    assert len(b.rows)==60025
    write_table(output/'bubble_coords.parquet',b.rows)
    b.rows.clear();b.pair_bubble('pair',5,*extra_embedding,extra[0]['token'],extra[1]['token'],9005)
    assert len(b.rows)==4002
    write_table(output/'bubble_coords_extra.parquet',b.rows)
    save(output/'bubble_coords_extra.selection.json',dict(seed=20261004,records=extra,excluded_original_records=records))
    # Fast/full conditional probe agreement on freshly generated native anchors.
    full=make_forward(model,torch.tensor([[bos,bos]],device=model.device),15,25,fast=False)
    with torch.inference_mode():
        audit=torch.tensor(embedding[:2],device=model.device)
        a=full(audit).cpu().numpy();c=forward(audit).cpu().numpy()
    np.testing.assert_allclose(a,c,rtol=3e-4,atol=3e-4)
    save(output/'bubble_validation.json',dict(full_fast_max_abs=float(np.max(abs(a-c))),rows=60025,extra_rows=4002,
        recovered_source='cc2a8ac:paper_scripts/gen_bubble_coords.py',fresh=True,
        frame='Average of unit output images, then mean normalized per-point tangent derivatives; origin-centered'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--revision',help='Explicit model revision override (historical producer was unpinned)')
    a=p.parse_args();compute(a.output,a.revision)

if __name__=='__main__':main()
