"""Fresh author-commit refusal extraction/selection, no published vector reuse.
Run in the separate pinned historical environment (MAPS.md).
"""
import argparse
import ast
import json
import random
import subprocess
import sys
from pathlib import Path

COMMIT='0b9c8bc56a73b42f595fd173156d4aefe9a9143c'
REVISION='8afb486c1db24fe5011ec46dfbe5b5dccdb575c2'
MODEL='meta-llama/Meta-Llama-3-8B-Instruct'

def extract_functions(path,names,namespace):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--author-repo',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(); repo=a.author_repo.resolve(); a.output.mkdir(parents=True,exist_ok=True)
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==COMMIT
    sys.path.insert(0,str(repo))
    import torch
    from huggingface_hub import snapshot_download
    from dataset.load_dataset import load_dataset_split
    from pipeline.config import Config
    from pipeline.model_utils.llama3_model import Llama3Model
    from pipeline.submodules.generate_directions import generate_directions
    from pipeline.submodules.select_direction import get_refusal_scores, select_direction
    from .inputs import save
    snapshot=snapshot_download(MODEL,revision=REVISION)
    cfg=Config(model_alias='meta-llama-3-8b-instruct',model_path=snapshot)
    functions=extract_functions(repo/'pipeline/run_pipeline.py',['load_and_sample_datasets','filter_data'],
        dict(torch=torch,random=random,load_dataset_split=load_dataset_split,get_refusal_scores=get_refusal_scores))
    torch.set_num_threads(4)
    original=functions['load_and_sample_datasets'](cfg)
    model=Llama3Model(snapshot)
    assert all(p.dtype==torch.bfloat16 for p in model.model.parameters())
    filtered=functions['filter_data'](cfg,model,*original)
    save(a.output/'sources.json',dict(original=original,filtered=filtered))
    candidates=generate_directions(model,filtered[0],filtered[1],str(a.output/'generate_directions'))
    position,layer,direction=select_direction(model,filtered[2],filtered[3],candidates,str(a.output/'select_direction'))
    # Refuse to silently change the paper's input block.
    assert layer==12, f'Expected input block13, fresh selection chose {layer+1}'
    torch.save(direction.cpu(),a.output/'direction.pt')
    save(a.output/'direction.json',dict(position=position,layer=layer,model=MODEL,revision=REVISION,
         author_commit=COMMIT,extraction='BF16 native activations; FP64 means; raw selected vector',
         fresh=True,no_harmful_completions_generated=True))

if __name__=='__main__':main()
