import torch, numpy as np, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from manifold_attractors.maps.inputs import snapshot
from manifold_attractors.maps.refusal import BENIGN, TEMPLATE
from manifold_attractors.maps.refusal_direction import REVISION
a=torch.randn(512,4096,dtype=torch.float64); w=torch.randn(4096,4096,dtype=torch.float64)
print("gemm err", (a@w - torch.from_numpy(a.numpy()@w.numpy())).abs().max().item())
path=snapshot("refusal",REVISION); tok=AutoTokenizer.from_pretrained(path)
ids=tok(TEMPLATE.format(instruction=BENIGN))["input_ids"]
for dt,impl in [(torch.float32,"eager"),(torch.float64,"eager")]:
    m=AutoModelForCausalLM.from_pretrained(path,dtype=dt,attn_implementation=impl).eval()
    with torch.inference_mode(): out=m(torch.tensor([ids]),use_cache=False)
    print(dt, impl, out.logits[0,-1].topk(3)); del m
