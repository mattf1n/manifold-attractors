import torch, importlib, types
from transformers import AutoModelForCausalLM, AutoTokenizer
from manifold_attractors.maps.inputs import snapshot
from manifold_attractors.maps.refusal import BENIGN, TEMPLATE
from manifold_attractors.maps.refusal_direction import REVISION
from manifold_attractors.maps.probes import enable_float64_math
m=importlib.import_module("transformers.models.llama.modeling_llama")
path=snapshot("refusal",REVISION); tok=AutoTokenizer.from_pretrained(path)
ids=tok(TEMPLATE.format(instruction=BENIGN))["input_ids"]
model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float64,attn_implementation="eager").eval().requires_grad_(False)
def top():
    with torch.inference_mode(): out=model(torch.tensor([ids]),use_cache=False)
    l=out.logits[0,-1]; return l.topk(3).indices.tolist(), round(l.max().item(),4)
orig_attn=m.eager_attention_forward
print("none", top())
print(enable_float64_math(model,"llama3")); print("both", top())
patched=m.eager_attention_forward; m.eager_attention_forward=orig_attn; print("norm only", top())
for mod in model.modules():
    if "forward" in mod.__dict__ and mod.__class__.__name__.endswith("RMSNorm"): del mod.forward
m.eager_attention_forward=patched; print("attn only", top())
import inspect; print(inspect.signature(orig_attn)); print(inspect.getsource(orig_attn)[:1500])
