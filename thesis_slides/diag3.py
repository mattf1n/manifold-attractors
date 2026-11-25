import split, torch, json, importlib
from manifold_attractors.maps.refusal import Backend, BENIGN, TEMPLATE
m=importlib.import_module("transformers.models.llama.modeling_llama")
d=torch.load("/home1/mfinlays/manifold-attractors/data/refusal_direction/direction.pt",map_location="cpu",weights_only=True).double().numpy()
b=Backend(d)
f=m.eager_attention_forward; seen=[]
def spy(module,q,k,v,mask,*a,**kw):
    if not seen: seen.append(1); print("mask", None if mask is None else (mask.shape, mask.dtype, mask[0,0,:3,:3]))
    return f(module,q,k,v,mask,*a,**kw)
m.eager_attention_forward=spy
ids=b.tokenizer(TEMPLATE.format(instruction=BENIGN))["input_ids"]
print("patched call", b.call(ids,{})[0])
print(b.model.model.layers[0].self_attn.config._attn_implementation, type(b.model.model.layers[0].self_attn).forward.__module__)
