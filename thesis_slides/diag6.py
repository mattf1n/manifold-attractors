import torch
T0, A0 = torch.tensor, torch.as_tensor
import split
from manifold_attractors.maps.refusal import Backend, BENIGN, TEMPLATE
d=torch.load("/home1/mfinlays/manifold-attractors/data/refusal_direction/direction.pt",map_location="cpu",weights_only=True).double().numpy()
b=Backend(d)
ids=b.tokenizer(TEMPLATE.format(instruction=BENIGN))["input_ids"]
def top():
    with torch.inference_mode(): out=b.model(T0([ids]),use_cache=False)
    return out.logits[0,-1].topk(3).indices.tolist()
print("wrapped", top())
torch.tensor, torch.as_tensor = T0, A0
print("restored", top())
import transformers.models.llama.modeling_llama as m
print("rope", b.model.model.rotary_emb.inv_freq[:4], b.model.model.rotary_emb.inv_freq.dtype, getattr(b.model.model.rotary_emb, "attention_scaling", None))
print("dir", b.direction.dtype, b.direction.device)
