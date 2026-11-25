import split, torch, json, numpy as np
from manifold_attractors.maps.refusal import Backend, BENIGN, TEMPLATE
d=torch.load("/home1/mfinlays/manifold-attractors/data/refusal_direction/direction.pt",map_location="cpu",weights_only=True).double().numpy()
b=Backend(d); print(b.model.dtype, next(b.model.parameters()).device, b.model.config._attn_implementation)
ids=b.tokenizer(TEMPLATE.format(instruction=BENIGN))["input_ids"]
with torch.inference_mode(): out=b.model(torch.tensor([ids]),use_cache=False,output_hidden_states=True)
print("nan logits", torch.isnan(out.logits).any().item(), [torch.isnan(h).any().item() for h in out.hidden_states][:6])
print(out.logits[0,-1].topk(3))
