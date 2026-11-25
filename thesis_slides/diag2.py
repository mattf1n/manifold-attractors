import split, torch, json, numpy as np
from manifold_attractors.maps.refusal import Backend, BENIGN, TEMPLATE
seq=json.load(open("/home1/mfinlays/manifold-attractors/data/refusal_steering_heatmaps/benign_unsteered_sequence.json"))["ids"]
d=torch.load("/home1/mfinlays/manifold-attractors/data/refusal_direction/direction.pt",map_location="cpu",weights_only=True).double().numpy()
b=Backend(d)
ids=b.tokenizer(TEMPLATE.format(instruction=BENIGN))["input_ids"]
print("ids match", ids==seq[:len(ids)], len(ids))
print("call", b.call(ids,{})[0])
with torch.inference_mode(): out=b.model(torch.tensor([ids]),use_cache=False)
print("plain", out.logits[0,-1].argmax().item())
p=b.probe(ids,{}); print("after probe call", b.call(ids,{})[0])
with torch.inference_mode(): out=b.model(torch.tensor([ids]),use_cache=False)
print("after probe plain", out.logits[0,-1].argmax().item(), torch.isnan(out.logits).any().item())
for k in range(4): ids2=seq[:len(ids)+k]; print(k, b.call(ids2,{})[0], seq[len(ids2)] if len(ids2)<len(seq) else None)
print("inv_freq", b.model.model.rotary_emb.inv_freq[:3])
