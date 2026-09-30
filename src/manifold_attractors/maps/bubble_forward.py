"""Minimal original pullback_solve forward helpers (no inversion solver)."""
from typing import Callable
import torch
from torch import Tensor
from transformers.cache_utils import DynamicCache
Forward=Callable[[Tensor],Tensor]

def _hook_cat_last(H):
    """Full forward: replace the LAST position's layer output with H (keep the context)."""
    def h(m, i, o):
        out = o[0] if isinstance(o, tuple) else o
        new = torch.cat([out[:, :-1, :], H.to(out.dtype)[:, None, :]], dim=1)
        return (new,) + o[1:] if isinstance(o, tuple) else new
    return h

def _hook_single(H):
    """Fast forward: the only (decode) token's layer output IS the injected state H."""
    def h(m, i, o):
        out = o[0] if isinstance(o, tuple) else o
        new = H.to(out.dtype)[:, None, :]
        return (new,) + o[1:] if isinstance(o, tuple) else new
    return h

def make_forward(model, ids: Tensor, inject_layer: int, out_layer: int, fast: bool = True) -> Forward:
    """Build F(H): inject H (B,d) at the last position of layer (inject_layer-1) under the
    `ids` context, return the out_layer last-token output (B,d). Autograd-enabled.

    fast=True (default): prefill context KV once, decode only the injected token (exact,
    much faster, scales with batch not sequence length).
    fast=False: re-forward the whole sequence each call (reference implementation).
    """
    dev = ids.device

    if not fast:
        def F(H: Tensor) -> Tensor:
            hh = model.model.layers[inject_layer - 1].register_forward_hook(_hook_cat_last(H))
            try:
                return model(ids.repeat(H.shape[0], 1), output_hidden_states=True,
                             use_cache=False).hidden_states[out_layer][:, -1, :].float()
            finally:
                hh.remove()
        return F

    if DynamicCache is None:
        raise RuntimeError("DynamicCache unavailable in this transformers version; pass fast=False")
    with torch.no_grad():
        pf = model(ids[:, :-1], use_cache=True)
    layers = getattr(pf.past_key_values, "layers", None)
    if not layers:
        raise RuntimeError("fast=True: prefill produced no cache layers; pass fast=False")
    # A truncated model (layers cut to out_layer+1) leaves trailing None layer slots in
    # the cache (sized to config.num_hidden_layers); take only the populated leading run.
    K, V = [], []
    for lyr in layers:
        if lyr is None or getattr(lyr, "keys", None) is None:
            break
        K.append(lyr.keys.detach().clone()); V.append(lyr.values.detach().clone())
    L = int(ids.shape[1])
    last = ids[:, -1:]

    def F(H: Tensor) -> Tensor:
        B = H.shape[0]
        c = DynamicCache(config=model.config)
        for i in range(len(K)):
            c.update(K[i].expand(B, -1, -1, -1).contiguous(),
                     V[i].expand(B, -1, -1, -1).contiguous(), i)
        hh = model.model.layers[inject_layer - 1].register_forward_hook(_hook_single(H))
        try:
            return model(last.expand(B, 1), past_key_values=c, use_cache=True,
                         output_hidden_states=True,
                         cache_position=torch.arange(L - 1, L, device=dev)
                         ).hidden_states[out_layer][:, -1, :].float()
        finally:
            hh.remove()
    return F

@torch.no_grad()
def forward_batched(F: Forward, X: Tensor, bs: int = 512) -> Tensor:
    """F over X (N,d) in chunks of bs (no grad)."""
    out = torch.empty((X.shape[0], X.shape[1]), device=X.device, dtype=torch.float32)
    for s in range(0, X.shape[0], bs):
        e = min(s + bs, X.shape[0])
        out[s:e] = F(X[s:e])
    return out
