"""Device placement only (arithmetic unchanged, FP64 everywhere). The released Backend loads the FP64 model (64 GB) with
device_map='cuda'.
- GPU smaller than 70 GB: the embedding and blocks 1-16 (including the probed blocks 13-16 and the steering hook at block 13)
  go on cuda:0, the remaining blocks, final norm and LM head on cuda:1 if there is a second GPU, otherwise on the CPU.
- No GPU: everything on the CPU; torch.tensor/as_tensor(device='cuda') in the Backend are redirected to the CPU. The redirect
  is suspended while the model loads (active during loading it left the rotary inv_freq buffer uninitialized: garbage output)."""
import torch, transformers
_orig = transformers.AutoModelForCausalLM.from_pretrained.__func__

if torch.cuda.is_available():
    def from_pretrained(cls, path, *args, **kwargs):
        need = 70e9 if kwargs.get('dtype') == torch.float64 else 40e9   # FP64 weights 64 GB, FP32 32 GB, plus room for activations
        if kwargs.get('device_map') == 'cuda' and torch.cuda.get_device_properties(0).total_memory < need:
            rest = 1 if torch.cuda.device_count() >= 2 else 'cpu'
            dm = {'model.embed_tokens': 0, 'model.rotary_emb': 0, 'model.norm': rest, 'lm_head': rest}
            dm.update({f'model.layers.{i}': 0 if i < 16 else rest for i in range(32)})
            kwargs['device_map'] = dm
            print('device_map:', dm, flush=True)
        return _orig(cls, path, *args, **kwargs)
else:
    _T0, _A0 = torch.tensor, torch.as_tensor
    def _cpu(fn):
        def wrapped(*args, **kwargs):
            if kwargs.get('device') == 'cuda': kwargs['device'] = 'cpu'
            return fn(*args, **kwargs)
        return wrapped
    _T1, _A1 = _cpu(_T0), _cpu(_A0)
    torch.tensor, torch.as_tensor = _T1, _A1
    def from_pretrained(cls, path, *args, **kwargs):
        if kwargs.get('device_map') == 'cuda': kwargs.pop('device_map')
        torch.tensor, torch.as_tensor = _T0, _A0
        try: return _orig(cls, path, *args, **kwargs)
        finally: torch.tensor, torch.as_tensor = _T1, _A1
transformers.AutoModelForCausalLM.from_pretrained = classmethod(from_pretrained)
