"""Content-addressed native-document activation extraction, truncated causally.

Only tokens through the sampled position and blocks before the last requested
INPUT state are needed. Verify against full-document/full-model evaluation on
two records; serialize atomically and verify exact cache round-trip. Cache keys
include snapshots, full token records, layers, dtype, device and library versions.
"""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


class ExtractionComplete(Exception):
    pass


def extract_cached(model,records,layers,device,snapshot,cache_dir,label):
    import torch
    import transformers
    manifest=dict(schema=1,snapshot=str(snapshot),records=records,layers=list(layers),device=device,
                  dtype='float32',torch=torch.__version__,transformers=transformers.__version__,
                  attention='eager',truncation='tokens<=sample_position; stop before last requested block')
    key=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    path=Path(cache_dir)/f'{key}.npz'; meta_path=path.with_suffix('.json')
    def extract_one(record,fast):
        values={}; handles=[]
        def capture(li):
            def hook(module,args):values[li]=args[0][0,record['position']].cpu().numpy().copy()
            return hook
        for li in layers:handles.append(model.layers[li].register_forward_pre_hook(capture(li)))
        if fast:
            def stop(module,args):raise ExtractionComplete()
            handles.append(model.layers[layers[-1]].register_forward_pre_hook(stop))
        ids=record['input_ids'][:record['position']+1] if fast else record['input_ids']
        ids=torch.tensor([ids],device=device)
        try:
            model(ids,attention_mask=torch.ones_like(ids),use_cache=False)
        except ExtractionComplete:
            if not fast:raise
        finally:
            for h in handles:h.remove()
        assert set(values)==set(layers)
        return np.stack([values[li] for li in layers])
    started=time.monotonic()
    if path.exists() and meta_path.exists():
        info=json.loads(meta_path.read_text())
        assert info['key']==key
        assert hashlib.sha256(path.read_bytes()).hexdigest()==info['npz_sha256']
        with np.load(path) as data:values=data['values']
        # Verify a true disk-cache hit against fresh evaluation, not only a checksum.
        fresh=extract_one(records[0],True)
        np.testing.assert_allclose(values[:,0],fresh,rtol=2e-4,atol=2e-4)
        info.update(cache_hit=True,cache_hit_recompute_max_abs=float(np.max(np.abs(values[:,0]-fresh))),
                    load_and_verify_seconds=time.monotonic()-started)
        print(f'{label}: activation cache HIT {key[:12]} ({len(records)} records)',flush=True)
        return values,info
    values=np.empty((len(layers),len(records),model.config.hidden_size),np.float32)
    checks=[]
    for i,record in enumerate(records):
        values[:,i]=extract_one(record,True)
        if i<2:
            full=extract_one(record,False)
            error=float(np.max(np.abs(values[:,i]-full)))
            np.testing.assert_allclose(values[:,i],full,rtol=2e-4,atol=2e-4)
            checks.append(dict(record=i,full_vs_truncated_max_abs=error))
        if (i+1)%32==0:print(f'{label}: cached/truncated extraction {i+1}/{len(records)}',flush=True)
    assert np.isfinite(values).all()
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f'{key}.{os.getpid()}.tmp.npz')
    np.savez_compressed(temporary,values=values)
    with np.load(temporary) as data:assert np.array_equal(data['values'],values)
    temporary.replace(path)
    info=dict(key=key,path=str(path),manifest=manifest,cache_hit=False,full_vs_truncated_checks=checks,
              atomic_round_trip_exact=True,compute_seconds=time.monotonic()-started,
              npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    temporary=meta_path.with_name(f'{key}.{os.getpid()}.tmp.json')
    temporary.write_text(json.dumps(info,indent=2)+'\n');temporary.replace(meta_path)
    # Exercise the public cache-hit path once immediately to verify fresh reload.
    reloaded,hit_info=extract_cached(model,records,layers,device,snapshot,cache_dir,label)
    assert np.array_equal(reloaded,values)
    info['cache_hit_recompute_max_abs']=hit_info['cache_hit_recompute_max_abs']
    return values,info
