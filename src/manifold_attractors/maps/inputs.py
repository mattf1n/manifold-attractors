"""Public input preparation. Bundled JSON contains source records, never activations."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path
import numpy as np

INPUTS = Path(__file__).with_name('inputs')
WIKI_REVISION = 'b04c8d1ceb2f5cd4588862100d08de323dccfbaa'
MODELS = {'llama3': 'meta-llama/Llama-3.1-8B', 'olmo3': 'allenai/OLMo-3-1025-7B',
          'refusal': 'meta-llama/Meta-Llama-3-8B-Instruct'}

def load(name):
    return json.loads((INPUTS / (name + '.json')).read_text())

def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def snapshot(key, revision):
    from huggingface_hub import snapshot_download
    return snapshot_download(MODELS[key], revision=revision)

def public_corpus(kind, language):
    """Rebuild the two DISTINCT centroid corpora; fail on changed speed corpus."""
    from datasets import load_dataset
    stream = load_dataset('wikimedia/wikipedia', f'20231101.{language}',
                          revision=WIKI_REVISION, split='train', streaming=True)
    if kind == 'speed':
        texts = []
        for row in stream:
            text = (row.get('text') or '').strip()
            if len(text) >= 200:
                texts.append(text[:1200])
            if len(texts) == 200:
                break
        expected = load('speed_corpus_hashes')[language]
        actual = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        if actual != expected:
            raise ValueError('Public Wikipedia no longer matches original centroid passages; no substitution allowed')
        return texts
    protocol = load('area_corpus_protocol')
    def norm(t): return ' '.join(unicodedata.normalize('NFKC', t).lower().split())
    def shingles(t):
        t = norm(t); return {t[i:i+5] for i in range(max(1, len(t)-4))}
    pool = []; ids = set(); texts = set()
    for number, row in enumerate(stream):
        if number >= protocol['scan_cap'] or len(pool) >= protocol['pool_target']:
            break
        text = row['text'].strip()
        if len(text) < 200: continue
        text = text[:1200]; digest = hashlib.sha256(norm(text).encode()).hexdigest()
        if str(row['id']) in ids or digest in texts: continue
        ids.add(str(row['id'])); texts.add(digest)
        pool.append(dict(text=text, title=row['title']))
    rng = np.random.default_rng(protocol['corpus_seed'] + protocol['languages'].index(language))
    chosen = []; sets = []; families = set()
    for i in rng.permutation(len(pool)):
        row = pool[int(i)]; family = re.sub(r'\d+', '<number>', norm(row['title']))
        if len(chosen) >= 200 and family in families: continue
        value = shingles(row['text'])
        if any(len(value & old) / max(1, len(value | old)) >= .8 for old in sets): continue
        chosen.append(row['text']); sets.append(value)
        if len(chosen) > 200: families.add(family)
        if len(chosen) == 204: break
    if len(chosen) != 204: raise ValueError('Insufficient public corpus; no fallback')
    return chosen[:200]

def extract_states(model, records, index, device='cuda'):
    """Native FP32, causal prefixes, raw INPUT-block states (zero-based index)."""
    import torch
    class Stop(Exception): pass
    captured = []
    def hook(module, args):
        captured.append(args[0][0, -1].detach().float().cpu().numpy().copy())
        raise Stop()
    handle = model.layers[index].register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            for record in records:
                ids = record['input_ids'][:record['position']+1]
                try: model(torch.tensor([ids], device=device), use_cache=False)
                except Stop: pass
    finally: handle.remove()
    return np.stack(captured)

def centroids(model, tokenizer, corpora, index=15, device='cuda'):
    """Original batch4 token-weighted valid-token FP64 sum of FP32 states."""
    import torch
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    result = {}
    with torch.inference_mode():
        for language, texts in corpora.items():
            total = 0.; acc = torch.zeros(model.config.hidden_size, dtype=torch.float64, device=device)
            for offset in range(0, len(texts), 4):
                encoded = tokenizer(texts[offset:offset+4], padding=True, truncation=True,
                                    max_length=128, return_tensors='pt').to(device)
                h = model(**encoded, output_hidden_states=True, use_cache=False).hidden_states[index]
                mask = encoded['attention_mask'].to(torch.float64)
                acc += (h.to(torch.float64) * mask[..., None]).sum((0, 1))
                total += float(mask.sum())
            result[language] = (acc / max(total, 1.)).cpu().numpy()
    return result

def prepare(kind, output, device='cuda'):
    """Download public weights/text, then recompute anchors and centroids."""
    import gc
    import torch
    from transformers import AutoModel, AutoTokenizer
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    key = 'olmo3' if kind == 'triple' else 'llama3'
    metadata = load({'triple':'triple', 'area':'language_area', 'speed':'language_speed'}[kind])
    records = metadata['records']
    if kind == 'speed': records += load('language_speed_chinese')['records']
    source = snapshot(key, metadata['revision'])
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModel.from_pretrained(source, dtype=torch.float32, device_map=device,
                                     attn_implementation='eager').eval().requires_grad_(False)
    states = extract_states(model, records, 15, device)
    arrays = dict(states=states)
    if kind != 'triple':
        corpora = {lang: public_corpus(kind, lang) for lang in (['en','zh','el'] if kind=='area' else ['en','zh'])}
        save(output / 'corpus.json', corpora)
        arrays.update({'centroid_'+k:v for k,v in centroids(model,tokenizer,corpora,device=device).items()})
    np.savez_compressed(output / 'inputs.npz', **arrays)
    save(output / 'inputs.json', dict(kind=kind, model=MODELS[key], revision=metadata['revision'],
         snapshot=source, records=records, state_dtype='float32', hidden_state_index=15,
         corpus_revision=WIKI_REVISION if kind!='triple' else None))
    del model; gc.collect(); torch.cuda.empty_cache()
