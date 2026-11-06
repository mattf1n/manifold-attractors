"""Prepare frozen source texts, pinned weights and entirely fresh FP32 states.

No Wikimedia/Wikidata API, archived activations, archived partition or fitted
strength is used. The bundled excerpts are INPUTS selected before inference.
"""
import argparse
import gc
import gzip
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .common import (DEFAULT_MODEL, DEFAULT_ROOT, MODELS, SIZES, atomic_json, atomic_npz,
                     digest, read, freeze_run)

INPUTS = Path(__file__).parent / "inputs"
LIMITATIONS = [
    "Frozen source selections reproduce the practical historical identity contract; live identity APIs are not rerun.",
    "Historical translation mappings are unavailable for some source articles; no perfect historical family-isolation claim.",
    "All 52 English articles lacking current QIDs remained frozen document exclusions in original source selection.",
    "Canonical ZH397 has no verified EN counterpart; absence of detected overlap is not proof of absence.",
    "Current QIDs/sitelinks recorded at selection may differ from 2023 identities.",
    "Lexical/mirror filters cannot certify absence of translated/paraphrased FineWeb copies.",
]


def source_manifests():
    import json
    metadata = read(INPUTS / "sources.json")
    result = {}
    for split, size in SIZES.items():
        specification = metadata["splits"][split]
        path = INPUTS / specification["file"]
        if digest(path) != specification["sha256"]:
            raise ValueError(f"Bundled source input checksum mismatch: {split}")
        manifest = json.loads(gzip.decompress(path.read_bytes()))
        if manifest["split"] != split or len(manifest["records"]) != size:
            raise ValueError("Wrong frozen split")
        result[split] = manifest
    return result


def validate_sources(manifests, tokenizer):
    """Replay tokenizer and original position RNG, not outcome-based resampling."""
    specials = set(tokenizer.all_special_ids) | {
        i for i, token in tokenizer.added_tokens_decoder.items() if token.special
    }
    if len(specials) != 256:
        raise ValueError("Expected the pinned Llama vocabulary's 256 special IDs")
    random = np.random.default_rng(20261022)
    documents, contexts, groups = set(), set(), set()
    for split in SIZES:
        records = manifests[split]["records"]
        encoded = []
        for start in range(0, len(records), 256):
            encoded.extend(tokenizer([r["text"] for r in records[start:start+256]],
                                     truncation=True, max_length=128)["input_ids"])
        for index, (row, ids) in enumerate(zip(records, encoded, strict=True)):
            assert row["sample_id"] == index and row["split"] == split
            assert hashlib.sha256(row["text"].encode()).hexdigest() == row["text_sha256"]
            assert len(row["text"]) <= 1200 and ids == row["input_ids"]
            positions = [i for i, token in enumerate(ids) if i > 0 and token not in specials]
            assert row["position"] == int(random.choice(positions))
            key = (manifests[split]["dataset"], row["document_id"])
            assert key not in documents and tuple(ids) not in contexts
            assert row["article_group"] not in groups
            documents.add(key); contexts.add(tuple(ids)); groups.add(row["article_group"])
    return {"passed": True, "counts": SIZES, "positions_replayed": sum(SIZES.values()),
            "exact_tokens_verified": True, "selected_known_identity_groups_disjoint": True,
            "limitations": LIMITATIONS}


def prepare_partition(root, snapshot, tokenizer, target="zh"):
    from .vocabulary import (CLASSES, TARGETS, class_names, classify_cached,
                             greek_classify_cached, prepare_subtypes)
    folder = root / "vocabulary"
    name, _, _, metric = TARGETS[target]
    names = class_names(target)
    classify = (greek_classify_cached(tokenizer) if target == "el"
                else classify_cached(tokenizer, metric))
    specials = set(tokenizer.all_special_ids) | {
        i for i, token in tokenizer.added_tokens_decoder.items() if token.special
    }
    assert len(tokenizer) == 128256 and len(specials) == 256
    # Each classifier names its target class by the target's class name.
    labels = CLASSES[:2] + (name,) + CLASSES[3:]
    categories = np.array([labels.index("Other nonalphabetic" if i in specials else classify(i))
                           for i in range(len(tokenizer))], dtype=np.int8)
    atomic_npz(folder / "vocabulary.npz", categories=categories,
               class_names=names, special_ids=sorted(specials))
    atomic_json(folder / "definition.json", {
        "snapshot": snapshot, "temperature": 1, "token_metric": metric,
        "class_names": names, "fresh_from_tokenizer": True,
        "loss": f"Latin plus Other alphabet; not one minus {name}", "target": target,
    })
    # Regenerate rather than accepting an old label array in the work directory.
    (folder / "vocabulary_subtypes.npz").unlink(missing_ok=True)
    prepare_subtypes(SimpleNamespace(output=folder))


def prepare(root, inputs_only=False, target="zh", model=DEFAULT_MODEL):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    root = Path(root)
    stamp = freeze_run(root, model)
    spec = MODELS[model]
    MODEL, REVISION, LAYER = spec["model"], spec["revision"], spec["layer"]
    if (root / "prepared.json").exists():
        from .common import require_prepared
        require_prepared(root)
        if read(root / "vocabulary/definition.json").get("target", "zh") != target:
            raise ValueError("Prepared root has a different --target; use a new --root")
        return
    # Pinned public weights; HF credential/license acceptance may be required.
    snapshot = snapshot_download(MODEL, revision=REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken"],
        ignore_patterns=["original/*"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    manifests = source_manifests()
    audit = validate_sources(manifests, tokenizer)
    for split, manifest in manifests.items():
        manifest.update(snapshot=snapshot, input_layer_index=LAYER)
        atomic_json(root / f"manifests/{split}.json", manifest)
    atomic_json(root / "split_audit.json", audit)
    prepare_partition(root, snapshot, tokenizer, target)
    if inputs_only:
        return  # Deliberately NOT prepared; no inference completion claim.

    import torch
    from transformers import AutoModel
    from ..maps.inputs import public_corpus, centroids
    from ..core.trajectory_activation_cache import extract_cached
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    corpora = {language: public_corpus("speed", language) for language in ("en", target)}
    # This is the 200+200 speed/confirmatory corpus, NOT the area corpus.
    atomic_json(root / "direction/source_texts.json", corpora)
    network = AutoModel.from_pretrained(snapshot, dtype=torch.float32, device_map="cuda",
                                     attn_implementation="eager").eval().requires_grad_(False)
    centers = centroids(network, tokenizer, corpora, index=LAYER, device="cuda")
    raw = centers["en"] - centers[target]
    unit = raw / np.linalg.norm(raw)
    atomic_npz(root / "direction/llama3.npz", direction=unit, raw_direction=raw,
               centroid_en=centers["en"], **{f"centroid_{target}": centers[target]})
    atomic_json(root / "direction/llama3.json", {
        "snapshot": snapshot, "model": MODEL, "revision": REVISION,
        "hidden_state_index": LAYER, "max_tokens": 128, "batch": 4,
        "pooling": "All attention_mask=1 tokens including special tokens; token weighted",
        "precision": "FP32 model, FP64 accumulation; unrounded FP64 centroid difference",
        "stored_sign": f"en-minus-{target}; evaluation negates it", "fresh": True,
        "target": target,
    })
    approval = {key: True for key in ("approved", "split_audit_passed", "protocol_frozen",
                                     "direction_provenance_approved")}
    approval.update(manifest_sha256={s: digest(root / f"manifests/{s}.json") for s in SIZES},
                    evaluation_datasets=["wiki_test", "fineweb_test"],
                    source="Verified frozen source INPUT selections; fresh direction and vocabulary")
    atomic_json(root / "inference_approval.json", approval)
    for split, manifest in manifests.items():
        for shard in range(SIZES[split] // 500):
            records = manifest["records"][shard*500:(shard+1)*500]
            with torch.inference_mode():
                values, checks = extract_cached(
                    network, records, [LAYER], "cuda", snapshot, root / "activation_cache",
                    f"{split}/{shard:03d}",
                )
            path = root / f"states/{split}/{shard:03d}.npz"
            atomic_npz(path, states=values[0], sample_ids=[r["sample_id"] for r in records])
            atomic_json(path.with_suffix(".json"), {
                "manifest_sha256": approval["manifest_sha256"][split],
                "approval_sha256": digest(root / "inference_approval.json"),
                "native_dtype": "float32", "hidden_state_index": LAYER,
                "count": len(records), "extraction": checks,
            })
    del network
    gc.collect()
    torch.cuda.empty_cache()
    files = ["inference_approval.json", "split_audit.json", "direction/llama3.npz",
             "direction/llama3.json", "direction/source_texts.json", "vocabulary/vocabulary.npz",
             "vocabulary/vocabulary_subtypes.npz", "vocabulary/definition.json"]
    files += [f"manifests/{split}.json" for split in SIZES]
    files += [f"states/{split}/{shard:03d}.{extension}" for split, size in SIZES.items()
              for shard in range(size // 500) for extension in ("npz", "json")]
    atomic_json(root / "prepared.json", {
        "complete": True, "run_sha256": stamp, "counts": SIZES,
        "artifacts_sha256": {name: digest(root / name) for name in files},
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--inputs-only", action="store_true", help="No GPU inference; does not complete prepare")
    p.add_argument("--target", choices=["zh", "el"], default="zh",
                   help="Steering target language (direction and vocabulary partition)")
    p.add_argument("--model", choices=list(MODELS), default=DEFAULT_MODEL)
    args = p.parse_args()
    prepare(args.root, args.inputs_only, args.target, args.model)


if __name__ == "__main__":
    main()
