"""Reproduce reference/heldout sampling without any generated-data dependency.

Derived only from paper_scripts/great_circle_synthetic_grid.py and
norm_growth_classifier.py. Token selection has a model-specific RNG; document
selection does not. Pair exclusions depend on document identities, not tokens.
"""
import hashlib
import numpy as np
from .inputs import MODELS


def candidate_positions(input_ids, special_ids):
    return [i for i, token in enumerate(input_ids) if i > 0 and token not in special_ids]


def heldout_records(documents, tokenizer, key, reference_records, samples=128, seed=20260910):
    reference_rows = {r['document_row'] for r in reference_records}
    eligible_rows = sorted(set(range(len(documents))) - reference_rows)
    order = np.random.default_rng(seed).permutation(eligible_rows)
    rng = np.random.default_rng(seed + 1000 * (list(MODELS).index(key) + 1))
    records = []
    for row in order:
        ids = tokenizer(documents[int(row)], truncation=True, max_length=256)['input_ids']
        candidates = candidate_positions(ids, tokenizer.all_special_ids)
        if not candidates:
            continue
        position = int(rng.choice(candidates))
        records.append(dict(document_row=int(row), position=position, input_ids=ids,
                            token=tokenizer.decode([ids[position]]),
                            text_sha256=hashlib.sha256(documents[int(row)].encode()).hexdigest()))
        if len(records) == samples:
            break
    assert len(records) == samples
    return records


def pair_exclusions(documents):
    """Exact union of the twelve archived input groups, generated from seeds.

    All three models have identical document rows in those archived groups.
    The 128 heldout documents have eligible tokens for all three tokenizers
    (verified in the metadata/record tests). Their selection is the first 128
    entries of the classifier's shared document permutation. No heldout
    activations, scores, or archived record lists are used here.
    """
    fit = np.random.default_rng(20260909).choice(len(documents), 256, replace=False)
    eligible = sorted(set(range(len(documents))) - set(fit))
    evaluation = np.random.default_rng(20260910).permutation(eligible)[:128]
    norm = np.random.default_rng(20260908).choice(len(documents), 20, replace=False)
    speed = np.random.default_rng(20260910).choice(len(documents), 20, replace=False)
    rows = set(map(int, np.concatenate([fit, evaluation, norm, speed])))
    texts = {hashlib.sha256(documents[row].encode()).hexdigest() for row in rows}
    return rows, texts
