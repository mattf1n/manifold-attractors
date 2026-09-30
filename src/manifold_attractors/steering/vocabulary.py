"""Rebuild the complete vocabulary partition, never load archived labels."""
import codecs
import json
import unicodedata
from pathlib import Path
import numpy as np
from .common import atomic_json, atomic_npz
from .probe import Probe
LEGACY_METRIC = "han_only_v0"
CJK_METRIC = "han_cjk_punctuation_v1"
HAN_PREFIX_METRIC = "han_cjk_punctuation_utf8_prefix_v2"
SEED = 20261002
CLASSES = (
    "Latin",
    "Neutral punctuation/symbols",
    "Chinese",
    "Other alphabet",
    "Other nonalphabetic",
)
SUBTYPES = (
    "Latin",
    "Neutral punctuation/symbols",
    "Han",
    "CJK punctuation",
    "Han UTF-8 prefix",
    "Other alphabet",
    "Whitespace",
    "Numbers",
    "Special/meta",
    "Unresolved UTF-8",
    "Residual other",
)
COARSE_FROM_SUBTYPE = np.array([0, 1, 2, 2, 2, 3, 4, 4, 4, 4, 4], dtype=np.int8)
BAD = (0, 3)
CHINESE = 2

def token_class(token_id, tokenizer):
    if token_id in tokenizer.all_special_ids:
        return "Other nonalphabetic"
    text = tokenizer.decode([token_id])
    if "\ufffd" in text:
        return "Other nonalphabetic"
    letters = [char for char in text if char.isalpha()]
    if letters:
        if all("CJK" in unicodedata.name(char, "") for char in letters):
            return "Chinese (Han)"
        if all("LATIN" in unicodedata.name(char, "") for char in letters):
            return "Latin"
        return "Other alphabet"
    chars = [char for char in text if not char.isspace()]
    if chars and all(unicodedata.category(char)[0] in "PS" for char in chars):
        return "Punctuation/symbols"
    return "Other nonalphabetic"

def metric_labels(version):
    if version == LEGACY_METRIC:
        return "Chinese (Han)", "Punctuation/symbols"
    if version in (CJK_METRIC, HAN_PREFIX_METRIC):
        return "Chinese", "Neutral punctuation/symbols"
    raise ValueError(f"Unknown token metric: {version}")


def is_cjk_punctuation(text):
    return any(
        unicodedata.category(char).startswith("P")
        and (
            unicodedata.east_asian_width(char) in ("W", "F")
            or "\uff61" <= char <= "\uff65"
        )
        for char in text
    )


def han_prefix_checker(tokenizer):
    """Recognize incomplete ByteLevel UTF-8 prefixes with only Han completions.

    Continuation-only bytes are deliberately left unresolved: native source bytes
    are not part of the actual fixed-BOS probe. No context-based inference here.
    """
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codepoints = byte_values.copy()
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    inverse = {chr(code): byte for byte, code in zip(byte_values, codepoints)}
    ranges = [
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
        (0xF900, 0xFAFF),
        (0x20000, 0x2A6DF),
        (0x2A700, 0x2B73F),
        (0x2B740, 0x2B81F),
        (0x2B820, 0x2CEAF),
        (0x2CEB0, 0x2EBEF),
        (0x2F800, 0x2FA1F),
        (0x30000, 0x3134F),
    ]

    def check(token_id):
        if token_id in tokenizer.all_special_ids:
            return False
        piece = tokenizer.convert_ids_to_tokens(token_id)
        try:
            raw = bytes(inverse[char] for char in piece)
        except KeyError:
            return False
        assert raw.decode("utf8", errors="replace") == tokenizer.decode([token_id])
        decoder = codecs.getincrementaldecoder("utf8")()
        try:
            complete = decoder.decode(raw, final=False)
        except UnicodeDecodeError:
            return False
        pending = decoder.getstate()[0]
        if not pending or "\ufffd" in complete:
            return False
        if any(
            char.isalpha() and "CJK" not in unicodedata.name(char, "")
            for char in complete
        ):
            return False
        length = 2 if pending[0] < 0xE0 else 3 if pending[0] < 0xF0 else 4
        value = pending[0] & ((1 << (7 - length)) - 1)
        for byte in pending[1:]:
            value = (value << 6) | (byte & 63)
        bits = 6 * (length - len(pending))
        low = value << bits
        high = low + (1 << bits) - 1
        if not any(start <= low <= high <= end for start, end in ranges):
            return False
        return all(
            unicodedata.name(chr(code), "").startswith(
                ("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")
            )
            for code in range(low, high + 1)
        )

    return check


def classify_cached(tokenizer, version=LEGACY_METRIC):
    chinese, punctuation = metric_labels(version)
    cache = {}
    han_prefix = han_prefix_checker(tokenizer) if version == HAN_PREFIX_METRIC else None

    def classify(token_id):
        if token_id not in cache:
            category = token_class(token_id, tokenizer)
            # Preserve the original ordering: special/invalid tokens and script
            # classes are resolved before punctuation. Emoji alone are not CJK punctuation.
            if version in (CJK_METRIC, HAN_PREFIX_METRIC):
                if category == "Chinese (Han)" or (
                    han_prefix is not None
                    and category == "Other nonalphabetic"
                    and "\ufffd" in tokenizer.decode([token_id])
                    and han_prefix(token_id)
                ):
                    category = chinese
                elif category == "Punctuation/symbols":
                    category = (
                        chinese
                        if is_cjk_punctuation(tokenizer.decode([token_id]))
                        else punctuation
                    )
            cache[token_id] = category
        return cache[token_id]

    return classify

def read(path):
    return json.loads(path.read_text())


def prepare_subtypes(args):
    """Refine diagnostics without changing the agreed five-way partition."""
    import unicodedata

    from transformers import AutoTokenizer

    path = args.output / "vocabulary_subtypes.npz"
    if path.exists():
        return
    definition = read(args.output / "definition.json")
    tokenizer = AutoTokenizer.from_pretrained(definition["snapshot"])
    with np.load(args.output / "vocabulary.npz") as data:
        coarse = data["categories"].copy()
        special = set(data["special_ids"].tolist())
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codes = byte_values.copy()
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codes.append(256 + extra)
            extra += 1
    inverse = {chr(code): byte for byte, code in zip(byte_values, codes)}
    labels = np.empty(len(coarse), dtype=np.int8)
    for index, category in enumerate(coarse):
        if category == 0:
            label = "Latin"
        elif category == 1:
            label = "Neutral punctuation/symbols"
        elif category == 3:
            label = "Other alphabet"
        elif category == 2:
            previous = token_class(index, tokenizer)
            label = {
                "Chinese (Han)": "Han",
                "Punctuation/symbols": "CJK punctuation",
            }.get(previous, "Han UTF-8 prefix")
        elif index in special:
            label = "Special/meta"
        else:
            text = tokenizer.decode([index])
            chars = [char for char in text if not char.isspace()]
            if not chars:
                label = "Whitespace"
            elif any(unicodedata.category(char)[0] == "N" for char in chars) and all(
                unicodedata.category(char)[0] in "NPS" for char in chars
            ):
                label = "Numbers"
            elif "\ufffd" in text:
                raw = bytes(
                    inverse[char] for char in tokenizer.convert_ids_to_tokens(index)
                )
                try:
                    raw.decode("utf8", errors="strict")
                    label = "Residual other"  # A literal replacement character, not broken UTF-8.
                except UnicodeDecodeError:
                    label = "Unresolved UTF-8"
            else:
                label = "Residual other"
        labels[index] = SUBTYPES.index(label)
    np.testing.assert_array_equal(COARSE_FROM_SUBTYPE[labels], coarse)
    atomic_npz(
        path,
        categories=labels,
        subtype_names=SUBTYPES,
        coarse_from_subtype=COARSE_FROM_SUBTYPE,
    )
    definition.update(
        subtypes=SUBTYPES,
        subtype_counts={
            name: int((labels == i).sum()) for i, name in enumerate(SUBTYPES)
        },
        subtype_coarse_partition_exact=True,
        numeric_subtype="Number-only or number+punctuation/symbol tokens with no letters",
        unresolved_subtype="Strict UTF-8 decoding fails; literal replacement characters fall in residual other",
    )
    atomic_json(args.output / "definition.json", definition)
    print("Diagnostic subtype partition:", np.bincount(labels).tolist(), flush=True)


def aggregate_subtypes(masses):
    return np.stack(
        [
            masses[..., COARSE_FROM_SUBTYPE == i].sum(axis=-1)
            for i in range(len(CLASSES))
        ],
        axis=-1,
    )


def load_partition(args, probe, subtypes=False):
    import torch

    filename = "vocabulary_subtypes.npz" if subtypes else "vocabulary.npz"
    names = SUBTYPES if subtypes else CLASSES
    with np.load(args.output / filename) as data:
        categories = data["categories"].copy()
        assert data["subtype_names" if subtypes else "class_names"].tolist() == list(
            names
        )
    assert len(categories) == probe.model.config.vocab_size
    matrix = torch.eye(len(names), dtype=torch.float64, device=probe.device)[
        torch.as_tensor(categories, dtype=torch.long, device=probe.device)
    ]
    return categories, matrix


def checked_distribution(probe, points, matrix, expected=None):
    masses, ids = probe.probability_mass(points, matrix)
    assert masses.shape == (len(points), matrix.shape[1])
    assert (
        np.isfinite(masses).all()
        and np.all(masses >= 0)
        and np.all(masses <= 1 + 1e-12)
    )
    np.testing.assert_allclose(masses.sum(1), 1, rtol=0, atol=2e-12)
    if expected is not None:
        np.testing.assert_array_equal(ids, expected)
    return masses, ids


def audit_distribution(probe, point, matrix, expected_mass, expected_id):
    """Compare batched/manual propagation against an actual full-model injection."""
    import torch

    def inject(module, inputs):
        hidden = inputs[0].clone()
        hidden[:, 1] = torch.as_tensor(point, dtype=torch.float64, device=probe.device)
        return (hidden, *inputs[1:])

    with torch.inference_mode():
        handle = probe.model.model.layers[probe.start].register_forward_pre_hook(inject)
        try:
            logits = probe.model(probe.input_ids, use_cache=False).logits[0, 1]
            reference = (
                (torch.softmax(logits, dim=-1, dtype=torch.float64) @ matrix)
                .cpu()
                .numpy()
            )
            reference_id = int(logits.argmax())
        finally:
            handle.remove()
    assert reference_id == int(expected_id)
    np.testing.assert_allclose(reference, expected_mass, rtol=1e-10, atol=2e-12)
    single, _ = checked_distribution(
        probe, np.asarray(point)[None], matrix, [expected_id]
    )
    np.testing.assert_allclose(single[0], expected_mass, rtol=1e-10, atol=2e-12)
    return {
        "full_injection_mass_max_abs": float(np.max(abs(reference - expected_mass))),
        "scalar_batch_mass_max_abs": float(np.max(abs(single[0] - expected_mass))),
        "argmax_exact": True,
    }


def bootstrap_means(values, draws=10000, seed=SEED):
    """Common paired resampling indices for every comparison; bounded-memory batches."""
    random = np.random.default_rng(seed)
    estimates = np.empty((draws, values.shape[1]))
    for start in range(0, draws, 100):
        size = min(100, draws - start)
        indices = random.integers(len(values), size=(size, len(values)))
        for column in range(values.shape[1]):
            estimates[start : start + size, column] = values[:, column][indices].mean(
                axis=1
            )
    return np.quantile(estimates, [0.025, 0.975], axis=0).T
