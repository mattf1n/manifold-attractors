"""CPU-only fresh steering checks; never read historical experimental outputs."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from manifold_attractors.steering import fixed, geometry, oracle, policies, vocabulary
from manifold_attractors.steering.common import atomic_json, atomic_npz
from manifold_attractors.steering.prepare import source_manifests


def test_frozen_source_inputs_only():
    manifests = source_manifests()
    assert {k: len(v["records"]) for k, v in manifests.items()} == {
        "wiki_cal": 1000, "wiki_test": 10000, "fineweb_test": 10000}
    documents, contexts, families = set(), set(), set()
    for split, manifest in manifests.items():
        for i, row in enumerate(manifest["records"]):
            assert row["sample_id"] == i and row["split"] == split
            assert 0 < row["position"] < len(row["input_ids"]) <= 128
            assert not {"states", "direction", "prediction", "masses", "strength", "token_ids"} & row.keys()
            identity = (manifest["dataset"], row["document_id"])
            assert identity not in documents and tuple(row["input_ids"]) not in contexts
            assert row["article_group"] not in families
            documents.add(identity); contexts.add(tuple(row["input_ids"])); families.add(row["article_group"])
    assert manifests["wiki_cal"]["revision"] == "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
    assert manifests["fineweb_test"]["revision"] == "9bb295ddab0e05d785b879661af7260fed5140fc"


def test_config_explicit_dependencies():
    root = Path(__file__).parents[1]
    config = json.loads((root / "configs/steering_table.json").read_text())
    tasks = {t["id"]: t for t in config["tasks"]}
    assert set(tasks["data-adaptive_steering"]["deps"]) == {
        "data-adaptive_steering-prepare", "data-adaptive_steering-compute", "data-adaptive_steering-oracle"}
    assert tasks["data-adaptive_steering-compute"]["deps"] == ["data-adaptive_steering-prepare"]
    assert all(t["module"].startswith("manifold_attractors.steering.") for t in tasks.values())


def test_calibration_objectives_ties_and_controls():
    parameters = np.array([0., 1., 2., 3.])
    masses = np.array([[[.2, 0, .7, 0, .1], [.1, 0, .8, 0, .1],
                        [.1, 0, .7, 0, .2], [.1, 0, .8, 0, .1]]])
    classes = np.array([[0, 2, 2, 2]])
    for track in fixed.TRACKS:
        assert fixed.choose(parameters, masses, classes, track)[0] == 1
    x, d = np.array([1., 3.]), np.array([1., 0.])
    np.testing.assert_array_equal(fixed.point(x, d, "additive", 2), [3, 3])
    assert np.linalg.norm(fixed.point(x, d, "norm_preserving", 2)) == pytest.approx(np.linalg.norm(x))
    assert fixed.point(x, d, "constant_projection", -2) @ d == pytest.approx(-2)


def test_endpoint_expansion_blocks_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(fixed, "approval", lambda a: "test")
    definition = fixed.grid_definition(tmp_path)
    for method, grid in fixed.grids(definition).items():
        for shard in (0, 1):
            masses = np.zeros((500, len(grid), 5)); masses[..., 2] = 1
            atomic_npz(tmp_path / f"calibration/v000/{shard:03d}/{method}.npz",
                       parameters=grid, class_masses=masses,
                       argmax_classes=np.full((500, len(grid)), 2),
                       sample_ids=np.arange(shard*500, (shard+1)*500))
    with pytest.raises(SystemExit, match="CALIBRATION EXPANDED"):
        fixed.select_stage(SimpleNamespace(root=tmp_path))
    revised = fixed.grid_definition(tmp_path)
    assert revised["version"] == 1
    assert revised["bounds"]["additive"] == [-8, 8]
    assert revised["bounds"]["constant_projection"] == [-16, 8]
    assert not (tmp_path / "calibration/selection.json").exists()


def test_straight_zero_maximum_chooses_left_basin():
    grid = np.arange(-16., 17.)
    # A maximum at zero brackets the LEFT basin, not the right one.
    speed = np.interp(grid, [-16, -8, -4, 0, 3, 8, 16], [2, 20, 1, 20, 2, 20, 1])
    chosen = policies.basin_selection(grid, speed)
    assert chosen["mode"] == "right_minimum"
    np.testing.assert_array_equal(chosen["strengths"], [-4, 3])
    selected = geometry.select_profile("straight", grid, speed)
    assert selected["reference"] == -4 and selected["scalar"] == 3
    assert not geometry.select_profile("straight", grid, np.full(len(grid), np.nan))["flags"] == []


def test_derivative_refinement_and_prespecified_quarter_audit():
    class AnalyticProbe:
        def forward(self, points, full=False):
            assert not full
            s = points[:, 0]
            return np.column_stack([s + .9*np.sin(s), np.zeros(len(s))])
    profile = geometry._profile(AnalyticProbe(), np.array([0., 1.]),
                                np.array([1., 0.]), "straight", 32, True)
    assert profile["valid"] and profile["converged"]
    assert profile["mode"] == "right_minimum"
    assert profile["relative_l2"] < .05 and profile["relative_p99"] < .1
    assert profile["scalar"] == pytest.approx(np.pi, abs=.05)
    assert profile["reference"] < 0
    assert profile["quarter_step"] == profile["step"] / 2
    assert profile["quarter_valid"]
    assert profile["quarter_token_agreement_status"] == "pending_shared_decoder"


def test_fixed_fallback_is_delayed_and_track_specific():
    x, d = np.array([2., 3.]), np.array([1., 0.])
    with pytest.raises(ValueError, match="separately calibrated"):
        geometry.point_constructor(x, d, "straight", "fixed")
    for strength in (1.1, 2.7):
        np.testing.assert_array_equal(
            geometry.point_constructor(x, d, "straight", "fixed", scalar=np.nan, fixed_strength=strength),
            x + strength*d)
    np.testing.assert_array_equal(geometry.point_constructor(x, d, "straight", "no_steer"), x)
    assert np.linalg.norm(geometry.point_constructor(x, d, "angular", "right_minimum", 30)) == pytest.approx(np.linalg.norm(x))


def test_finite_oracle_grids_and_lexicographic_choices():
    assert len(oracle.GRID) == 401 and len(oracle.COARSE) == 408
    assert np.all((oracle.GRID > 0) & (oracle.GRID < 10))
    assert oracle.COARSE.sum() == 208
    strengths = np.array([[1., 2., 3., 0., 10.]])
    masses = np.array([[[.1, 0, .8, 0, .1], [.1, 0, .85, 0, .05],
                        [.2, 0, .8, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]])
    ids = np.array([[0, 0, 2, 2, 2]])
    valid = (strengths > 0) & (strengths < 10)
    best = oracle.choices(strengths, masses, ids, valid, np.arange(5))
    assert best["mass"].tolist() == [1]
    assert best["greedy"].tolist() == [2]
    assert best["chinese_ceiling"].tolist() == [1]


def test_vocabulary_partition_not_one_minus_chinese():
    fine = np.zeros((1, 11)); fine[0, [0, 1, 2, 5, 6]] = [.1, .2, .3, .1, .3]
    coarse = vocabulary.aggregate_subtypes(fine)
    np.testing.assert_allclose(coarse, [[.1, .2, .3, .1, .3]])
    assert coarse[0, list(vocabulary.BAD)].sum() == pytest.approx(.2)
    assert 1 - coarse[0, vocabulary.CHINESE] == pytest.approx(.7)


def test_han_prefix_punctuation_and_special_categories():
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codes = byte_values.copy()
    extra = 0
    for b in range(256):
        if b not in byte_values:
            byte_values.append(b); codes.append(256+extra); extra += 1
    encoder = dict(zip(byte_values, map(chr, codes)))
    raw = [b"hello", "中".encode(), "。".encode(), b"\xe4\xb8", b"\xb8", b"123", "😀".encode(), b"<special>"]
    class Tokenizer:
        all_special_ids = [7]
        def decode(self, ids): return b"".join(raw[i] for i in ids).decode("utf8", errors="replace")
        def convert_ids_to_tokens(self, i): return "".join(encoder[b] for b in raw[i])
    classify = vocabulary.classify_cached(Tokenizer(), vocabulary.HAN_PREFIX_METRIC)
    assert [classify(i) for i in range(8)] == [
        "Latin", "Chinese", "Chinese", "Chinese", "Other nonalphabetic",
        "Other nonalphabetic", "Neutral punctuation/symbols", "Other nonalphabetic"]


def test_calibration_decoding_filesystem_and_invalid_policy(tmp_path, capsys):
    fixed.dev_self_test(tmp_path)
    assert "PASS:" in capsys.readouterr().out


def test_unprovenanced_root_and_missing_dependencies_fail_closed(tmp_path):
    from manifold_attractors.steering.common import freeze_run, require_prepared
    (tmp_path / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="unprovenanced"):
        freeze_run(tmp_path)
    with pytest.raises(FileNotFoundError):
        require_prepared(tmp_path / "fresh")


def test_real_tiny_llama_native_to_fp64_mass_and_oracle(tmp_path, monkeypatch):
    """Real local randomly initialized weights, CPU: not mocked model inference."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.llama import modeling_llama
    from manifold_attractors.core.trajectory_activation_cache import extract_cached
    from manifold_attractors.steering.probe import Probe
    original_attention = modeling_llama.eager_attention_forward
    torch.manual_seed(7)
    config = transformers.LlamaConfig(hidden_size=16, intermediate_size=32,
        num_hidden_layers=20, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=32, max_position_embeddings=128, bos_token_id=1, eos_token_id=2)
    model = transformers.LlamaForCausalLM(config).eval()
    model.config._attn_implementation = "eager"
    snapshot = tmp_path / "tiny"
    model.save_pretrained(snapshot)
    records = [{"sample_id": 0, "input_ids": [1, 3, 4, 5], "position": 2},
               {"sample_id": 1, "input_ids": [1, 6, 7, 8, 9], "position": 3}]
    with torch.inference_mode():
        values, info = extract_cached(model.model, records, [15], "cpu", str(snapshot), tmp_path / "cache", "tiny")
    assert values.dtype == np.float32 and values.shape == (1, 2, 16)
    assert info["atomic_round_trip_exact"]
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: SimpleNamespace(bos_token_id=1))
    try:
        probe = Probe(str(snapshot), device="cpu")
        states = values[0].astype(np.float64)
        categories = np.arange(32) % 5
        matrix = torch.eye(5, dtype=torch.float64)[categories]
        masses, ids = vocabulary.checked_distribution(probe, states, matrix)
        assert probe.audit(states[0]) < 1e-9
        vocabulary.audit_distribution(probe, states[0], matrix, masses[0], ids[0])
        direction = np.ones(16) / 4
        strengths = np.array([[.1, .2, .3]])
        points = states[:1] + strengths[0, :, None]*direction
        mass, tokens = vocabulary.checked_distribution(probe, points, matrix)
        selections = oracle.choices(strengths, mass[None], tokens[None],
                                    np.ones((1, 3), dtype=bool), categories)
        assert all(0 <= index[0] < 3 for index in selections.values())
        assert probe.forward(points).shape == points.shape
        # A genuine learned-state calibration objective, not a hard-coded chosen scalar.
        index, selection = fixed.choose(strengths[0], mass[None], categories[tokens][None], "mass")
        assert selection["parameter"] == strengths[0, index]
    finally:
        modeling_llama.eager_attention_forward = original_attention
