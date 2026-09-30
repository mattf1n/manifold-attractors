import numpy as np
from manifold_attractors.experiments.great_circle_norm_grid import candidate_positions
from manifold_attractors.pipeline import load_tasks


def test_historical_norm_keeps_non_special_position_zero():
    ids = [10, 11, 12, 0]
    assert candidate_positions(ids, [0]) == [1, 2]
    assert candidate_positions(ids, [0], include_position_zero=True) == [0, 1, 2]
    # A leading special BOS makes the policies coincide (the Llama case).
    ids = [0, 10, 11, 12]
    assert candidate_positions(ids, [0], include_position_zero=True) == candidate_positions(ids, [0])


def test_removing_zero_can_change_draws_that_never_select_zero():
    ids = list(range(256))
    old = np.random.default_rng(20260908).choice(candidate_positions(ids, [], include_position_zero=True), 20)
    new = np.random.default_rng(20260908).choice(candidate_positions(ids, []), 20)
    assert np.all(old > 0)
    assert np.any(old != new)


def test_norm_recipes_explicitly_choose_historical_policy():
    tasks = load_tasks()
    for model in ['llama3', 'olmo3', 'qwen3']:
        args = tasks[f'data-great_circle_norm_grid-{model}']['args']
        assert args[args.index('--sampling-policy')+1] == 'historical-all-nonspecial'
        assert '--sampling-policy' not in tasks[f'data-great_circle_speed_grid-{model}']['args']
