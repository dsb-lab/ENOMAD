import numpy as np
import pytest

from ENOMAD import ENOMAD


@pytest.mark.parametrize("init_arg", ["init_pop", "init_vec"])
@pytest.mark.parametrize("fill_value", [0.0, 1.0])
def test_numpy_initialization(init_arg, fill_value):
    shape = (4, 3) if init_arg == "init_pop" else (3,)
    initial = np.full(shape, fill_value)

    opt = ENOMAD(
        "EA",
        population_size=4,
        objective_fn=np.sum,
        subset_size=2,
        seed=0,
        use_ray=False,
        **{init_arg: initial},
    )

    assert opt.D == 3
    np.testing.assert_array_equal(opt.pop, np.full((4, 3), fill_value))
    assert not np.shares_memory(opt.pop, initial)


@pytest.mark.parametrize("dimension", [1, 3])
def test_numpy_initializers_are_mutually_exclusive(dimension):
    with pytest.raises(
        ValueError, match="Only one of init_pop or init_vec should be given"
    ):
        ENOMAD(
            "EA",
            population_size=1,
            objective_fn=np.sum,
            subset_size=1,
            init_pop=np.zeros((1, dimension)),
            init_vec=np.zeros(dimension),
            use_ray=False,
        )


@pytest.mark.parametrize("seed", [0, 7, None])
def test_initialization_uses_explicit_or_generated_seed(seed, monkeypatch):
    generated_seeds = []

    def random_seed(low, high):
        generated_seeds.append((low, high))
        return 123

    monkeypatch.setattr(np.random, "randint", random_seed)
    expected_rng = np.random.default_rng(123 if seed is None else seed)
    expected_pop = expected_rng.uniform(-1, 1, size=(4, 3))

    opt = ENOMAD(
        "EA",
        population_size=4,
        dimension=3,
        objective_fn=np.sum,
        subset_size=2,
        seed=seed,
        use_ray=False,
    )

    assert generated_seeds == ([(1, 100001)] if seed is None else [])
    np.testing.assert_array_equal(opt.pop, expected_pop)
    np.testing.assert_array_equal(opt.rng.random(5), expected_rng.random(5))
