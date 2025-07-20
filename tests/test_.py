import numpy as np
from pure_nomad import PureNOMAD
"""Minimal smoke tests for PureNOMAD.
Run with `pytest -q`.
These are deliberately lightweight – they just make sure the
optimizer can be imported, instantiated, and run deterministically.
"""

# ------------------------------------------------------------------
# Utility objective (easy, convex)
# ------------------------------------------------------------------

def sphere(x: np.ndarray) -> float:
    """Negative sphere – global maximum at 0 with fitness 0."""
    return -np.sum(x ** 2)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_smoke_run():
    """The optimizer should run a few generations without error."""
    opt = PureNOMAD(
        population_size=8,
        dimension=3,
        objective_fn=sphere,
        subset_size=2,
        bounds=0.1,
        max_bb_eval=20,
        n_elites=2,
        n_mutate_coords=1,
        seed=123,
        use_ray=False,
    )

    best_x, best_fit = opt.run(generations=5)

    # Basic sanity checks
    assert isinstance(best_fit, float)
    assert best_x.shape == (3,)

