# Pure‑NOMAD

**Pure‑NOMAD** is a lightweight evolutionary optimiser that marries global search (crossover + mutation) with local, derivative‑free refinement powered by the [NOMAD](https://www.gerad.ca/nomad) optimizer (via **PyNomad**). It is domain‑agnostic: supply any Python function that maps a NumPy 1‑D vector to a scalar fitness, and Pure‑NOMAD will do the rest.

> **Why use it?**
>
> * Global exploration keeps you from getting stuck.
> * NOMAD’s poll/search steps give fast local convergence—no gradients required.
> * Works out of the box on laptops and clusters; optional **Ray** backend parallelises local searches.

---

## Installation

```bash
# Stable release (once published)
pip install pure-nomad

# Development version (clone + editable)
git clone https://github.com/yourname/pure-nomad.git
cd pure-nomad
pip install -e .[ray]   # extras: ray parallelisation
```

> **Requirements** python >= 3.9 · numpy >= 1.23 · tqdm · PyNomad >= 0.9 (Ray optional)

---

## Quick start

```python
import numpy as np
from pure_nomad import PureNOMAD

# Toy objective (negative sphere ⇒ maximise ⇒ minimum at 0)
obj = lambda x: -np.sum(x**2)

opt = PureNOMAD(
    population_size=32,
    dimension=10,
    objective_fn=obj,
    subset_size=5,       # coordinates NOMAD refines per elite
    bounds=0.2,          # ± box around the slice
)

best_x, best_fit = opt.run(generations=100)
print(f"Best fitness: {best_fit:.4f}\nBest vector : {best_x}")
```

---

## API highlights

```python
PureNOMAD(
    population_size: int,
    dimension: int | None = None,
    objective_fn: Callable[[ndarray], float],
    subset_size: int = 20,
    bounds: float = 0.1,
    max_bb_eval: int = 200,
    n_elites: int | None = None,
    n_mutate_coords: int = 5,
    crossover_rate: float = 0.5,
    init_pop: ndarray | None = None,
    init_vec: ndarray | None = None,
    low: float = -1.0,
    high: float = 1.0,
    use_ray: bool | None = None,
    seed: int | None = None,
)
```

* **population\_size** (μ) — number of individuals.
* **subset\_size** — how many coordinates of each elite are refined by NOMAD.
* **bounds** — ± search box around that coordinate slice.
* **max\_bb\_eval** — NOMAD black‑box evaluations per call.
* **use\_ray** — `True`/`False`/`None`; `None` auto‑enables if Ray is importable.

---

## Running the test suite

```bash
pip install -e .[dev]     # includes pytest
pytest -q                 # smoke tests should pass
```

---

## Contributing

1. Fork & create feature branch.
2. `pre-commit install` (black, ruff, mypy hooks).
3. Add unit tests for any new behaviour.
4. Open a pull request.

---

## License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

### Acknowledgements

* [PyNomad](https://github.com/bertsky/pynomad) – Python bindings for NOMAD.
* NOMAD team at GERAD & Polytechnique Montréal.
