# gs_simulation

> **Reference implementation of grey systems methods, models, and simulation in Python.**
> Faithful to Sifeng Liu (2024), *Grey Systems Analysis: Methods, Models and Applications*, 2nd Ed., Springer.
> Pure NumPy + scipy. MIT-licensed code; CC BY-NC-ND attribution to the source book.

[![Status: Pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange)](#status)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## What this is

A self-contained Python library implementing the canonical grey systems analysis methods covered in Sifeng Liu's 2nd-edition Springer monograph (2024, open access under CC BY-NC-ND 4.0). The library covers:

- **Sequence operators** — initialing / averaging / interval normalization (D₁, D₂, D₃); accumulation (AGO) / inverse accumulation (IAGO); buffer operators (weakening, strengthening); average and moving-average operators.
- **Grey numbers** — interval grey numbers, general grey numbers, kernel and degree of greyness, algebraic operations.
- **Grey relational analysis (GRA)** — Deng's classic γ; absolute / relative / synthetic; similitude / closeness; negative GRA (anti-correlation); cross-sequence corrections; superiority analysis.
- **Grey forecasting models** — GM(1,1) four basic forms; remnant GM(1,1); group GM(1,1); fractional GM; GM(r,h).
- **Grey clustering evaluation** — relational clustering, variable-weight, fixed-weight, mixed possibility functions.
- **Combined grey models** — grey econometrics, grey-Cobb-Douglas, grey-Markov, grey-rough hybrid.
- **Forecasting verification** — interval, distortion, wave-form, system forecasting.
- **Grey decision-making** — grey target decisions, multi-attribute weighted intelligent grey target decisions.
- **Grey control** — controllability, observability, transfer functions, robust stability.
- **Spectrum analysis** — operator filtering effects, buffer spectrum analysis.

## Status

**Pre-alpha** as of 2026-05-10. Active staged implementation. See [`PLAN.md`](PLAN.md) for the sprint roadmap.

| Sprint | Status | Coverage |
|---|---|---|
| 1 — Sequence operators + GM(1,1) family | in progress | Liu §4 + §7.2-7.7 |
| 2 — Grey relational analysis (Ch 5) | not started | Liu §5 (all sections) |
| 3 — Grey clustering (Ch 6) | not started | Liu §6 |
| 4 — Combined models + verification (Ch 8 + 9) | not started | Liu §8 + §9 |
| 5 — Decision + control (Ch 10 + 11) | not started | Liu §10 + §11 |
| 6 — Spectrum + foundations (Ch 12 + 1-3) | not started | Liu §12 + §1-§3 |

## Why a fresh implementation

Existing Python packages for grey systems are either fragmentary (cover only GM(1,1)) or non-canonical (deviate from textbook formulations). Production research workflows that need to verify methodological alignment with the canonical 2nd-edition exposition have no single library to depend on. `gs_simulation` aims to fill that gap with:

- **Faithful exposition** — module docstrings cite the book chapter and section; numerical tests reproduce worked examples from the book.
- **Pure NumPy + scipy** — no rpy2, no R-INLA, no compiled-language extensions. Portable across platforms.
- **Type-annotated, immutable dataclasses** — modern Python idioms; safe defaults.
- **Comprehensive test coverage** — every method tested against book worked examples or against analytic edge cases.

## Install

```bash
git clone https://github.com/danlinyu/gs_simulation.git
cd gs_simulation
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest tests/
```

## Quick start

```python
# Sprint 1+ examples will land here as modules ship.
# For now, see PLAN.md for the implementation timeline.
```

## Project structure

```
gs_simulation/
├── src/gs_simulation/
│   ├── operators/      # Sequence operators (Liu §4 + §5.2)
│   ├── numbers/        # Grey numbers (Liu §3)
│   ├── relational/     # Grey relational analysis (Liu §5)
│   ├── gm/             # GM forecasting models (Liu §7)
│   ├── combined/       # Combined models (Liu §8)
│   ├── verification/   # Forecasting verification (Liu §9)
│   ├── clustering/     # Grey clustering evaluation (Liu §6)
│   ├── decision/       # Grey decision-making (Liu §10)
│   ├── control/        # Grey control (Liu §11)
│   ├── spectrum/       # Spectrum analysis (Liu §12)
│   └── foundations/    # Foundational concepts (Liu §1, §2)
├── tests/              # Mirror of src layout; book examples + edge cases
├── examples/           # Runnable scripts reproducing book applications
└── docs/               # Architecture notes, method index, usage guides
```

## Citation

If you use `gs_simulation` in academic work, please cite both this software and the canonical reference:

```
@software{gs_simulation_2026,
  author = {Yu, Danlin},
  title = {gs_simulation: Reference Python implementation of grey systems methods},
  year = {2026},
  url = {https://github.com/danlinyu/gs_simulation},
  version = {0.1.0}
}

@book{liu_grey_systems_2024,
  author = {Liu, Sifeng},
  title = {Grey Systems Analysis: Methods, Models and Applications},
  year = {2024},
  edition = {2nd},
  publisher = {Springer Nature Singapore},
  isbn = {978-981-97-8726-5},
  doi = {10.1007/978-981-97-8727-2},
  note = {Open Access under CC BY-NC-ND 4.0}
}
```

## License

Code: [MIT](LICENSE).
Attribution to source book: [NOTICE](NOTICE).

## Author

Danlin Yu — danlinyu@gmail.com
Department of Earth and Environmental Studies, Montclair State University.
