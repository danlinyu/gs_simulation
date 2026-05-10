# CLAUDE.md

Guidance for Claude Code when working in `gs_simulation`.

## What this repo is

Reference Python implementation of grey systems analysis methods. Faithful to:

> Sifeng Liu (2024), *Grey Systems Analysis: Methods, Models and Applications*, 2nd Ed., Springer Singapore.
> ISBN 978-981-97-8726-5. Open access under CC BY-NC-ND 4.0.

Source markdown is at `D:\cm_recovery\Covariation_Mining\greysystem.md` (sibling repo). PDF at `D:\cm_recovery\Covariation_Mining\GreySystem.pdf`.

## Architectural commitments

- **Pure NumPy + scipy.** No rpy2, no R-INLA, no compiled extensions. If a method needs more, document why and propose an alternative.
- **Type-annotated.** Public surface has type hints; immutable dataclasses for fitted-model containers.
- **Tested against book examples.** Each public function has at least one `book_example`-marked test reproducing a worked example from Liu (2024). Tolerance: typically `atol=1e-6` for numerical agreement.
- **Module docstrings cite the book.** Every module's docstring identifies the relevant chapter and section (e.g., `"""Liu §5.3 Deng's grey relational degree."""`).
- **MIT-licensed code, CC BY-NC-ND attribution.** Methods are not copyrightable; the book's prose, figures, and worked-example datasets are. Test fixtures using book-example inputs are clearly cited.

## Repository layout

```
gs_simulation/
├── pyproject.toml
├── README.md
├── LICENSE         # MIT
├── NOTICE          # Attribution to Liu (2024) and antecedents
├── PLAN.md         # 6-sprint roadmap
├── CLAUDE.md       # this file
├── .gitignore
├── src/gs_simulation/
│   ├── __init__.py
│   ├── operators/      # Sprint 1 (Liu §4 + §5.2)
│   ├── gm/             # Sprint 1 (Liu §7)
│   ├── relational/     # Sprint 2 (Liu §5)
│   ├── clustering/     # Sprint 3 (Liu §6)
│   ├── combined/       # Sprint 4 (Liu §8)
│   ├── verification/   # Sprint 4 (Liu §9)
│   ├── decision/       # Sprint 5 (Liu §10)
│   ├── control/        # Sprint 5 (Liu §11)
│   ├── spectrum/       # Sprint 6 (Liu §12)
│   ├── numbers/        # Sprint 6 (Liu §3)
│   └── foundations/    # Sprint 6 (Liu §1, §2)
├── tests/              # Mirror of src layout
├── examples/           # Runnable scripts reproducing book applications
└── docs/               # Architecture notes, method index
```

## Sprint discipline (per PLAN.md)

Six sprints, each self-contained. Active sprint shown in PLAN.md and README.md. Don't start a future sprint before its predecessor passes its acceptance criteria.

Per-sprint workflow:
1. Read the chapter sections in `Covariation_Mining/greysystem.md`.
2. Catalog worked examples with explicit input/output values for test fixtures.
3. Implement each method as a small, typed function. Use immutable dataclasses for fitted-model containers.
4. Write tests: each public function has at least one `book_example`-marked test reproducing the corresponding worked example.
5. Lint with `ruff check src tests`; type-check with `mypy --strict src/gs_simulation`.
6. Atomic commit per logical method (e.g., "feat(gm): GM(1,1) mean form per Liu §7.2.1").

## Commands

```bash
# Install (editable + dev deps)
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run only book-example tests
pytest -m book_example

# Skip slow tests
pytest -m "not slow"

# Lint
ruff check src tests

# Type-check
mypy --strict src/gs_simulation
```

## Cross-reference to Covariation_Mining

A separate `Covariation_Mining` project (Yu, in progress) at `D:\cm_recovery\Covariation_Mining\` explores covariation mining methodology. After Sprint 2 (full GRA), CM will adopt thin adapters for:

- **GM(1,1) baseline** alongside AR(1) for predictive harness (Disc 58 §1 falsifiability).
- **AR(1)-residual GRA** as a novel angle on the AR(1)-absorption problem (Disc 50 §2 Bullet 5).
- **Superiority analysis + lead-lag classification** as supplementary post-hoc verification.

See `Covariation_Mining/reports/grey_relational_analysis_for_cm_2026-05-10.md` for the full CM-relevance analysis.

The two projects are independent; `gs_simulation` does not depend on `Covariation_Mining`.
