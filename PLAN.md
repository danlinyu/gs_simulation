# gs_simulation — Implementation Plan

**Date created:** 2026-05-10
**Author:** Danlin Yu (with Claude Code overnight Session 62 implementing skeleton + Sprint 1)
**Status:** Sprint 1 active
**Source book:** Sifeng Liu (2024), *Grey Systems Analysis: Methods, Models and Applications*, 2nd Ed., Springer (open access).

---

## Goal

Faithful Python implementation of the methods, models, and algorithms covered in Liu (2024). Six chapters → six sprints. Each sprint = one focused implementation session; each method tested against book worked examples or against analytic edge cases.

## Scope discipline

- **Faithful, not novel.** Methods follow the book's mathematical specification exactly. Where the book offers multiple formulations of the same concept (e.g., GM(1,1) four basic forms), all formulations are implemented.
- **Pure NumPy + scipy.** No rpy2, no compiled-language extensions, no opinionated frameworks. Standard scientific-Python.
- **Type-annotated.** All public surface has type hints. Immutable dataclasses for fitted-model containers.
- **Testable.** Each module ships with pytest tests. Where the book provides worked examples with explicit input/output values, those become `book_example`-marked tests.
- **Documented.** Every module docstring cites the relevant chapter and section. Public API has at least one runnable example.

## Sprint roadmap

### Sprint 1 — Foundations: sequence operators + GM(1,1) family (Liu §4 + §7) [ACTIVE]

**Modules:**

| Module | Liu reference | Implements |
|---|---|---|
| `operators/relational.py` | §5.2 | D₁ initialing, D₂ averaging, D₃ interval grey relational operators |
| `operators/accumulation.py` | §4.6 | AGO (1-AGO and r-AGO), IAGO (inverse accumulation) |
| `operators/buffer.py` | §4.2-4.3 | Weakening (mean buffer, geometric mean), strengthening buffer operators |
| `operators/moving.py` | §4.4-4.5 | Average operator, moving-average denoise, quasi-smooth sequence test |
| `gm/base.py` | §7.1 | `GMFit` immutable dataclass; common evaluation/forecasting infrastructure |
| `gm/gm11.py` | §7.2-7.3 | GM(1,1) four basic forms (mean, original-difference, even-difference, discrete); suitable-range validation |
| `gm/remnant.py` | §7.4 | Remnant GM(1,1) for residual modification |
| `gm/group.py` | §7.5 | Group of GM(1,1) — multiple-source weighted aggregation |
| `gm/fractional.py` | §7.6 | Fractional-order GM model |
| `gm/gm_rh.py` | §7.7 | Generalized GM(r, h) for multi-variate inputs |

**Tests:** `tests/operators/`, `tests/gm/`. Each module has at least one book-example test (§7.2.x examples are the canonical worked ones).

**Deliverable:** importable `gs_simulation.operators` and `gs_simulation.gm` packages with passing test suite.

**Acceptance criteria:**
- All public functions have type annotations.
- All public functions have docstrings with `Liu (2024) §X.Y` citation.
- Worked examples from §4 and §7 reproduce within numerical tolerance (atol=1e-6 typical).
- `pytest tests/operators tests/gm` exits 0.

---

### Sprint 2 — Grey relational analysis (Liu Ch 5)

**Modules:**

| Module | Liu reference | Implements |
|---|---|---|
| `relational/deng.py` | §5.3 | Deng's grey relational degree γ₀ᵢ; discrimination coefficient ξ |
| `relational/absolute.py` | §5.4 | Absolute relational degree εᵢⱼ; zero-starting-point transform; trapezoidal area integration |
| `relational/relative.py` | §5.5.1 | Relative relational degree rᵢⱼ; initial-image transform |
| `relational/synthetic.py` | §5.5.2 | Synthetic relational degree ρᵢⱼ = θε + (1−θ)r |
| `relational/similitude.py` | §5.6.1 | Grey similitude relational degree ε̃ᵢⱼ (shape-only) |
| `relational/closeness.py` | §5.6.2 | Grey closeness relational degree ρ̃ᵢⱼ (level + shape) |
| `relational/negative.py` | §5.7 | Negative grey relational analysis (anti-correlation, Liu et al. 2022) |
| `relational/cross_sequences.py` | §5.8 | Cross-sequence correction Δᵢⱼ; modified εᴱᶜᵢⱼ |
| `relational/superiority.py` | §5.9 | Superiority analysis on s×m relational matrix; favorable/quasi-favorable detection |

**Tests:** `tests/relational/`. Worked examples: §5.3 Jiangsu GDP, §5.4 X₀/X₁ alignment, §5.5 same data with relative degree, §5.7 reverse sequences, §5.8 oscillating cross-sequences, §5.9 outputs vs. factors.

**Deliverable:** `gs_simulation.relational` package with passing test suite covering all seven degree variants.

**CM-relevance:** This sprint produces the load-bearing CM borrowings — see `Covariation_Mining/reports/grey_relational_analysis_for_cm_2026-05-10.md` for the angles. Sprint 2 makes CM-side adapters trivial.

---

### Sprint 3 — Grey clustering evaluation (Liu Ch 6)

**Modules:**

| Module | Liu reference | Implements |
|---|---|---|
| `clustering/relational.py` | §6.2 | Grey relational clustering model |
| `clustering/possibility.py` | §6.3 | Common possibility functions (triangular, end-point center) |
| `clustering/variable_weight.py` | §6.4 | Variable-weight grey clustering |
| `clustering/fixed_weight.py` | §6.5 | Fixed-weight grey clustering |
| `clustering/mixed.py` | §6.6 | Grey clustering based on mixed possibility functions |

**Tests:** `tests/clustering/`.

**Deliverable:** `gs_simulation.clustering` package.

---

### Sprint 4 — Combined models + verification (Liu Ch 8 + Ch 9)

**Modules:**

| Module | Liu reference | Implements |
|---|---|---|
| `combined/econometrics.py` | §8.2 | Grey econometrics models |
| `combined/linear_regression.py` | §8.3 | Combined grey-linear regression |
| `combined/cobb_douglas.py` | §8.4 | Grey Cobb–Douglas production function |
| `combined/neural_network.py` | §8.5 | Grey artificial neural network (lightweight; numpy-only) |
| `combined/markov.py` | §8.6 | Grey-Markov hybrid model |
| `combined/rough.py` | §8.7 | Combined grey-rough model |
| `verification/criteria.py` | §9.2 | Model verification criteria (residual, posterior error, MAPE, etc.) |
| `verification/interval.py` | §9.3 | Interval forecasting |
| `verification/distortion.py` | §9.4 | Grey distortion forecasting |
| `verification/waveform.py` | §9.5 | Wave-form forecasting |
| `verification/system.py` | §9.6 | System forecasting (multivariate) |

**Tests:** `tests/combined/`, `tests/verification/`.

---

### Sprint 5 — Decision + control (Liu Ch 10 + Ch 11)

**Modules:**

| Module | Liu reference | Implements |
|---|---|---|
| `decision/event.py` | §10.2 | Event and decision scheme primitives |
| `decision/grey_target.py` | §10.3 | Grey target decisions |
| `decision/multi_attribute.py` | §10.5 | Multi-attribute weighted intelligent grey target decision |
| `decision/paradox.py` | §10.6 | Maximum-value paradox + resolution |
| `control/controllability.py` | §11.2 | Controllability and observability of grey systems |
| `control/transfer.py` | §11.3 | Transfer functions of grey systems |
| `control/stability.py` | §11.4 | Robust stability of grey systems |
| `control/models.py` | §11.5 | Several typical grey control models |

**Tests:** `tests/decision/`, `tests/control/`.

---

### Sprint 6 — Spectrum + foundations (Liu Ch 12 + Ch 1-3)

**Modules:**

| Module | Liu reference | Implements |
|---|---|---|
| `spectrum/time_series.py` | §12.2 | Spectrum analysis of time-series data |
| `spectrum/operator_filter.py` | §12.3 | Filtering effect of mean and accumulation operators |
| `spectrum/buffer_filter.py` | §12.4 | Spectrum analysis of buffer operators |
| `numbers/grey_number.py` | §3.1 | Grey number primitives |
| `numbers/whitenization.py` | §3.2 | Whitenization weight function, degree of greyness |
| `numbers/axioms.py` | §3.3 | Axiomatic degree of greyness |
| `numbers/intervals.py` | §3.4 | Interval grey number operations |
| `numbers/general.py` | §3.5 | General grey numbers and their algebraic system |
| `foundations/concepts.py` | §1.4-1.5 | Elementary concepts and fundamental principles |
| `foundations/poor_data.py` | §2.x | Poor-data analysis primitives |

**Tests:** `tests/spectrum/`, `tests/numbers/`, `tests/foundations/`.

**Deliverable:** Full repo coverage of Liu (2024).

---

## Out-of-scope

- **Visual C# software.** Liu (2024) ships an attached C#-based GUI (Bo Zeng). Re-implementing the GUI is out of scope; users compose via the Python API or build their own UI on top.
- **Domain-specific application case studies** beyond the worked examples in the book. Domain studies belong in companion repositories.
- **Speculative extensions.** Extensions to the canonical methods (e.g., novel grey-deep-learning hybrids) are not part of this library; they belong in user-side research code that depends on `gs_simulation` as a base layer.

## Per-sprint discipline

1. Read the relevant chapter sections of `greysystem.md` (or the original PDF when figures are essential).
2. List worked examples with input/output values for test fixtures.
3. Implement each method as a small typed function; immutable dataclass for fitted-model containers.
4. Write tests: at least one `book_example`-marked test per public function reproducing the corresponding worked example.
5. Lint with ruff; type-check with mypy strict.
6. Atomic commit per logical method. Squash on sprint close.

## Open architectural questions (for future Discs)

1. **API uniformity.** Should every fit function return an `XxxFit` dataclass, or should some return raw arrays (Deng's γ as a scalar)? Likely: structured output for parametric models (GM family); scalars/arrays for relational degrees.
2. **Public re-export at package root.** Should `import gs_simulation` expose all 50+ functions, or require sub-package imports? Likely: sub-package imports for hygiene; explicit re-export only of headline API (e.g., `gs_simulation.gm11_fit`).
3. **Versioning.** Pre-release alpha (0.1.x) until Sprint 6 lands; minor-bump on each subsequent sprint; 1.0 when full Liu (2024) coverage tested + documented.

---

## Cross-reference to Covariation_Mining

A separate `Covariation_Mining` project (Yu, in progress) explores covariation mining methodology. The CM-relevance synthesis at `Covariation_Mining/reports/grey_relational_analysis_for_cm_2026-05-10.md` enumerates which `gs_simulation` capabilities `Covariation_Mining` will adopt as adapters once they ship. The two projects are independent; `gs_simulation` does not depend on `Covariation_Mining`.
