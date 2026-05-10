"""gs_simulation — Reference Python implementation of grey systems methods.

Faithful to Sifeng Liu (2024), *Grey Systems Analysis: Methods, Models and
Applications*, 2nd Ed., Springer (open access under CC BY-NC-ND 4.0).

Sub-packages:
    operators       — Sequence operators (Liu §4 + §5.2)
    numbers         — Grey numbers (Liu §3)
    relational      — Grey relational analysis (Liu §5)
    gm              — GM forecasting models (Liu §7)
    combined        — Combined grey models (Liu §8)
    verification    — Forecasting verification (Liu §9)
    clustering      — Grey clustering evaluation (Liu §6)
    decision        — Grey decision-making (Liu §10)
    control         — Grey control (Liu §11)
    spectrum        — Spectrum analysis (Liu §12)
    foundations     — Foundational concepts (Liu §1, §2)

See PLAN.md at the repo root for the active sprint.
"""

__version__ = "0.1.0"
__author__ = "Danlin Yu"
__license__ = "MIT"

__all__: list[str] = []  # Public API exported lazily as sprints complete.
