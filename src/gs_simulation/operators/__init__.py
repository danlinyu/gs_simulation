"""Sequence operators for grey systems (Liu §4 and §5.2).

Modules:
    relational      — D₁ (initialing), D₂ (averaging), D₃ (interval) operators (§5.2)
    accumulation    — AGO and IAGO operators (§4.6)
    buffer          — Weakening and strengthening buffer operators (§4.2-4.3)
    moving          — Average operator and moving-average denoise (§4.4-4.5)

Public API populated as Sprint 1 modules ship.
"""

from gs_simulation.operators.accumulation import apply_ago, apply_iago
from gs_simulation.operators.buffer import (
    apply_asbo,
    apply_awbo,
    apply_gfbo,
    apply_wasbo,
    apply_wawbo,
    apply_wgawbo,
)
from gs_simulation.operators.moving import (
    is_quasi_smooth,
    mean_operator,
    moving_average_denoise,
    smoothness_ratio,
    stepwise_ratio,
)
from gs_simulation.operators.relational import (
    apply_d1_initialing,
    apply_d2_averaging,
    apply_d3_interval,
)

__all__ = [
    "apply_ago",
    "apply_asbo",
    "apply_awbo",
    "apply_d1_initialing",
    "apply_d2_averaging",
    "apply_d3_interval",
    "apply_gfbo",
    "apply_iago",
    "apply_wasbo",
    "apply_wawbo",
    "apply_wgawbo",
    "is_quasi_smooth",
    "mean_operator",
    "moving_average_denoise",
    "smoothness_ratio",
    "stepwise_ratio",
]
