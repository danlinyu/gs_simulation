"""Grey relational analysis — Liu Chapter 5.

Modules:
    deng            — Deng's grey relational degree γ₀ᵢ (§5.3) [Sprint 2 partial]
    absolute        — Absolute relational degree εᵢⱼ (§5.4) [pending]
    relative        — Relative relational degree rᵢⱼ (§5.5.1) [pending]
    synthetic       — Synthetic relational degree ρᵢⱼ (§5.5.2) [pending]
    similitude      — Grey similitude relational degree (§5.6.1) [pending]
    closeness       — Grey closeness relational degree (§5.6.2) [pending]
    negative        — Negative grey relational analysis (§5.7) [pending]
    cross_sequences — Cross-sequence correction (§5.8) [pending]
    superiority     — Superiority analysis on relational matrix (§5.9) [pending]
"""

from gs_simulation.relational.absolute import (
    absolute_relational_degree,
    trapezoidal_signed_area,
    zero_starting_point_image,
)
from gs_simulation.relational.closeness import closeness_relational_degree
from gs_simulation.relational.deng import (
    deng_relational_coefficient,
    deng_relational_degree,
)
from gs_simulation.relational.relative import relative_relational_degree
from gs_simulation.relational.similitude import similitude_relational_degree
from gs_simulation.relational.synthetic import synthetic_relational_degree

__all__ = [
    "absolute_relational_degree",
    "closeness_relational_degree",
    "deng_relational_coefficient",
    "deng_relational_degree",
    "relative_relational_degree",
    "similitude_relational_degree",
    "synthetic_relational_degree",
    "trapezoidal_signed_area",
    "zero_starting_point_image",
]
