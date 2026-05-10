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
from gs_simulation.relational.cross_sequences import (
    absolute_relational_degree_corrected,
    degree_of_difference,
    trapezoidal_absolute_area,
)
from gs_simulation.relational.deng import (
    deng_relational_coefficient,
    deng_relational_degree,
)
from gs_simulation.relational.negative import (
    negative_similitude_relational_degree,
)
from gs_simulation.relational.relative import relative_relational_degree
from gs_simulation.relational.similitude import similitude_relational_degree
from gs_simulation.relational.superiority import (
    SuperiorityResult,
    build_relational_matrix,
    favorable_indices,
    quasi_favorable_ranking,
    superiority_analysis,
)
from gs_simulation.relational.synthetic import synthetic_relational_degree

__all__ = [
    "SuperiorityResult",
    "absolute_relational_degree",
    "absolute_relational_degree_corrected",
    "build_relational_matrix",
    "closeness_relational_degree",
    "degree_of_difference",
    "deng_relational_coefficient",
    "deng_relational_degree",
    "favorable_indices",
    "negative_similitude_relational_degree",
    "quasi_favorable_ranking",
    "relative_relational_degree",
    "similitude_relational_degree",
    "superiority_analysis",
    "synthetic_relational_degree",
    "trapezoidal_absolute_area",
    "trapezoidal_signed_area",
    "zero_starting_point_image",
]
