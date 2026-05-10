"""Grey forecasting models — the GM family (Liu §7).

Modules:
    base            — `GMFit` immutable dataclass; common forecasting infrastructure (§7.1)
    gm11            — GM(1,1) four basic forms (§7.2-7.3)
    remnant         — Remnant GM(1,1) for residual modification (§7.4) [pending]
    group           — Group of GM(1,1) — multiple-source weighted aggregation (§7.5) [pending]
    fractional      — Fractional-order GM model (§7.6) [pending]
    gm_rh           — Generalized GM(r, h) for multi-variate inputs (§7.7) [pending]
"""

from gs_simulation.gm.base import GMFit
from gs_simulation.gm.fractional import (
    FractionalGMFit,
    apply_fractional_ago,
    fit_fractional_gm11,
    forecast_fractional_gm11,
    simulate_fractional_gm11,
)
from gs_simulation.gm.gm11 import (
    fit_dgm11,
    fit_edgm11,
    fit_egm11,
    fit_odgm11,
    forecast_gm11,
    simulate_gm11,
)
from gs_simulation.gm.group import (
    fit_all_data_gm11,
    fit_metabolic_gm11,
    fit_new_information_gm11,
    fit_partial_data_gm11,
)
from gs_simulation.gm.remnant import (
    RemnantGMFit,
    fit_remnant_gm11,
    forecast_remnant_gm11,
    simulate_remnant_gm11,
)

__all__ = [
    "FractionalGMFit",
    "GMFit",
    "RemnantGMFit",
    "apply_fractional_ago",
    "fit_all_data_gm11",
    "fit_dgm11",
    "fit_edgm11",
    "fit_egm11",
    "fit_fractional_gm11",
    "fit_metabolic_gm11",
    "fit_new_information_gm11",
    "fit_odgm11",
    "fit_partial_data_gm11",
    "fit_remnant_gm11",
    "forecast_fractional_gm11",
    "forecast_gm11",
    "forecast_remnant_gm11",
    "simulate_fractional_gm11",
    "simulate_gm11",
    "simulate_remnant_gm11",
]
