"""Grey forecasting models — the GM family (Liu §7).

Modules:
    base            — `GMFit` immutable dataclass; common forecasting infrastructure (§7.1)
    gm11            — GM(1,1) four basic forms (§7.2-7.3)
    remnant         — Remnant GM(1,1) for residual modification (§7.4)
    group           — Group of GM(1,1) — multiple-source weighted aggregation (§7.5)
    fractional      — Fractional-order GM model (§7.6)
    gm_rh           — Generalized GM(r, h) for multi-variate inputs (§7.7)

Public API populated as Sprint 1 modules ship.
"""

__all__: list[str] = []
