"""Pure screening metrics shared by MLFF and DFT inference evaluation.

Candidate order is part of the benchmark definition.  For every budget ``k``
the selector may inspect only the first ``k`` generated candidates, filters
invalid/anomalous structures, and then chooses the lowest MLFF energy.  DFT,
when available, evaluates that already selected candidate; it never re-ranks
the candidate set with DFT energies.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math

import numpy as np


def normalize_budgets(budgets: Iterable[int], candidate_count: int) -> list[int]:
    """Return sorted, unique candidate budgets bounded by the generated set."""
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    result = sorted({int(value) for value in budgets})
    if not result or result[0] < 1 or result[-1] > candidate_count:
        raise ValueError(
            f"budgets must lie in [1, {candidate_count}], received {result}"
        )
    return result


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def build_screening_curve(
    mlff_energies_eV: np.ndarray,
    eligible: np.ndarray,
    reference_mlff_min_eV: float,
    budgets: Iterable[int],
    *,
    success_threshold_eV: float = 0.1,
    valid: np.ndarray | None = None,
    anomalous: np.ndarray | None = None,
    converged: np.ndarray | None = None,
    cluster_ids: np.ndarray | None = None,
    dft_energies_eV: Mapping[int, float] | None = None,
    reference_dft_min_eV: float | None = None,
) -> list[dict]:
    """Construct prefix-budget SR@k and regret metrics.

    ``MLFF-SR@k`` is a proxy metric.  ``DFT-SR@k`` is populated only when the
    MLFF-selected candidate has a supplied DFT single-point energy.
    """
    energies = np.asarray(mlff_energies_eV, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if energies.ndim != 1 or eligible.shape != energies.shape:
        raise ValueError("energies and eligible must be same-length rank-1 arrays")
    if not math.isfinite(float(reference_mlff_min_eV)):
        raise ValueError("reference MLFF minimum must be finite")
    if not math.isfinite(success_threshold_eV) or success_threshold_eV < 0:
        raise ValueError("success threshold must be finite and nonnegative")
    arrays = {}
    for name, values, dtype in (
        ("valid", valid, bool),
        ("anomalous", anomalous, bool),
        ("converged", converged, bool),
        ("cluster_ids", cluster_ids, int),
    ):
        if values is None:
            continue
        array = np.asarray(values, dtype=dtype)
        if array.shape != energies.shape:
            raise ValueError(f"{name} must have shape {energies.shape}")
        arrays[name] = array
    budgets = normalize_budgets(budgets, len(energies))
    dft_energies_eV = dft_energies_eV or {}
    dft_reference = (
        None if reference_dft_min_eV is None else _finite_or_none(reference_dft_min_eV)
    )

    curve = []
    for budget in budgets:
        prefix_eligible = np.flatnonzero(
            eligible[:budget] & np.isfinite(energies[:budget])
        )
        selected = (
            int(prefix_eligible[np.argmin(energies[prefix_eligible])])
            if len(prefix_eligible)
            else None
        )
        mlff_energy = None if selected is None else float(energies[selected])
        mlff_regret = (
            None if mlff_energy is None else mlff_energy - float(reference_mlff_min_eV)
        )
        row = {
            "k": budget,
            "eligible_count": int(len(prefix_eligible)),
            "eligible_fraction": float(len(prefix_eligible) / budget),
            "selected_candidate_index": selected,
            "selected_mlff_energy_eV": mlff_energy,
            "mlff_energy_regret_eV": mlff_regret,
            "mlff_success": bool(
                mlff_regret is not None
                and mlff_regret <= success_threshold_eV + 1e-12
            ),
        }
        for name in ("valid", "anomalous", "converged"):
            if name in arrays:
                row[f"{name}_count"] = int(np.count_nonzero(arrays[name][:budget]))
                row[f"{name}_fraction"] = float(np.mean(arrays[name][:budget]))
        if "cluster_ids" in arrays:
            cluster_values = arrays["cluster_ids"][:budget]
            row["unique_relaxed_basin_count"] = int(
                np.unique(cluster_values[cluster_values >= 0]).size
            )

        candidate_dft = None if selected is None else dft_energies_eV.get(selected)
        candidate_dft = None if candidate_dft is None else _finite_or_none(candidate_dft)
        dft_regret = (
            None
            if candidate_dft is None or dft_reference is None
            else candidate_dft - dft_reference
        )
        row.update(
            {
                "selected_dft_total_energy_eV": candidate_dft,
                "dft_energy_regret_eV": dft_regret,
                "dft_evaluated": dft_regret is not None,
                "dft_success": (
                    None
                    if dft_regret is None
                    else bool(dft_regret <= success_threshold_eV + 1e-12)
                ),
            }
        )
        curve.append(row)
    return curve


def selected_candidate_budgets(curve: Iterable[dict]) -> dict[int, list[int]]:
    """Map every unique selected candidate to the budgets selecting it."""
    selected: dict[int, list[int]] = {}
    for row in curve:
        index = row.get("selected_candidate_index")
        if index is not None:
            selected.setdefault(int(index), []).append(int(row["k"]))
    return selected
