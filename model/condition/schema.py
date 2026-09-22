"""Validated schema for independently controllable material construction factors.

The full slab graph is deliberately not used as a categorical system ID.  It is
still consumed by the local equivariant trunk as the realized atomic geometry,
while this schema records the variables from which that realization was built.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import gcd
from functools import reduce
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


CONDITION_BATCH_KEYS = (
    "primitive_atomic_numbers",
    "primitive_fractional_positions",
    "primitive_atom_mask",
    "primitive_cell",
    "miller_index",
    "miller_conventional_cell",
    "termination_shift",
    "top",
    "supercell_matrix",
    "slab_layers",
    "vacuum_A",
    "strain_voigt",
    "adsorbates_per_cell",
)


@dataclass(frozen=True)
class FactorizedCondition:
    """One crystal construction, before it is realized as a complete slab."""

    system_id: str
    primitive_atomic_numbers: np.ndarray
    primitive_fractional_positions: np.ndarray
    primitive_cell: np.ndarray
    miller_index: np.ndarray
    miller_conventional_cell: np.ndarray
    termination_shift: float
    top: bool
    supercell_matrix: np.ndarray
    slab_layers: int
    vacuum_A: float
    strain_voigt: np.ndarray
    adsorbates_per_cell: int
    adsorbate_atomic_numbers: np.ndarray

    @property
    def primitive_atom_count(self) -> int:
        return int(len(self.primitive_atomic_numbers))


def _array(value: object, dtype: np.dtype, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return result


def _validate_miller(miller: np.ndarray) -> None:
    absolute = [abs(int(value)) for value in miller if int(value) != 0]
    if not absolute:
        raise ValueError("miller_index cannot be (0,0,0)")
    if reduce(gcd, absolute) != 1:
        raise ValueError(
            "miller_index must be reduced to coprime integers, e.g. [1,1,1] not [2,2,2]"
        )


def load_factorized_condition(
    path: str | Path,
    *,
    expected_system_id: str | None = None,
    expected_adsorbate_atomic_numbers: np.ndarray | None = None,
) -> FactorizedCondition:
    """Load one strict ``condition_factors.json`` file without guessing fields."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing factorized condition file: {path}. test_18 never infers primitive "
            "cells or supercell matrices from an already-built slab."
        )
    values = json.loads(path.read_text())
    if int(values.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported condition schema in {path}")
    system_id = str(values["system_id"])
    if expected_system_id is not None and system_id != expected_system_id:
        raise ValueError(f"Condition system_id {system_id!r} != {expected_system_id!r}")

    primitive = values["primitive"]
    numbers = np.asarray(primitive["atomic_numbers"], dtype=np.int64)
    fractions = np.asarray(primitive["fractional_positions"], dtype=np.float32)
    cell = _array(primitive["cell"], np.float32, (3, 3), "primitive.cell")
    if numbers.ndim != 1 or len(numbers) < 1:
        raise ValueError("primitive.atomic_numbers must be a nonempty vector")
    if not np.all((1 <= numbers) & (numbers <= 118)):
        raise ValueError("primitive.atomic_numbers contains an invalid element")
    if fractions.shape != (len(numbers), 3) or not np.isfinite(fractions).all():
        raise ValueError("primitive.fractional_positions must have shape [P,3]")
    if np.any(fractions < -1.0e-6) or np.any(fractions >= 1.0 + 1.0e-6):
        raise ValueError("primitive fractional positions must lie in [0,1)")
    if abs(float(np.linalg.det(cell))) < 1.0e-8:
        raise ValueError("primitive.cell must be invertible")

    surface = values["surface"]
    miller = _array(surface["miller_index"], np.int64, (3,), "miller_index")
    _validate_miller(miller)
    miller_conventional_cell = _array(
        surface["conventional_cell"],
        np.float32,
        (3, 3),
        "surface.conventional_cell",
    )
    if abs(float(np.linalg.det(miller_conventional_cell))) < 1.0e-8:
        raise ValueError("surface.conventional_cell must be invertible")
    termination_shift = float(surface["termination_shift"])
    if not np.isfinite(termination_shift):
        raise ValueError("termination_shift must be finite")

    construction = values["construction"]
    matrix_raw = np.asarray(construction["supercell_matrix"])
    if matrix_raw.shape != (3, 3) or not np.equal(matrix_raw, np.rint(matrix_raw)).all():
        raise ValueError("supercell_matrix must be a 3x3 integer matrix")
    matrix = matrix_raw.astype(np.int64)
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1 or abs(determinant - round(determinant)) > 1.0e-6:
        raise ValueError("supercell_matrix must have a nonzero integer determinant")
    layers = int(construction["slab_layers"])
    vacuum = float(construction["vacuum_A"])
    strain = _array(
        construction.get("strain_voigt", [0.0] * 6),
        np.float32,
        (6,),
        "strain_voigt",
    )
    adsorbates_per_cell = int(construction.get("adsorbates_per_cell", 1))
    if layers < 1 or vacuum < 0 or not np.isfinite(vacuum):
        raise ValueError("slab_layers must be positive and vacuum_A nonnegative")
    if adsorbates_per_cell < 1:
        raise ValueError("adsorbates_per_cell must be positive")

    adsorbate = values["adsorbate"]
    ads_numbers = np.asarray(adsorbate["atomic_numbers"], dtype=np.int64)
    if ads_numbers.ndim != 1 or len(ads_numbers) < 1:
        raise ValueError("adsorbate.atomic_numbers must be a nonempty vector")
    if not np.all((1 <= ads_numbers) & (ads_numbers <= 118)):
        raise ValueError("adsorbate.atomic_numbers contains an invalid element")
    if expected_adsorbate_atomic_numbers is not None and not np.array_equal(
        ads_numbers, np.asarray(expected_adsorbate_atomic_numbers, dtype=np.int64)
    ):
        raise ValueError(
            "adsorbate.atomic_numbers does not match the ordered tag-2 atoms in inputs.npz"
        )

    return FactorizedCondition(
        system_id=system_id,
        primitive_atomic_numbers=numbers,
        primitive_fractional_positions=np.mod(fractions, 1.0),
        primitive_cell=cell,
        miller_index=miller,
        miller_conventional_cell=miller_conventional_cell,
        termination_shift=termination_shift,
        top=bool(surface["top"]),
        supercell_matrix=matrix,
        slab_layers=layers,
        vacuum_A=vacuum,
        strain_voigt=strain,
        adsorbates_per_cell=adsorbates_per_cell,
        adsorbate_atomic_numbers=ads_numbers,
    )


def collate_factorized_conditions(
    conditions: Sequence[FactorizedCondition],
) -> dict[str, Tensor]:
    """Pad only the primitive atom set; all construction variables stay explicit."""
    if not conditions:
        raise ValueError("At least one factorized condition is required")
    batch = len(conditions)
    max_primitive_atoms = max(item.primitive_atom_count for item in conditions)
    primitive_numbers = np.zeros((batch, max_primitive_atoms), dtype=np.int64)
    primitive_fractions = np.zeros((batch, max_primitive_atoms, 3), dtype=np.float32)
    primitive_mask = np.zeros((batch, max_primitive_atoms), dtype=bool)
    for index, item in enumerate(conditions):
        count = item.primitive_atom_count
        primitive_numbers[index, :count] = item.primitive_atomic_numbers
        primitive_fractions[index, :count] = item.primitive_fractional_positions
        primitive_mask[index, :count] = True
    return {
        "primitive_atomic_numbers": torch.from_numpy(primitive_numbers),
        "primitive_fractional_positions": torch.from_numpy(primitive_fractions),
        "primitive_atom_mask": torch.from_numpy(primitive_mask),
        "primitive_cell": torch.from_numpy(
            np.stack([item.primitive_cell for item in conditions])
        ),
        "miller_index": torch.from_numpy(
            np.stack([item.miller_index for item in conditions])
        ),
        "miller_conventional_cell": torch.from_numpy(
            np.stack([item.miller_conventional_cell for item in conditions])
        ),
        "termination_shift": torch.tensor(
            [[item.termination_shift] for item in conditions], dtype=torch.float32
        ),
        "top": torch.tensor([[item.top] for item in conditions], dtype=torch.float32),
        "supercell_matrix": torch.from_numpy(
            np.stack([item.supercell_matrix for item in conditions])
        ),
        "slab_layers": torch.tensor(
            [[item.slab_layers] for item in conditions], dtype=torch.float32
        ),
        "vacuum_A": torch.tensor(
            [[item.vacuum_A] for item in conditions], dtype=torch.float32
        ),
        "strain_voigt": torch.from_numpy(
            np.stack([item.strain_voigt for item in conditions])
        ),
        "adsorbates_per_cell": torch.tensor(
            [[item.adsorbates_per_cell] for item in conditions], dtype=torch.float32
        ),
    }
