"""Convert a clean crystal/slab plus molecule into the generator's OC role input."""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import surface
from ase.cell import Cell
from ase.constraints import FixAtoms
from ase.io import read


def surface_normal(cell) -> np.ndarray:
    normal = np.cross(cell[0], cell[1])
    normal /= np.linalg.norm(normal)
    return normal


def check_cell(cell):
    values = np.asarray(cell)
    if not np.isfinite(values).all() or np.linalg.det(values) <= 1e-8:
        raise ValueError("Need a finite right-handed 3D cell. Ordinary XYZ needs --cell A B C (or 6 cell parameters / 9 matrix entries)")


def prepare_slab(path, *, index=0, input_kind="slab", cell_override=None,
                 miller=None, layers=None, repeat=(1, 1), vacuum=15.0,
                 fixed_layers=None, fixed_indices=None, use_input_tags=False,
                 layer_tolerance=.3) -> tuple[Atoms, dict]:
    source = read(path, index=index)
    if not isinstance(source, Atoms) or not len(source) or not np.isfinite(source.positions).all():
        raise ValueError("Input must contain one nonempty clean crystal/slab with finite coordinates")
    if not np.isin(source.numbers, np.arange(1, 119)).all():
        raise ValueError("Input contains unsupported/dummy atomic numbers")
    source = source.copy()
    original_cell = source.cell.array.copy()
    if cell_override is not None:
        values = np.asarray(cell_override, dtype=float)
        if values.size not in (3, 6, 9):
            raise ValueError("--cell needs 3 lengths, 6 cell parameters, or 9 row-major entries")
        source.set_cell(Cell.new(values.reshape(3, 3) if values.size == 9 else values), scale_atoms=False)
    check_cell(source.cell)
    if input_kind == "bulk":
        if miller is None or layers is None or layers < 1 or not any(miller):
            raise ValueError("Bulk input requires nonzero --miller H K L and positive --layers")
        if vacuum is None:
            raise ValueError("--keep-cell is for existing slabs; bulk cutting requires --vacuum")
        if source.constraints or use_input_tags:
            raise ValueError("Bulk cutting does not preserve atom constraints/tags; assign bottom layers after cutting")
        slab = surface(source, miller, layers, vacuum=vacuum, periodic=False)
    else:
        if miller is not None or layers is not None:
            raise ValueError("--miller/--layers require --input-kind bulk")
        slab = source
    if len(repeat) != 2 or min(repeat) < 1:
        raise ValueError("--repeat NX NY requires positive integers")
    slab = slab.repeat((repeat[0], repeat[1], 1))
    if vacuum is not None:
        if not np.isfinite(vacuum) or vacuum <= 0:
            raise ValueError("--vacuum is a positive vacuum thickness on EACH side, in Angstrom")
        slab.center(vacuum=vacuum, axis=2)
    check_cell(slab.cell)
    slab.set_pbc([True, True, False])
    normal = surface_normal(slab.cell)
    height = slab.positions @ normal
    unsupported = [c for c in slab.constraints if not isinstance(c, FixAtoms)]
    if unsupported:
        raise ValueError("Only FixAtoms is supported; remove partial constraints explicitly")
    source_fixed = sorted({int(i) for c in slab.constraints for i in c.get_indices()})
    if fixed_indices is not None:
        indices = np.asarray(fixed_indices, dtype=int)
        if np.any(indices < 0) or np.any(indices >= len(slab)) or len(np.unique(indices)) != len(indices):
            raise ValueError("--fixed-indices are unique zero-based indices in the final repeated slab")
        policy = "explicit_indices"
    elif use_input_tags:
        tags = slab.get_tags()
        if not np.isin(tags, [0, 1]).all() or not np.any(tags == 1):
            raise ValueError("--use-input-tags requires OC slab tags 0=fixed, 1=movable (no adsorbate)")
        indices = np.flatnonzero(tags == 0)
        if source_fixed and set(source_fixed) != set(indices):
            raise ValueError("Input tags disagree with FixAtoms constraints")
        policy = "input_oc_tags"
    elif source_fixed and fixed_layers is None:
        indices = np.asarray(source_fixed, dtype=int)
        policy = "input_FixAtoms"
    else:
        count = 1 if fixed_layers is None else fixed_layers
        if count < 0 or not np.isfinite(layer_tolerance) or layer_tolerance <= 0:
            raise ValueError("Fixed layer count must be nonnegative and layer tolerance positive")
        planes = []
        for i in np.argsort(height):
            if not planes or height[i] - height[planes[-1][0]] > layer_tolerance:
                planes.append([])
            planes[-1].append(int(i))
        if count > len(planes):
            raise ValueError("--fixed-layers exceeds the number of detected atomic planes")
        indices = np.asarray([i for plane in planes[:count] for i in plane], dtype=int)
        policy = f"bottom_{count}_planes"
    tags = np.ones(len(slab), dtype=int)
    tags[indices] = 0
    if not np.any(tags == 1):
        raise ValueError("Generator requires movable slab atoms; decrease --fixed-layers (0 allows all slab atoms to move)")
    # Strip imported calculators, velocities, arbitrary tags and molecule arrays.
    slab = Atoms(numbers=slab.numbers, positions=slab.positions, cell=slab.cell, pbc=[True, True, False], tags=tags)
    slab.set_constraint(FixAtoms(indices=indices))
    thickness = float(np.ptp(height))
    period = float(slab.cell[2] @ normal)
    report = dict(input_kind=input_kind, input_cell_A=original_cell.tolist(), cell_A=slab.cell.array.tolist(),
                  cell_override_changes_coordinates=False, miller=miller, layers=layers, repeat=list(repeat),
                  vacuum_per_side_A=vacuum, kept_input_cell=vacuum is None,
                  total_empty_normal_length_A=period-thickness, slab_thickness_A=thickness,
                  surface_normal=normal.tolist(), fixed_policy=policy, fixed_indices=indices.tolist(),
                  movable_indices=np.flatnonzero(tags == 1).tolist(), slab_atom_count=len(slab),
                  layer_tolerance_A=layer_tolerance, pbc=[True, True, False])
    return slab, report


def combine_reference(slab: Atoms, adsorbate: Atoms, gap_max: float,
                      margin: float = 1.0) -> Atoms:
    """Check space for every rigid orientation; do not change molecular geometry."""
    check_cell(slab.cell)
    normal = surface_normal(slab.cell)
    if gap_max <= 0 or not np.isfinite(gap_max):
        raise ValueError("The maximum initial gap must be positive and finite")
    extent = float(np.max(adsorbate.get_all_distances())) if len(adsorbate) > 1 else 0.
    # Requiring the whole object to fit in the exported cell also ensures it is
    # not placed beyond its top boundary in nonperiodic-z visualizations.
    available = float(slab.cell[2] @ normal - np.max(slab.positions @ normal))
    if available < gap_max + extent + margin:
        raise ValueError(f"Insufficient upper vacuum: {available:.2f} A available, {gap_max+extent+margin:.2f} A needed; increase --vacuum")
    molecule_positions = adsorbate.positions - adsorbate.positions.mean(axis=0)
    molecule_positions += .5 * (slab.cell[0] + slab.cell[1])
    molecule_positions += normal * (np.max(slab.positions @ normal) + gap_max - np.min(molecule_positions @ normal))
    result = Atoms(numbers=np.concatenate([slab.numbers, adsorbate.numbers]),
                   positions=np.concatenate([slab.positions, molecule_positions]), cell=slab.cell,
                   pbc=slab.pbc, tags=np.concatenate([slab.get_tags(), np.full(len(adsorbate), 2)]))
    result.set_constraint(FixAtoms(indices=np.flatnonzero(result.get_tags() == 0)))
    return result
