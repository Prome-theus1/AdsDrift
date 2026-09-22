"""OC20-compatible anomaly checks and slab-symmetry structure matching.

The connectivity convention mirrors ``fairchem``/the project's Dense audit.
Surface operations are recovered directly from the finite reference slab by
enumerating small two-dimensional unimodular lattice operations and matching
tag-preserving atoms.  This avoids making inference depend on ``spglib``.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from ase.geometry import find_mic
from ase.neighborlist import NeighborList, natural_cutoffs
from scipy.optimize import linear_sum_assignment


def connectivity(atoms, multiplier: float = 1.0) -> np.ndarray:
    neighbors = NeighborList(
        natural_cutoffs(atoms, mult=multiplier),
        skin=0.3,
        self_interaction=False,
        bothways=True,
    )
    neighbors.update(atoms)
    return neighbors.get_connectivity_matrix(sparse=False).astype(bool)


def anomalies(initial, final, tags: np.ndarray, clean_slab) -> dict[str, bool]:
    tags = np.asarray(tags)
    ads = np.flatnonzero(tags == 2)
    slab = np.flatnonzero(tags != 2)
    frozen = np.flatnonzero(tags == 0)
    full = connectivity(final)
    loose = connectivity(final, 1.5)
    final_slab = final[slab]
    final_strict = connectivity(final_slab)
    clean_strict = connectivity(clean_slab)
    final_loose = connectivity(final_slab, 1.5)
    clean_loose = connectivity(clean_slab, 1.5)
    return {
        "dissociated": not np.array_equal(
            connectivity(initial[ads]), connectivity(final[ads])
        ),
        "desorbed": not bool(loose[np.ix_(ads, slab)].any()),
        "intercalated": bool(full[np.ix_(ads, frozen)].any()),
        "surface_changed": bool(
            (final_strict & ~clean_loose).any()
            or (clean_strict & ~final_loose).any()
        ),
    }


class SlabGeometry:
    def __init__(self, cell: np.ndarray):
        self.cell = np.asarray(cell, dtype=float)
        self.inverse = np.linalg.inv(self.cell)

    def distance(self, displacement: np.ndarray) -> np.ndarray:
        shape = np.asarray(displacement).shape[:-1]
        flat = np.asarray(displacement, dtype=float).reshape(-1, 3)
        return find_mic(flat, self.cell, pbc=[True, True, False])[1].reshape(shape)


def _lattice_rotations(cell: np.ndarray, search_radius: int = 2) -> list[np.ndarray]:
    """Enumerate 2-D unimodular operations preserving the Cartesian metric."""
    inverse = np.linalg.inv(cell)
    result = []
    seen = set()
    values = range(-search_radius, search_radius + 1)
    for entries in product(values, repeat=4):
        plane = np.asarray(entries, dtype=int).reshape(2, 2)
        determinant = int(round(np.linalg.det(plane)))
        if abs(determinant) != 1:
            continue
        rotation = np.eye(3, dtype=int)
        rotation[:2, :2] = plane
        matrix = inverse @ rotation.T @ cell
        if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6, rtol=0):
            continue
        key = tuple(rotation.reshape(-1))
        if key not in seen:
            seen.add(key)
            result.append(rotation)
    if not result:
        raise ValueError("could not recover even the identity lattice operation")
    return result


def surface_operations(reference, slab_tags: np.ndarray, geometry: SlabGeometry):
    """Recover tag-preserving slab rotations and fractional translations."""
    cell = reference.cell.array
    fractional = reference.get_scaled_positions(wrap=False)
    types = reference.numbers * 3 + np.asarray(slab_tags)
    normal = np.cross(cell[0], cell[1])
    normal /= np.linalg.norm(normal)
    unique_types, counts = np.unique(types, return_counts=True)
    anchor_type = unique_types[np.argmin(counts)]
    anchor_ids = np.flatnonzero(types == anchor_type)
    anchor_source = int(anchor_ids[0])
    operations = []
    seen = set()
    for rotation in _lattice_rotations(cell):
        matrix = geometry.inverse @ rotation.T @ cell
        if np.linalg.norm(normal @ matrix - normal) > 1e-5:
            continue
        rotated_fractional = fractional @ rotation.T
        for anchor_target in anchor_ids:
            translation = fractional[anchor_target] - rotated_fractional[anchor_source]
            translation[:2] -= np.floor(translation[:2])
            transformed_fractional = rotated_fractional + translation
            translation[2] -= np.rint(
                np.mean(transformed_fractional[:, 2] - fractional[:, 2])
            )
            offset = translation @ cell
            positions = reference.positions @ matrix + offset
            order = np.zeros(len(reference), dtype=int)
            valid = True
            for atom_type in unique_types:
                ids = np.flatnonzero(types == atom_type)
                distances = geometry.distance(
                    positions[ids, None] - reference.positions[None, ids]
                )
                rows, columns = linear_sum_assignment(distances**2)
                if distances[rows, columns].max(initial=0.0) > 0.03:
                    valid = False
                    break
                order[ids[columns]] = ids[rows]
            if not valid:
                continue
            key = (
                tuple(np.round(matrix, 8).reshape(-1)),
                tuple(int(value) for value in order),
            )
            if key not in seen:
                seen.add(key)
                operations.append((matrix, offset, order))
    if not operations:
        raise ValueError("no tag-preserving surface operation found")
    return operations


def _compare_ads(
    left: np.ndarray,
    right: np.ndarray,
    numbers: np.ndarray,
    geometry: SlabGeometry,
    rms_tolerance: float,
    max_tolerance: float,
) -> bool:
    errors = []
    for number in np.unique(numbers):
        ids = np.flatnonzero(numbers == number)
        distances = geometry.distance(left[ids, None] - right[None, ids])
        rows, columns = linear_sum_assignment(distances**2)
        errors.extend(distances[rows, columns].tolist())
    errors = np.asarray(errors)
    return bool(
        np.sqrt(np.mean(errors**2)) <= rms_tolerance
        and errors.max(initial=0.0) <= max_tolerance
    )


def cluster(
    records: list[dict],
    reference_slab,
    tags: np.ndarray,
    tolerance_A: float,
    *,
    use_symmetry: bool = True,
):
    """Lowest-representative clustering with 2-D PBC and optional symmetry."""
    geometry = SlabGeometry(reference_slab.cell.array)
    tags = np.asarray(tags)
    slab = np.flatnonzero(tags != 2)
    ads = np.flatnonzero(tags == 2)
    movable = np.flatnonzero(tags[slab] == 1)
    numbers = records[0]["atoms"].numbers[ads]
    operations = (
        surface_operations(reference_slab, tags[slab], geometry)
        if use_symmetry
        else [(np.eye(3), np.zeros(3), np.arange(len(slab)))]
    )
    representatives = []
    assignments = []
    for record in records:
        positions = record["atoms"].positions
        variants = [
            (positions[ads] @ matrix + offset, (positions[slab] @ matrix + offset)[order])
            for matrix, offset, order in operations
        ]
        assigned = None
        for representative_index, representative in enumerate(representatives):
            target = representative["atoms"].positions
            for transformed_ads, transformed_slab in variants:
                if not _compare_ads(
                    transformed_ads,
                    target[ads],
                    numbers,
                    geometry,
                    tolerance_A,
                    tolerance_A * 2,
                ):
                    continue
                distances = geometry.distance(
                    transformed_slab[movable] - target[slab][movable]
                )
                if len(distances) == 0 or (
                    np.sqrt(np.mean(distances**2)) <= tolerance_A
                    and distances.max(initial=0.0) <= tolerance_A * 2
                ):
                    assigned = representative_index
                    break
            if assigned is not None:
                break
        if assigned is None:
            assigned = len(representatives)
            representatives.append(record)
        assignments.append(assigned)
    return representatives, assignments, len(operations)
