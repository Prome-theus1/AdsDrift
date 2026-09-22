"""Sample a conditional adsorption prior from ONE reference structure.

No relaxation, learned generator, or positive-coordinate sampling happens here.
Coordinates/cell are in Angstrom; atom order and the entire slab are preserved.
The output NPZ uses only numeric/string arrays and needs no pickle to load.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import re

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.data import covalent_radii
from ase.geometry import find_mic
from ase.io import read


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _existing_default(*relative_paths: str) -> Path:
    """Locate the demo inputs after the local project/archive reorganization."""
    paths = [PROJECT_ROOT / value for value in relative_paths]
    return next((path for path in paths if path.is_file()), paths[0])


DEFAULT_REFERENCE = _existing_default(
    "data/oc20_dense_extracted/trajs/0_1190_0/0_1190_0_heur0.traj",
    "test_1/model/0_1190_0/0_1190_0_heur0.traj",
)
DEFAULT_TAGS = _existing_default(
    "test_0/AdsDrift/data/positive_0p50eV/0_1190_0_rank01_rand4.extxyz",
    "test_1/model/drift/data/positive_0p50eV/0_1190_0_rank01_rand4.extxyz",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "test_0/AdsDrift/data/random_input_0_1190_0"
SCHEMA_VERSION = 1
ADSORBATE_BOND_FACTOR = 1.25


def prepare_contiguous_adsorbate(
        reference: Atoms,
        bond_factor: float = ADSORBATE_BOND_FACTOR,
        ) -> tuple[Atoms, dict]:
    """Return a copy whose tag=2 atoms form one PBC-unwrapped object.

    OC structures may store a chemically intact molecule on opposite sides of
    the periodic cell. Rotating those raw wrapped Cartesian coordinates turns
    a lattice-vector offset into a real molecular separation. We therefore
    infer the adsorbate covalent graph with minimum-image distances and unwrap
    that graph before applying a rigid SO(3) rotation.

    If tag=2 contains more than one covalent component, the components are
    connected by shortest minimum-image bridges. Intracomponent geometry is
    still preserved, while the whole tagged adsorbate receives one rigid pose.
    """
    if not np.isfinite(bond_factor) or bond_factor <= 0:
        raise ValueError("Adsorbate bond factor must be finite and positive.")
    tags = reference.get_tags()
    ads_indices = np.flatnonzero(tags == 2)
    if not len(ads_indices):
        raise ValueError("Reference has no tag=2 adsorbate atoms.")
    prepared = reference.copy()
    if len(ads_indices) == 1:
        return prepared, {
            "adsorbate_unwrapped_before_rotation": False,
            "adsorbate_covalent_bond_factor": bond_factor,
            "adsorbate_covalent_bond_count": 0,
            "adsorbate_covalent_component_count": 1,
            "adsorbate_component_bridge_count": 0,
            "adsorbate_unwrap_max_displacement_angstrom": 0.0,
            "adsorbate_unwrap_max_bond_error_angstrom": 0.0,
        }

    raw = np.asarray(reference.positions[ads_indices], dtype=float)
    numbers = reference.numbers[ads_indices]
    pair_i, pair_j = np.triu_indices(len(ads_indices), k=1)
    pair_delta = raw[pair_j] - raw[pair_i]
    pair_mic, pair_distance = find_mic(
        pair_delta, reference.cell, reference.pbc
    )
    thresholds = bond_factor * (
        covalent_radii[numbers[pair_i]] + covalent_radii[numbers[pair_j]]
    )
    bonded = (pair_distance > 1.0e-8) & (pair_distance <= thresholds)
    covalent_edges = [
        (int(i), int(j), np.asarray(delta, dtype=float), float(distance))
        for i, j, delta, distance in zip(
            pair_i[bonded], pair_j[bonded], pair_mic[bonded], pair_distance[bonded]
        )
    ]

    covalent_adjacency: list[list[int]] = [[] for _ in ads_indices]
    for i, j, _, _ in covalent_edges:
        covalent_adjacency[i].append(j)
        covalent_adjacency[j].append(i)
    components: list[list[int]] = []
    unseen = set(range(len(ads_indices)))
    while unseen:
        root = min(unseen)
        stack = [root]
        unseen.remove(root)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in covalent_adjacency[node]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))

    # Build a minimum spanning tree over the complete MIC graph, preferring
    # true covalent edges. Additional edges only place disconnected tagged
    # fragments in mutually nearest periodic images.
    parent = list(range(len(ads_indices)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(first: int, second: int) -> bool:
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return False
        parent[second_root] = first_root
        return True

    tree_edges: list[tuple[int, int, np.ndarray, float]] = []
    for edge in sorted(covalent_edges, key=lambda value: value[3]):
        if union(edge[0], edge[1]):
            tree_edges.append(edge)
    bridge_count = 0
    all_pairs = sorted(
        zip(pair_i, pair_j, pair_mic, pair_distance), key=lambda value: value[3]
    )
    for i, j, delta, distance in all_pairs:
        if union(int(i), int(j)):
            tree_edges.append(
                (int(i), int(j), np.asarray(delta, dtype=float), float(distance))
            )
            bridge_count += 1
        if len(tree_edges) == len(ads_indices) - 1:
            break
    if len(tree_edges) != len(ads_indices) - 1:
        raise ValueError("Could not construct a connected adsorbate MIC graph.")

    adjacency: list[list[tuple[int, np.ndarray]]] = [[] for _ in ads_indices]
    for i, j, delta, _ in tree_edges:
        adjacency[i].append((j, delta))
        adjacency[j].append((i, -delta))
    unwrapped = np.empty_like(raw)
    unwrapped[0] = raw[0]
    visited = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbour, delta in adjacency[node]:
            if neighbour in visited:
                continue
            unwrapped[neighbour] = unwrapped[node] + delta
            visited.add(neighbour)
            stack.append(neighbour)
    if len(visited) != len(ads_indices):
        raise ValueError("Adsorbate unwrapping did not visit every tagged atom.")

    max_bond_error = 0.0
    for i, j, _, mic_distance in covalent_edges:
        max_bond_error = max(
            max_bond_error,
            abs(float(np.linalg.norm(unwrapped[j] - unwrapped[i])) - mic_distance),
        )
    if max_bond_error > 1.0e-5:
        raise ValueError(
            "PBC unwrapping produced an inconsistent molecular cycle: "
            f"maximum covalent-bond error is {max_bond_error:.6g} A."
        )
    displacement = np.linalg.norm(unwrapped - raw, axis=1)
    prepared.positions[ads_indices] = unwrapped
    return prepared, {
        "adsorbate_unwrapped_before_rotation": bool(displacement.max() > 1.0e-7),
        "adsorbate_covalent_bond_factor": bond_factor,
        "adsorbate_covalent_bond_count": len(covalent_edges),
        "adsorbate_covalent_component_count": len(components),
        "adsorbate_component_bridge_count": bridge_count,
        "adsorbate_unwrap_max_displacement_angstrom": float(displacement.max()),
        "adsorbate_unwrap_max_bond_error_angstrom": float(max_bond_error),
    }


def fixed_indices(atoms: Atoms) -> np.ndarray:
    """Reject unsupported constraints instead of silently losing them."""
    indices: set[int] = set()
    for constraint in atoms.constraints:
        if not isinstance(constraint, FixAtoms):
            raise ValueError("Only ASE FixAtoms constraints are supported.")
        indices.update(int(i) for i in constraint.get_indices())
    return np.asarray(sorted(indices), dtype=np.int64)


def load_reference(reference: Path, index: int = 0,
                   tags_from: Path | None = None,
                   tags_array: np.ndarray | None = None) -> Atoms:
    """Read geometry from reference, optionally borrowing ONLY OC20 role tags."""
    if tags_from is not None and tags_array is not None:
        raise ValueError("Use either tags_from or tags_array, not both.")
    atoms = read(str(reference), index=index)
    if not isinstance(atoms, Atoms) or not len(atoms):
        raise ValueError("Reference must contain one nonempty ASE structure.")
    source_fixed = fixed_indices(atoms)
    if tags_from is not None:
        labels = read(str(tags_from), index=0)
        if not np.array_equal(atoms.numbers, labels.numbers):
            raise ValueError("Tag source has different species or atom ordering.")
        if not np.allclose(atoms.cell.array, labels.cell.array, atol=1e-7, rtol=0):
            raise ValueError("Tag source and reference have different cells.")
        if not np.array_equal(atoms.pbc, labels.pbc):
            raise ValueError("Tag source and reference have different PBC flags.")
        tags = labels.get_tags()
        # A relaxed surface may move, but the fixed layer identifies the frame.
        if np.any(tags == 0) and not np.allclose(
                atoms.positions[tags == 0], labels.positions[tags == 0],
                atol=1e-5, rtol=0):
            raise ValueError("Tag source fixed-layer coordinates do not match the reference frame.")
        atoms.set_tags(tags)
    elif tags_array is not None:
        tags = np.asarray(tags_array)
        if tags.shape != (len(atoms),):
            raise ValueError("Explicit OC20 tags have the wrong atom count.")
        atoms.set_tags(tags.astype(np.int64))
    tags = atoms.get_tags()
    if not np.isin(tags, [0, 1, 2]).all() or not np.any(tags == 2) or not np.any(tags != 2):
        raise ValueError("Need OC20 tags: 0=fixed slab, 1=movable surface, 2=adsorbate. "
                         "For an untagged .traj, supply --tags-from a matching tagged structure.")
    expected_fixed = np.flatnonzero(tags == 0)
    if atoms.constraints and not np.array_equal(source_fixed, expected_fixed):
        raise ValueError("Reference FixAtoms indices disagree with tag=0.")
    if not atoms.pbc[:2].all():
        raise ValueError("Slab must be periodic along the first two cell vectors.")
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
        raise ValueError("Reference contains non-finite coordinates/cell.")
    if abs(np.linalg.det(atoms.cell.array)) < 1e-8:
        raise ValueError("Reference cell must be nonsingular.")
    # Do not propagate reference energies, forces, momenta, or calculator data.
    clean = Atoms(numbers=atoms.numbers, positions=atoms.positions,
                  cell=atoms.cell, pbc=atoms.pbc, tags=tags)
    clean.set_constraint(FixAtoms(indices=expected_fixed))
    return clean


def load_tags_pickle(path: Path, system_id: str) -> np.ndarray:
    """Load one system's role tags from the trusted OC-Dense tags dictionary."""
    with path.open("rb") as handle:
        mapping = pickle.load(handle)
    if not isinstance(mapping, dict) or system_id not in mapping:
        raise ValueError(f"System {system_id!r} is absent from the tags dictionary.")
    tags = np.asarray(mapping[system_id])
    if tags.ndim != 1 or not np.isin(tags, [0, 1, 2]).all():
        raise ValueError(f"Invalid OC20 role tags for system {system_id!r}.")
    return tags.astype(np.int64)


def quaternion_rotation(q: np.ndarray) -> np.ndarray:
    """Unit quaternion in (w,x,y,z) order -> proper rotation matrix."""
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-15:
        raise ValueError("Quaternion must be a finite nonzero vector of length four.")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def stratified_cells(count: int) -> np.ndarray:
    """Exactly count equal-area rectangles (u_min,u_max,v_min,v_max).

    A square count gives a square grid: 100 -> 10x10. Other counts use rows
    with unequal heights but equal cell areas, without dropping a partial row.
    """
    if count <= 0:
        raise ValueError("Stratum count must be positive.")
    rows = max(1, math.isqrt(count))
    columns = np.full(rows, count // rows, dtype=int)
    columns[:count % rows] += 1
    cells = []
    v0 = 0.0
    for cols in columns:
        v1 = v0 + cols / count
        for col in range(cols):
            cells.append([col / cols, (col + 1) / cols, v0, v1])
        v0 = v1
    return np.asarray(cells)


def decode_noise(reference: Atoms, z: np.ndarray, gap_min: float,
                 gap_max: float, uv_bounds: np.ndarray | None = None,
                 gap_fraction_bounds: np.ndarray | None = None,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P(c,z): Gaussian -> fractional COM position, clearance, rigid SO(3) pose.

    The reference adsorbate must already be a contiguous, unwrapped molecule.
    No independent per-atom wrapping is done, so its internal geometry survives.
    """
    z = np.asarray(z, dtype=float)
    if z.shape != (7,) or not np.isfinite(z).all():
        raise ValueError("Each proposal needs seven finite Gaussian values.")
    u = np.array([0.5 * (1 + math.erf(float(v) / math.sqrt(2))) for v in z[:3]])
    u = np.clip(u, 0.0, np.nextafter(1.0, 0.0))
    if uv_bounds is not None:
        bounds = np.asarray(uv_bounds, dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all() or not (
                0 <= bounds[0] < bounds[1] <= 1 + 1e-12 and
                0 <= bounds[2] < bounds[3] <= 1 + 1e-12):
            raise ValueError("Invalid lateral stratum bounds.")
        u[:2] = bounds[[0, 2]] + u[:2] * (bounds[[1, 3]] - bounds[[0, 2]])
        u[:2] = np.minimum(u[:2], np.nextafter(1.0, 0.0))
    if gap_fraction_bounds is not None:
        bounds = np.asarray(gap_fraction_bounds, dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or not (
                0 <= bounds[0] < bounds[1] <= 1 + 1e-12):
            raise ValueError("Invalid height stratum bounds.")
        u[2] = bounds[0] + u[2] * (bounds[1] - bounds[0])
        u[2] = min(u[2], np.nextafter(1.0, 0.0))
    gap = gap_min + (gap_max - gap_min) * u[2]
    rotation = quaternion_rotation(z[3:])
    q = z[3:] / np.linalg.norm(z[3:])
    cell = reference.cell.array
    normal = np.cross(cell[0], cell[1])
    normal /= np.linalg.norm(normal)
    if np.dot(normal, cell[2]) < 0:
        normal *= -1
    ads = reference.get_tags() == 2
    slab = ~ads
    masses = reference.get_masses()[ads]
    center = np.average(reference.positions[ads], axis=0, weights=masses)
    relative = (reference.positions[ads] - center) @ rotation.T
    top = np.max(reference.positions[slab] @ normal)
    lateral = u[0] * cell[0] + u[1] * cell[1]
    # Clearance measures the LOWEST adsorbate atom above the highest slab atom.
    translation = lateral + normal * (top + gap - np.min(relative @ normal))
    positions = reference.positions.copy()
    positions[ads] = relative + translation
    return positions, np.array([u[0], u[1], gap]), q


def minimum_contact(reference: Atoms, positions: np.ndarray,
                    covalent_factor: float, min_distance: float) -> tuple[bool, float]:
    """Check all adsorbate/slab pairs including periodic image contacts."""
    ads = reference.get_tags() == 2
    delta = positions[ads, None, :] - positions[None, ~ads, :]
    _, distances = find_mic(delta.reshape(-1, 3), reference.cell, reference.pbc)
    distances = distances.reshape(np.count_nonzero(ads), np.count_nonzero(~ads))
    radii = covalent_radii[reference.numbers]
    threshold = np.maximum(min_distance,
                           covalent_factor * (radii[ads, None] + radii[None, ~ads]))
    return bool(np.all(distances >= threshold)), float(distances.min())


def generate_batch(reference: Atoms, *, count: int = 100, seed: int = 0,
                   gap_min: float = 1.2, gap_max: float = 3.0,
                   covalent_factor: float = 0.75, min_distance: float = 1.0,
                   vacuum_margin: float = 1.0, max_attempts_per_sample: int = 100,
                   system_id: str = "unspecified", sampling: str = "stratified",
                   min_lateral_separation: float = 0.1) -> tuple[dict, dict]:
    """Return accepted prior structures and provenance; no files are modified."""
    if count <= 0 or max_attempts_per_sample <= 0 or seed < 0:
        raise ValueError("count/max-attempts must be positive and seed nonnegative.")
    settings = np.array([gap_min, gap_max, covalent_factor, min_distance,
                         vacuum_margin, min_lateral_separation])
    if not np.isfinite(settings).all() or gap_min <= 0 or gap_max < gap_min:
        raise ValueError("Require finite settings and 0 < gap-min <= gap-max.")
    if min(covalent_factor, min_distance, vacuum_margin, min_lateral_separation) < 0:
        raise ValueError("Contact thresholds and vacuum margin must be nonnegative.")
    if sampling not in {"stratified", "iid"}:
        raise ValueError("Sampling must be stratified or iid.")
    # Use only the PBC-unwrapped copy below. Both the rotation center and all
    # relative adsorbate coordinates are therefore one contiguous object.
    reference, unwrap_report = prepare_contiguous_adsorbate(reference)
    tags = reference.get_tags()
    ads = tags == 2
    slab = ~ads
    normal = np.cross(reference.cell[0], reference.cell[1])
    normal /= np.linalg.norm(normal)
    if np.dot(normal, reference.cell[2]) < 0:
        normal *= -1
    # Keep adsorbates away from the slab image ABOVE, including tilted c cells.
    ceiling = (np.min(reference.positions[slab] @ normal)
               + np.dot(reference.cell[2], normal) - vacuum_margin)
    rng = np.random.default_rng(seed)
    cells = stratified_cells(count) if sampling == "stratified" else np.tile([0, 1, 0, 1], (count, 1))
    stratum_ids = rng.permutation(count) if sampling == "stratified" else np.arange(count)
    sample_bounds = cells[stratum_ids]
    rejected_contact = rejected_vacuum = rejected_lateral = attempts = 0
    # A very low height bin may be incompatible with one particular lateral
    # cell after collision filtering.  Re-pair height bins and lateral cells
    # as a whole instead of silently dropping either coverage requirement.
    max_batch_assignments = 12 if sampling == "stratified" else 1
    for batch_assignment in range(max_batch_assignments):
        height_bin_ids = rng.permutation(count) if sampling == "stratified" else np.arange(count)
        height_bounds = (np.column_stack([height_bin_ids, height_bin_ids + 1]) / count
                         if sampling == "stratified" else np.tile([0, 1], (count, 1)))
        positions, noise, placement = [], [], []
        quaternions, contacts, attempts_used = [], [], []
        assignment_complete = True
        for sample_index, (bounds, height_interval) in enumerate(zip(sample_bounds, height_bounds)):
            for _ in range(max_attempts_per_sample):
                attempts += 1
                z = rng.standard_normal(7)
                proposed, uv_gap, quaternion = decode_noise(
                    reference, z, gap_min, gap_max, bounds, height_interval)
                if reference.pbc[2] and np.max(proposed[ads] @ normal) > ceiling:
                    rejected_vacuum += 1
                    continue
                valid, distance = minimum_contact(reference, proposed, covalent_factor, min_distance)
                if not valid:
                    rejected_contact += 1
                    continue
                if placement and min_lateral_separation > 0:
                    delta_uv = np.asarray(placement)[:, :2] - uv_gap[:2]
                    _, lateral_distances = find_mic(delta_uv @ reference.cell.array[:2],
                                                   reference.cell, [True, True, False])
                    if np.min(lateral_distances) < min_lateral_separation:
                        rejected_lateral += 1
                        continue
                positions.append(proposed)
                noise.append(z)
                placement.append(uv_gap)
                quaternions.append(quaternion)
                contacts.append(distance)
                attempts_used.append(attempts)
                break
            else:
                assignment_complete = False
                break
        if assignment_complete:
            break
    else:
        raise ValueError(f"Could not fill all lateral and height strata after "
                         f"{max_batch_assignments} batch assignments and "
                         f"{max_attempts_per_sample} attempts per sample. "
                         "No incomplete batch was saved. Check gap/contact/separation "
                         "settings and available vacuum.")
    batch = {
        "schema_version": np.array(SCHEMA_VERSION, dtype=np.int64),
        "system_id": np.array(system_id),
        "positions": np.asarray(positions, dtype=np.float64),
        "reference_positions": reference.positions.copy(),
        "atomic_numbers": reference.numbers.astype(np.int64),
        "cell": reference.cell.array.copy(), "pbc": reference.pbc.copy(),
        "tags": tags.astype(np.int64), "fixed_mask": tags == 0,
        "movable_mask": tags != 0, "adsorbate_mask": ads,
        "movable_indices": np.flatnonzero(tags != 0),
        "adsorbate_indices": np.flatnonzero(ads),
        "sample_ids": np.arange(count, dtype=np.int64),
        "latent_noise": np.asarray(noise),
        "placement_uv_gap": np.asarray(placement),
        "stratum_ids": stratum_ids,
        "stratum_uv_bounds": sample_bounds,
        "height_bin_ids": height_bin_ids,
        "height_fraction_bounds": height_bounds,
        "quaternions_wxyz": np.asarray(quaternions),
        "minimum_adsorbate_slab_distance": np.asarray(contacts),
        "proposal_attempt": np.asarray(attempts_used, dtype=np.int64),
        "surface_normal": normal,
    }
    report = {
        "schema_version": SCHEMA_VERSION, "system_id": system_id,
        "structure_role": "random_prior_input_not_relaxed_not_generator_output",
        "count": count, "atom_count": len(reference),
        "fixed_atom_count": int(np.count_nonzero(tags == 0)),
        "movable_surface_atom_count": int(np.count_nonzero(tags == 1)),
        "adsorbate_atom_count": int(np.count_nonzero(ads)),
        "coordinate_units": "angstrom", "seed": seed,
        "gap_range_angstrom": [gap_min, gap_max],
        "gap_definition": "lowest adsorbate atom minus highest slab atom along surface normal",
        "covalent_factor": covalent_factor, "min_distance_angstrom": min_distance,
        "vacuum_margin_angstrom": vacuum_margin,
        "sampling": sampling, "min_lateral_separation_angstrom": min_lateral_separation,
        "height_lateral_assignment_attempts": batch_assignment + 1,
        "proposal_count": attempts_used[-1], "rejected_contact": rejected_contact,
        "rejected_vacuum": rejected_vacuum, "rejected_lateral_duplicate": rejected_lateral,
        "slab_geometry": "all reference slab atoms unchanged",
        "prior": "Gaussian proposals -> one uniform uv point per assigned cell, one gap per Latin-hypercube height bin, uniform SO(3)",
        "accepted_prior": "stratification/rejection-conditioned batch; NOT an iid Gaussian coordinate prior",
        "deduplication": "lateral COM separation with 2D PBC; no crystallographic symmetry folding",
        "orientation": "uniform SO(3) before rejection; irrelevant for a single-atom adsorbate",
        "positive_coordinates_or_energies_used_for_sampling": False,
        **unwrap_report,
    }
    validate_batch(batch)
    return batch, report


def coverage_report(batch: dict, *, grid_size: int = 10,
                    dedup_tolerance: float = 0.1, probe_grid: int = 60) -> dict:
    """Audit ACTUAL generated COMs, not requested stratum metadata.

    Distances use the physical oblique cell with lateral periodic boundaries.
    The maximum probe distance is an approximation, not a rigorous cover radius.
    """
    if grid_size <= 0 or probe_grid <= 0 or not np.isfinite(dedup_tolerance) or dedup_tolerance < 0:
        raise ValueError("Invalid coverage grid or deduplication tolerance.")
    ads = np.asarray(batch["adsorbate_mask"], dtype=bool)
    masses = Atoms(numbers=batch["atomic_numbers"][ads]).get_masses()
    centers = np.average(batch["positions"][:, ads], axis=1, weights=masses)
    cell = np.asarray(batch["cell"])
    normal = np.cross(cell[0], cell[1]); normal /= np.linalg.norm(normal)
    planar_centers = centers - (centers @ normal)[:, None] * normal
    uv = np.linalg.lstsq(cell[:2].T, planar_centers.T, rcond=None)[0].T % 1.0
    n = len(uv)
    kept = []
    minimum_pair = np.inf
    # Chunk pair distances to avoid an O(N^2) resident coordinate array.
    chunk = max(1, min(128, 65536 // n))
    for first in range(0, n, chunk):
        delta = (uv[first:first + chunk, None] - uv[None, :]) @ cell[:2]
        _, distances = find_mic(delta.reshape(-1, 3), cell, [True, True, False])
        distances = distances.reshape(-1, n)
        distances[np.arange(len(distances)), first + np.arange(len(distances))] = np.inf
        minimum_pair = min(minimum_pair, float(distances.min()))
        for local, row in enumerate(distances):
            i = first + local
            if not kept or np.all(row[kept] >= dedup_tolerance):
                kept.append(i)
    counts, _, _ = np.histogram2d(uv[kept, 0], uv[kept, 1], bins=grid_size, range=[[0, 1], [0, 1]])
    occupied = int(np.count_nonzero(counts))
    axis = (np.arange(probe_grid) + 0.5) / probe_grid
    probes = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    nearest = []
    for first in range(0, len(probes), chunk):
        displacements = (probes[first:first + chunk, None] - uv[None, kept]) @ cell[:2]
        _, distances = find_mic(displacements.reshape(-1, 3), cell, [True, True, False])
        nearest.extend(distances.reshape(-1, len(kept)).min(axis=1).tolist())
    return {
        "sample_count": n, "periodic_unique_lateral_positions": len(kept),
        "dedup_tolerance_angstrom": dedup_tolerance,
        "dedup_definition": "greedy lateral COM distance, including translations by a/b; NOT crystal symmetry",
        "grid_shape": [grid_size, grid_size], "occupied_grid_cells_after_dedup": occupied,
        "total_grid_cells": grid_size**2,
        "grid_coverage_fraction_after_dedup": occupied / grid_size**2,
        "grid_counts_after_dedup": counts.astype(int).tolist(),
        "minimum_pairwise_lateral_distance_angstrom": minimum_pair if n > 1 else None,
        "probe_grid_shape": [probe_grid, probe_grid],
        "max_probe_distance_angstrom": float(np.max(nearest)),
        "p95_probe_distance_angstrom": float(np.percentile(nearest, 95)),
        "area_per_sample_angstrom2": float(np.linalg.norm(np.cross(cell[0], cell[1])) / n),
        "actual_com_uv": uv.tolist(),
        "interpretation": "spatial coverage only; not a guarantee of all chemical adsorption sites or low energy",
    }


def load_positive_markers(directory: Path, batch: dict) -> list[dict]:
    """Read endpoint markers for visualization ONLY; never feed them to sampling."""
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if str(manifest["system_id"]) != str(batch["system_id"].item()):
        raise ValueError("Positive endpoint system_id differs from the random input batch.")
    cell = np.asarray(batch["cell"])
    normal = np.cross(cell[0], cell[1]); normal /= np.linalg.norm(normal)
    fixed = batch["fixed_mask"]
    markers = []
    # Use the manifest to avoid also loading the combined file as duplicates.
    for item in manifest["structures"]:
        path = (directory / item["file"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Positive structure path must remain inside its directory.")
        atoms = read(path, index=0)
        if not np.array_equal(atoms.numbers, batch["atomic_numbers"]) or not np.array_equal(atoms.get_tags(), batch["tags"]):
            raise ValueError(f"Positive species/tags/order mismatch: {path.name}")
        if not np.allclose(atoms.cell.array, cell, atol=1e-7, rtol=0) or not np.array_equal(atoms.pbc, batch["pbc"]):
            raise ValueError(f"Positive cell/PBC mismatch: {path.name}")
        if not np.allclose(atoms.positions[fixed], batch["reference_positions"][fixed], atol=1e-5, rtol=0):
            raise ValueError(f"Positive fixed-layer frame mismatch: {path.name}")
        ads = batch["adsorbate_mask"]
        center = np.average(atoms.positions[ads], axis=0, weights=atoms.get_masses()[ads])
        planar = center - np.dot(center, normal) * normal
        uv = np.linalg.lstsq(cell[:2].T, planar, rcond=None)[0] % 1
        initial_centers = np.average(batch["positions"][:, ads], axis=1,
                                     weights=atoms.get_masses()[ads])
        differences = initial_centers - center
        differences -= (differences @ normal)[:, None] * normal
        _, distances = find_mic(differences, cell, [True, True, False])
        markers.append({"label": f"P{int(item['rank'])}", "config_id": item["config_id"],
                        "file": str(path), "uv": uv.tolist(),
                        "adsorbate_center_cartesian": center.tolist(),
                        "delta_E_eV": float(item["delta_E_eV"]),
                        "nearest_random_lateral_distance_angstrom": float(distances.min()),
                        "nearest_random_sample_id": int(batch["sample_ids"][distances.argmin()])})
    return markers


def write_coverage_plot(batch: dict, coverage: dict, destination: Path,
                        positive_markers: list[dict] | None = None) -> None:
    """Diagnostic image: physical cell coverage + an independent occupancy grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cell = batch["cell"]
    e1 = cell[0] / np.linalg.norm(cell[0])
    normal = np.cross(cell[0], cell[1]); normal /= np.linalg.norm(normal)
    e2 = np.cross(normal, e1)
    projection = np.stack([e1, e2], axis=1)
    ab = cell[:2] @ projection
    uv = np.asarray(coverage["actual_com_uv"])
    xy = uv @ ab
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    left, right = axes
    distinct_bounds = np.unique(batch["stratum_uv_bounds"], axis=0)
    for bounds in distinct_bounds:
        u0, u1, v0, v1 = bounds
        polygon = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1], [u0, v0]]) @ ab
        left.plot(*polygon.T, color="#d8dde3", linewidth=0.5, zorder=1)
    boundary = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]) @ ab
    left.plot(*boundary.T, color="#303845", linewidth=1.4, zorder=2)
    surface = batch["reference_positions"][batch["tags"] == 1]
    projected = surface - (surface @ normal)[:, None] * normal
    surface_uv = np.linalg.lstsq(cell[:2].T, projected.T, rcond=None)[0].T % 1
    surface_xy = surface_uv @ ab
    left.scatter(*surface_xy.T, marker="x", color="#747e89", s=45, label="Reference surface atoms", zorder=3)
    left.scatter(*xy.T, color="#087da4", s=19, label="Random adsorbate positions", zorder=4)
    if positive_markers:
        positive_xy = np.asarray([marker["uv"] for marker in positive_markers]) @ ab
        left.scatter(*positive_xy.T, marker="*", color="#cd302f", s=150,
                     edgecolors="white", linewidths=.8, zorder=6, label="Relaxed endpoints (P1-P4)")
        for marker, point in zip(positive_markers, positive_xy):
            left.annotate(marker["label"], point, xytext=(6, 7), textcoords="offset points",
                          fontsize=10, color="#a51818", weight="bold", zorder=7,
                          bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": .8})
    placement_title = ("Physical periodic cell: one position per region"
                       if len(distinct_bounds) == len(uv) else "Physical periodic cell: random positions")
    left.set(xlabel="In-plane X (Angstrom)", ylabel="In-plane Y (Angstrom)",
             title=placement_title, aspect="equal")
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3 if positive_markers else 2,
               frameon=False, fontsize=9)
    counts = np.asarray(coverage["grid_counts_after_dedup"])
    plot = right.imshow(counts.T, origin="lower", extent=[0, 1, 0, 1],
                        interpolation="nearest", cmap="Blues", vmin=0, vmax=max(2, counts.max()))
    right.set(xlabel="Fraction along cell vector a", ylabel="Fraction along cell vector b",
              title=f"After dedup: {coverage['occupied_grid_cells_after_dedup']}/{coverage['total_grid_cells']} regions occupied")
    if len(counts) <= 12:
        for i in range(len(counts)):
            for j in range(len(counts)):
                right.text((i + .5)/len(counts), (j + .5)/len(counts), str(counts[i, j]),
                           ha="center", va="center", fontsize=8, color="#152a40")
    fig.colorbar(plot, ax=right, label="Unique lateral positions per region", shrink=.8)
    fig.suptitle(f"Initialization coverage | {coverage['sample_count']} structures | "
                 f"{coverage['periodic_unique_lateral_positions']} distinct lateral positions", fontsize=13)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def write_diagnostics(batch: dict, coverage: dict, output_dir: Path,
                      positive_dir: Path | None = None) -> None:
    write_coverage_plot(batch, coverage, output_dir / "coverage.png")
    if positive_dir is not None:
        markers = load_positive_markers(positive_dir, batch)
        write_coverage_plot(batch, coverage, output_dir / "coverage_with_positives.png", markers)
        (output_dir / "positive_markers.json").write_text(
            json.dumps({"usage": "visualization_only_not_sampling", "markers": markers}, indent=2) + "\n",
            encoding="utf-8")


def validate_batch(batch: dict) -> None:
    """Validate the format shared by the sampler, exporter, and future trainer."""
    if int(np.asarray(batch["schema_version"]).item()) != SCHEMA_VERSION:
        raise ValueError("Unsupported random-input schema version.")
    numbers = np.asarray(batch["atomic_numbers"])
    tags = np.asarray(batch["tags"])
    positions = np.asarray(batch["positions"])
    n = len(numbers)
    if numbers.shape != (n,) or not np.issubdtype(numbers.dtype, np.integer) or not np.all((numbers > 0) & (numbers < len(covalent_radii))):
        raise ValueError("Invalid atomic_numbers.")
    if positions.ndim != 3 or positions.shape[1:] != (n, 3) or not len(positions):
        raise ValueError("positions must have shape (B, N, 3), with B > 0.")
    if tags.shape != (n,) or not np.isin(tags, [0, 1, 2]).all() or not np.any(tags == 2) or not np.any(tags != 2):
        raise ValueError("Invalid OC20 tags.")
    for key, expected in [("fixed_mask", tags == 0), ("movable_mask", tags != 0),
                          ("adsorbate_mask", tags == 2)]:
        if not np.array_equal(batch[key], expected):
            raise ValueError(f"{key} disagrees with tags.")
    for key, shape in [("reference_positions", (n, 3)), ("cell", (3, 3))]:
        if np.shape(batch[key]) != shape or not np.isfinite(batch[key]).all():
            raise ValueError(f"Invalid {key}.")
    if not np.isfinite(positions).all() or abs(np.linalg.det(batch["cell"])) < 1e-8:
        raise ValueError("Non-finite coordinates or singular cell.")
    if np.shape(batch["pbc"]) != (3,) or not np.asarray(batch["pbc"], bool)[:2].all():
        raise ValueError("Invalid slab PBC flags.")
    if not np.allclose(positions[:, tags == 0], np.asarray(batch["reference_positions"])[None, tags == 0], atol=1e-5, rtol=0):
        raise ValueError("Fixed atoms have moved.")
    ads_indices = np.flatnonzero(tags == 2)
    if len(ads_indices) > 1:
        pair_i, pair_j = np.triu_indices(len(ads_indices), k=1)
        reference_adsorbate = np.asarray(batch["reference_positions"])[ads_indices]
        expected = np.linalg.norm(
            reference_adsorbate[pair_j] - reference_adsorbate[pair_i], axis=1
        )
        generated_adsorbates = positions[:, ads_indices]
        actual = np.linalg.norm(
            generated_adsorbates[:, pair_j] - generated_adsorbates[:, pair_i],
            axis=2,
        )
        maximum_error = float(np.max(np.abs(actual - expected[None, :])))
        if maximum_error > 1.0e-5:
            raise ValueError(
                "Random initialization changed rigid adsorbate geometry: "
                f"maximum pair-distance error is {maximum_error:.6g} A."
            )
    if np.shape(batch["sample_ids"]) != (len(positions),):
        raise ValueError("sample_ids length disagrees with positions.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, help="One slab+adsorbate .traj/.extxyz; default: raw heur0.")
    parser.add_argument("--reference-index", type=int, default=0, help="Frame index; default is INITIAL frame 0.")
    parser.add_argument("--tags-from", type=Path, help="Matching tagged structure; only roles are borrowed.")
    parser.add_argument("--tags-pickle", type=Path,
                        help="Trusted OC-Dense system_id -> tags dictionary; alternative to --tags-from.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--system-id", help="Default: inferred from the reference filename.")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gap-min", type=float, default=1.2)
    parser.add_argument("--gap-max", type=float, default=3.0)
    parser.add_argument("--covalent-factor", type=float, default=0.75)
    parser.add_argument("--min-distance", type=float, default=1.0)
    parser.add_argument("--vacuum-margin", type=float, default=1.0)
    parser.add_argument("--max-attempts-per-sample", type=int, default=100)
    parser.add_argument("--sampling", choices=["stratified", "iid"], default="stratified",
                        help="Default: one random point per equal-area region; 100 points -> 10x10.")
    parser.add_argument("--min-lateral-separation", type=float, default=0.1,
                        help="Minimum adsorbate COM separation in Angstrom, with lateral PBC.")
    parser.add_argument("--coverage-grid", type=int, default=10,
                        help="Independent GxG occupancy audit after periodic position deduplication.")
    parser.add_argument("--no-plot", action="store_true", help="Skip coverage.png (avoids matplotlib dependency).")
    parser.add_argument("--positive-dir", type=Path,
                        help="Optionally overlay endpoints listed in this directory's manifest.json; visualization only.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Redraw diagnostics from existing output-dir/inputs.npz; never resample or modify the NPZ.")
    args = parser.parse_args()
    if args.tags_from is not None and args.tags_pickle is not None:
        parser.error("--tags-from and --tags-pickle are mutually exclusive")
    if args.plot_only:
        if args.no_plot:
            parser.error("--plot-only and --no-plot are mutually exclusive")
        with np.load(args.output_dir / "inputs.npz", allow_pickle=False) as archive:
            batch = {key: archive[key] for key in archive.files}
        validate_batch(batch)
        original_settings = json.loads(str(batch["metadata_json"].item()))
        coverage = coverage_report(batch, grid_size=args.coverage_grid,
                                   dedup_tolerance=original_settings["min_lateral_separation_angstrom"])
        write_diagnostics(batch, coverage, args.output_dir, args.positive_dir)
        (args.output_dir / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output_dir": str(args.output_dir.resolve()), "resampled": False,
                          "positive_overlay": args.positive_dir is not None}, indent=2))
        return
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    reference_path = (args.reference or DEFAULT_REFERENCE).resolve()
    system_id = args.system_id or re.sub(r"_(heur|rand)\d+$", "", reference_path.stem)
    tags_path = args.tags_from
    if args.reference is None and tags_path is None and args.tags_pickle is None:
        tags_path = DEFAULT_TAGS
    tags_array = (load_tags_pickle(args.tags_pickle, system_id)
                  if args.tags_pickle is not None else None)
    reference = load_reference(reference_path, args.reference_index, tags_path, tags_array)
    batch, report = generate_batch(reference, count=args.num_samples, seed=args.seed,
        gap_min=args.gap_min, gap_max=args.gap_max, covalent_factor=args.covalent_factor,
        min_distance=args.min_distance, vacuum_margin=args.vacuum_margin,
        max_attempts_per_sample=args.max_attempts_per_sample, system_id=system_id,
        sampling=args.sampling, min_lateral_separation=args.min_lateral_separation)
    coverage = coverage_report(batch, grid_size=args.coverage_grid,
                               dedup_tolerance=args.min_lateral_separation)
    report.update({"reference": str(reference_path), "reference_index": args.reference_index,
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "tags_from": str(tags_path.resolve()) if tags_path is not None else None,
        "tags_pickle": str(args.tags_pickle.resolve()) if args.tags_pickle is not None else None,
        "coverage": {key: value for key, value in coverage.items()
                     if key not in {"actual_com_uv", "grid_counts_after_dedup"}}})
    batch["metadata_json"] = np.array(json.dumps(report, ensure_ascii=False))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output_dir / "inputs.npz", **batch)
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    if not args.no_plot:
        write_diagnostics(batch, coverage, args.output_dir, args.positive_dir)
    print(json.dumps({"output": str(args.output_dir.resolve() / "inputs.npz"),
                      "count": report["count"], "shape": list(batch["positions"].shape),
                      "proposals": report["proposal_count"],
                      "unique_lateral_positions": coverage["periodic_unique_lateral_positions"],
                      "grid_coverage_fraction": coverage["grid_coverage_fraction_after_dedup"]}, indent=2))


if __name__ == "__main__":
    main()
