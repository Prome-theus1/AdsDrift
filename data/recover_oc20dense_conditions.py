"""Recover test_18 construction factors from official OC20-Dense provenance.

The script joins three authoritative inputs:

1. ``oc20dense_mapping.pkl`` for mpid, Miller index, shift, and top/bottom;
2. official ``bulks.pkl`` for the source periodic bulk;
3. the prepared system's ``inputs.npz`` for an exact slab reconstruction check.

A sidecar is written only when the current FAIR-Chem surface builder recreates
the stored slab cell, atom order, coordinates (up to one global translation),
and OC20 tags within the configured tolerances.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import pickle
import sys
import types

import numpy as np


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _mapping_by_system(path: Path) -> dict[str, dict]:
    raw = _load_pickle(path)
    records = raw.values() if isinstance(raw, dict) else raw
    result = {}
    for record in records:
        system_id = str(record["system_id"])
        previous = result.get(system_id)
        signature = (
            str(record["mpid"]),
            tuple(int(value) for value in record["miller_idx"]),
            float(record["shift"]),
            bool(record["top"]),
            str(record["adsorbate"]),
        )
        if previous is not None:
            old_signature = (
                str(previous["mpid"]),
                tuple(int(value) for value in previous["miller_idx"]),
                float(previous["shift"]),
                bool(previous["top"]),
                str(previous["adsorbate"]),
            )
            if signature != old_signature:
                raise ValueError(f"Conflicting metadata for {system_id}")
        result[system_id] = record
    return result


def _bulk_by_mpid(path: Path) -> dict[str, dict]:
    records = _load_pickle(path)
    result = {}
    for index, record in enumerate(records):
        source = str(record["src_id"])
        if source in result:
            raise ValueError(f"Duplicate bulk src_id {source}")
        result[source] = {"index": index, **record}
    return result


def _plane_count(positions: np.ndarray, cell: np.ndarray, tolerance_A: float) -> int:
    normal = np.cross(cell[0], cell[1])
    normal /= np.linalg.norm(normal)
    heights = np.sort(positions @ normal)
    if not len(heights):
        return 0
    groups = 1
    previous = heights[0]
    for value in heights[1:]:
        if value - previous > tolerance_A:
            groups += 1
        previous = value
    return groups


def _vacuum_A(positions: np.ndarray, cell: np.ndarray) -> float:
    normal = np.cross(cell[0], cell[1])
    normal /= np.linalg.norm(normal)
    height = abs(float(cell[2] @ normal))
    projected = positions @ normal
    return max(0.0, height - float(projected.max() - projected.min()))


def _unordered_geometry_match(
    generated_positions: np.ndarray,
    generated_numbers: np.ndarray,
    reference_positions: np.ndarray,
    reference_numbers: np.ndarray,
    cell: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Match like elements after a global translation and xy minimum images.

    FAIR-Chem/Pymatgen versions may emit an equivalent slab in a different atom
    order.  Candidate translations are therefore obtained from the least
    abundant element and each element block is matched independently with the
    Hungarian algorithm.  The returned index maps generated atoms to reference
    atoms and is later used for the OC20 tag check.
    """
    from scipy.optimize import linear_sum_assignment

    if Counter(generated_numbers) != Counter(reference_numbers):
        raise ValueError("Generated and reference compositions differ")
    inverse = np.linalg.inv(cell)
    generated_fractional = generated_positions @ inverse
    reference_fractional = reference_positions @ inverse
    elements = sorted(set(int(value) for value in generated_numbers))
    anchor_element = min(elements, key=lambda value: np.sum(generated_numbers == value))
    generated_anchors = np.flatnonzero(generated_numbers == anchor_element)
    reference_anchors = np.flatnonzero(reference_numbers == anchor_element)

    best_mapping = None
    best_max_error = float("inf")
    best_mean_error = float("inf")
    # One generated anchor is sufficient: pairing it with every like-element
    # reference atom enumerates every distinct global translation.
    for generated_anchor in generated_anchors[:1]:
        for reference_anchor in reference_anchors:
            translation = (
                reference_fractional[reference_anchor]
                - generated_fractional[generated_anchor]
            )
            translation[:2] -= np.rint(translation[:2])
            mapping = np.full(len(generated_numbers), -1, dtype=np.int64)
            errors = []
            for element in elements:
                generated_indices = np.flatnonzero(generated_numbers == element)
                reference_indices = np.flatnonzero(reference_numbers == element)
                delta = (
                    reference_fractional[reference_indices][None, :, :]
                    - generated_fractional[generated_indices][:, None, :]
                    - translation[None, None, :]
                )
                delta[:, :, :2] -= np.rint(delta[:, :, :2])
                distances = np.linalg.norm(delta @ cell, axis=2)
                rows, columns = linear_sum_assignment(distances)
                mapping[generated_indices[rows]] = reference_indices[columns]
                errors.extend(distances[rows, columns].tolist())
            max_error = float(max(errors, default=0.0))
            mean_error = float(np.mean(errors)) if errors else 0.0
            if (max_error, mean_error) < (best_max_error, best_mean_error):
                best_mapping = mapping
                best_max_error = max_error
                best_mean_error = mean_error

    if best_mapping is None or np.any(best_mapping < 0):
        raise ValueError("Unable to construct a complete element-wise atom mapping")
    return best_mapping, best_max_error


def _align_equivalent_slab_cells(
    generated_cell: np.ndarray,
    reference_cell: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Align equivalent slab cells without depending on basis-vector signs.

    The first two rows remain an in-plane unimodular basis.  The third row may
    be reversed and shifted by in-plane lattice vectors.  For each small exact
    integer basis change we solve the orthogonal Procrustes problem, allowing
    both rotations and reflections because historical OC20 top/bottom slabs
    were sometimes serialized with the opposite cell handedness.
    """
    generated_cell = np.asarray(generated_cell, dtype=np.float64)
    reference_cell = np.asarray(reference_cell, dtype=np.float64)
    unique_alignments: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray, float]] = {}
    best_error = float("inf")
    for a in range(-2, 3):
        for b in range(-2, 3):
            for c in range(-2, 3):
                for d in range(-2, 3):
                    determinant_2d = a * d - b * c
                    if abs(determinant_2d) != 1:
                        continue
                    for u in range(-2, 3):
                        for v in range(-2, 3):
                            for sign in (-1, 1):
                                transform = np.asarray(
                                    [[a, b, 0], [c, d, 0], [u, v, sign]],
                                    dtype=np.int64,
                                )
                                transformed = transform @ generated_cell
                                left, _, right_t = np.linalg.svd(
                                    transformed.T @ reference_cell
                                )
                                orthogonal = left @ right_t
                                residual = transformed @ orthogonal - reference_cell
                                error = float(np.max(np.abs(residual)))
                                best_error = min(best_error, error)
                                if error <= 1.0e-3:
                                    key = tuple(np.round(orthogonal, decimals=8).ravel())
                                    previous = unique_alignments.get(key)
                                    if previous is None or error < previous[2]:
                                        unique_alignments[key] = (
                                            transform,
                                            orthogonal,
                                            error,
                                        )
    if not unique_alignments:
        # Preserve the actual closest error in the caller's diagnostic.
        return [(np.eye(3, dtype=np.int64), np.eye(3), best_error)]
    alignments = sorted(unique_alignments.values(), key=lambda item: item[2])
    if not alignments:
        raise ValueError("Unable to align slab cells")
    return alignments


def _recover_one(
    system_directory: Path,
    metadata: dict,
    bulk_record: dict,
    *,
    shift_tolerance: float,
    cell_tolerance_A: float,
    coordinate_tolerance_A: float,
    layer_tolerance_A: float,
    overwrite: bool,
) -> dict:
    try:
        # fairchem-data-oc is usable without the much larger fairchem-core
        # training package, but its ``core/__init__.py`` imports fairchem-core
        # eagerly.  Register the data package directory as a namespace package
        # so importing the self-contained slab module does not replace the
        # AdsDrift Torch/CUDA stack with fairchem-core's dependencies.
        import fairchem.data.oc as fairchem_data_oc

        core_module_name = "fairchem.data.oc.core"
        if core_module_name not in sys.modules:
            core_module = types.ModuleType(core_module_name)
            core_module.__path__ = [
                str(Path(fairchem_data_oc.__file__).resolve().parent / "core")
            ]
            core_module.__package__ = core_module_name
            sys.modules[core_module_name] = core_module
        from fairchem.data.oc.core.slab import (
            compute_slabs,
            standardize_bulk,
            tile_and_tag_atoms,
        )
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as error:
        raise RuntimeError(
            "Install fairchem-data-oc and pymatgen in the processing environment"
        ) from error

    input_path = system_directory / "random_input" / "inputs.npz"
    with np.load(input_path, allow_pickle=False) as values:
        all_numbers = np.asarray(values["atomic_numbers"], dtype=np.int64)
        tags = np.asarray(values["tags"], dtype=np.int64)
        reference_cell = np.asarray(values["cell"], dtype=np.float64)
        reference_positions = np.asarray(values["positions"][0], dtype=np.float64)
        stored_system_id = str(np.asarray(values["system_id"]).item())
    system_id = system_directory.name
    if stored_system_id != system_id:
        raise ValueError(f"stored system_id {stored_system_id!r} != {system_id!r}")
    slab_mask = tags != 2
    reference_slab_positions = reference_positions[slab_mask]
    reference_slab_numbers = all_numbers[slab_mask]
    reference_slab_tags = tags[slab_mask]

    bulk_atoms = bulk_record["atoms"].copy()
    miller = tuple(int(value) for value in metadata["miller_idx"])
    candidates = compute_slabs(
        bulk_atoms,
        max_miller=max(abs(value) for value in miller),
        specific_millers=[miller],
    )
    matches = [
        item
        for item in candidates
        if item[1] == miller
        and abs(float(item[2]) - float(metadata["shift"])) <= shift_tolerance
        and bool(item[3]) is bool(metadata["top"])
    ]
    if len(matches) != 1:
        available = [
            {
                "miller": list(item[1]),
                "shift": float(item[2]),
                "top": bool(item[3]),
            }
            for item in candidates
            if item[1] == miller
        ]
        raise ValueError(
            f"Expected one slab provenance match, found {len(matches)}; "
            f"requested={{'miller': {list(miller)}, 'shift': {float(metadata['shift'])}, "
            f"'top': {bool(metadata['top'])}}}; available={available}"
        )
    unit_slab, _, shift, top, oriented_bulk = matches[0]
    generated = tile_and_tag_atoms(unit_slab, bulk_atoms, min_ab=8.0)
    generated_numbers = np.asarray(generated.numbers, dtype=np.int64)
    generated_tags = np.asarray(generated.get_tags(), dtype=np.int64)
    if Counter(generated_numbers) != Counter(reference_slab_numbers):
        raise ValueError(
            "Rebuilt slab composition differs: "
            f"rebuilt={Counter(generated_numbers)}, reference={Counter(reference_slab_numbers)}"
        )
    cell_alignments = _align_equivalent_slab_cells(
        np.asarray(generated.cell), reference_cell
    )
    cell_basis_transform, cartesian_transform, cell_error = cell_alignments[0]
    if cell_error > cell_tolerance_A:
        generated_cell = np.asarray(generated.cell, dtype=np.float64)
        generated_lengths = np.linalg.norm(generated_cell, axis=1)
        reference_lengths = np.linalg.norm(reference_cell, axis=1)
        generated_volume = abs(float(np.linalg.det(generated_cell)))
        reference_volume = abs(float(np.linalg.det(reference_cell)))
        raise ValueError(
            f"Rebuilt cell max error {cell_error:.6g} A; "
            f"rebuilt_cell={generated_cell.tolist()}; "
            f"reference_cell={reference_cell.tolist()}; "
            f"rebuilt_lengths={generated_lengths.tolist()}; "
            f"reference_lengths={reference_lengths.tolist()}; "
            f"volumes=({generated_volume}, {reference_volume})"
        )
    best_geometry = None
    for candidate_basis, candidate_cartesian, candidate_cell_error in cell_alignments:
        aligned_generated_positions = (
            np.asarray(generated.positions) @ candidate_cartesian
        )
        candidate_mapping, candidate_coordinate_error = _unordered_geometry_match(
            aligned_generated_positions,
            generated_numbers,
            reference_slab_positions,
            reference_slab_numbers,
            reference_cell,
        )
        tag_mismatches = int(
            np.count_nonzero(
                generated_tags != reference_slab_tags[candidate_mapping]
            )
        )
        score = (tag_mismatches, candidate_coordinate_error, candidate_cell_error)
        if best_geometry is None or score < best_geometry[0]:
            best_geometry = (
                score,
                candidate_basis,
                candidate_cartesian,
                candidate_mapping,
            )
    if best_geometry is None:
        raise ValueError("No geometrically equivalent cell alignment was found")
    (_, coordinate_error, cell_error), cell_basis_transform, cartesian_transform, atom_mapping = best_geometry
    if coordinate_error > coordinate_tolerance_A:
        raise ValueError(f"Rebuilt matched-coordinate max error {coordinate_error:.6g} A")
    if not np.array_equal(generated_tags, reference_slab_tags[atom_mapping]):
        raise ValueError("Rebuilt OC20 surface/subsurface tags differ after atom matching")

    canonical = standardize_bulk(bulk_atoms)
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    primitive = SpacegroupAnalyzer(canonical, symprec=0.1).get_primitive_standard_structure()
    primitive_cell = np.asarray(primitive.lattice.matrix, dtype=np.float64)
    canonical_cell = np.asarray(canonical.lattice.matrix, dtype=np.float64)
    oriented_cell = np.asarray(oriented_bulk.lattice.matrix, dtype=np.float64)
    orientation_float = oriented_cell @ np.linalg.inv(primitive_cell)
    orientation_matrix = np.rint(orientation_float).astype(np.int64)
    orientation_error = float(np.max(np.abs(orientation_float - orientation_matrix)))
    if orientation_error > 1.0e-6:
        raise ValueError(
            f"Bulk-to-oriented-cell transform is not integer; error={orientation_error:.3g}"
        )
    unit_cell = np.asarray(AseAtomsAdaptor.get_atoms(unit_slab).cell, dtype=np.float64)
    repeats = np.asarray(
        [
            int(math.ceil(8.0 / np.linalg.norm(unit_cell[0]))),
            int(math.ceil(8.0 / np.linalg.norm(unit_cell[1]))),
            1,
        ],
        dtype=np.int64,
    )
    supercell_matrix = np.diag(repeats) @ orientation_matrix
    if round(abs(float(np.linalg.det(supercell_matrix)))) < 1:
        raise ValueError("Recovered supercell matrix is singular")

    primitive_atoms = AseAtomsAdaptor.get_atoms(primitive)
    primitive_atoms.wrap()
    record = {
        "schema_version": 1,
        "system_id": system_id,
        "primitive": {
            "atomic_numbers": primitive_atoms.numbers.tolist(),
            "fractional_positions": np.mod(
                primitive_atoms.get_scaled_positions(wrap=True), 1.0
            ).tolist(),
            "cell": primitive_cell.tolist(),
            "representation": "Pymatgen primitive standard bulk",
        },
        "surface": {
            "miller_index": list(miller),
            "miller_basis": "FAIR-Chem/Pymatgen conventional standard bulk",
            "conventional_cell": canonical_cell.tolist(),
            "termination_shift": float(shift),
            "top": bool(top),
        },
        "construction": {
            "supercell_matrix": supercell_matrix.tolist(),
            "orientation_matrix": orientation_matrix.tolist(),
            "inplane_repeat": repeats.tolist(),
            "slab_layers": _plane_count(
                np.asarray(generated.positions), np.asarray(generated.cell), layer_tolerance_A
            ),
            "vacuum_A": _vacuum_A(
                np.asarray(generated.positions), np.asarray(generated.cell)
            ),
            "strain_voigt": [0.0] * 6,
            "adsorbates_per_cell": 1,
            "fairchem_surface_parameters": {
                "min_slab_size_A": 7.0,
                "min_vacuum_size_A": 20.0,
                "center_slab": True,
                "primitive_surface": True,
                "max_normal_search": 1,
                "tile_min_ab_A": 8.0,
            },
        },
        "adsorbate": {
            "name": str(metadata["adsorbate"]),
            "atomic_numbers": all_numbers[tags == 2].tolist(),
        },
        "provenance": {
            "mapping_mpid": str(metadata["mpid"]),
            "bulk_db_index": int(bulk_record["index"]),
            "bulk_src_id": str(bulk_record["src_id"]),
            "reconstruction": {
                "cell_max_error_A": cell_error,
                "matched_coordinate_max_error_A": coordinate_error,
                "orientation_integer_error": orientation_error,
                "reference_cell_basis_transform": cell_basis_transform.tolist(),
                "cartesian_orthogonal_transform": cartesian_transform.tolist(),
            },
        },
    }
    output = system_directory / "condition_factors.json"
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(output)
    return {
        "system_id": system_id,
        "status": "written",
        "output": str(output),
        "mpid": str(metadata["mpid"]),
        "cell_max_error_A": cell_error,
        "coordinate_max_error_A": coordinate_error,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--bulks", type=Path, required=True)
    parser.add_argument("--system-id", action="append", default=[])
    parser.add_argument("--shift-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--cell-tolerance-A", type=float, default=2.0e-4)
    parser.add_argument("--coordinate-tolerance-A", type=float, default=2.0e-3)
    parser.add_argument("--layer-tolerance-A", type=float, default=0.3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    systems_root = args.systems_root.expanduser().resolve()
    mapping = _mapping_by_system(args.mapping.expanduser().resolve())
    bulks = _bulk_by_mpid(args.bulks.expanduser().resolve())
    system_ids = args.system_id or sorted(
        path.parent.name for path in systems_root.glob("*/structure_manifest.json")
    )
    results = []
    for system_id in system_ids:
        try:
            metadata = mapping[system_id]
            bulk_record = bulks[str(metadata["mpid"])]
            results.append(
                _recover_one(
                    systems_root / system_id,
                    metadata,
                    bulk_record,
                    shift_tolerance=args.shift_tolerance,
                    cell_tolerance_A=args.cell_tolerance_A,
                    coordinate_tolerance_A=args.coordinate_tolerance_A,
                    layer_tolerance_A=args.layer_tolerance_A,
                    overwrite=args.overwrite,
                )
            )
            print(json.dumps(results[-1]), flush=True)
        except Exception as error:
            failure = {
                "system_id": system_id,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            results.append(failure)
            print(json.dumps(failure), file=sys.stderr, flush=True)

    summary = {
        "systems_requested": len(system_ids),
        "written": sum(item["status"] == "written" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
        "results": results,
    }
    report = args.report.expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("systems_requested", "written", "failed")}))
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
