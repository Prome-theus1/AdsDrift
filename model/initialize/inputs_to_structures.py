"""Convert a random-input NPZ batch into ASE structures without re-sampling.

Optional --positions accepts future generator output (.npy), either all atoms
(B,N,3) or movable atoms (B,N_movable,3), in the same Cartesian frame/atom order.
This exporter never attaches reference/positive energies or forces to new poses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import write

if __package__:
    from AdsDrift.model.initialize.generate_random_inputs import DEFAULT_OUTPUT, validate_batch
else:
    from generate_random_inputs import DEFAULT_OUTPUT, validate_batch


def load_batch(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        batch = {key: archive[key] for key in archive.files}
    validate_batch(batch)
    return batch


def replace_positions(batch: dict, predictions: np.ndarray) -> dict:
    """Decode a generator's Cartesian output, checking and preserving fixed atoms."""
    predictions = np.asarray(predictions)
    original = batch["positions"]
    movable = np.asarray(batch["movable_mask"], dtype=bool)
    if predictions.shape == original.shape:
        positions = predictions.copy()
    elif predictions.shape == (len(original), int(movable.sum()), 3):
        positions = np.broadcast_to(batch["reference_positions"], original.shape).copy()
        positions[:, movable] = predictions
    else:
        raise ValueError("Predictions must be (B,N,3) or (B,N_movable,3) with the archive's B/order.")
    result = dict(batch, positions=positions)
    validate_batch(result)
    return result


def to_atoms(batch: dict, index: int, *, generator_output: bool = False) -> Atoms:
    """Reconstruct one full interface, keeping cell/PBC/species/tags/FixAtoms."""
    if index < 0 or index >= len(batch["positions"]):
        raise IndexError("Sample index outside the batch.")
    positions = np.asarray(batch["positions"][index], dtype=float).copy()
    fixed = np.asarray(batch["fixed_mask"], dtype=bool)
    positions[fixed] = batch["reference_positions"][fixed]
    atoms = Atoms(numbers=batch["atomic_numbers"], positions=positions,
                  cell=batch["cell"], pbc=batch["pbc"], tags=batch["tags"])
    atoms.set_constraint(FixAtoms(indices=np.flatnonzero(fixed)))
    atoms.info.update({"system_key": str(batch["system_id"].item()),
                      "sample_id": int(batch["sample_ids"][index]),
                      "structure_role": "generator_output" if generator_output else "random_prior_input"})
    return atoms


def export_batch(batch: dict, output_dir: Path, *, start: int = 0,
                 count: int | None = None, stride: int = 1,
                 file_format: str = "both", individual: bool = False,
                 generator_output: bool = False) -> dict:
    validate_batch(batch)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    if start < 0 or start >= len(batch["positions"]) or stride <= 0 or (count is not None and count <= 0):
        raise ValueError("Invalid start/count/stride.")
    if file_format not in {"extxyz", "traj", "both"}:
        raise ValueError("Format must be extxyz, traj, or both.")
    indices = list(range(start, len(batch["positions"]), stride))
    if count is not None:
        indices = indices[:count]
    atoms_list = [to_atoms(batch, i, generator_output=generator_output) for i in indices]
    output_dir.mkdir(parents=True, exist_ok=False)
    files = []
    for fmt in ("extxyz", "traj") if file_format == "both" else (file_format,):
        name = f"structures.{fmt}"
        write(str(output_dir / name), atoms_list, format=fmt)
        files.append(name)
    if individual:
        directory = output_dir / "individual"
        directory.mkdir()
        for i, atoms in zip(indices, atoms_list):
            write(str(directory / f"sample_{i:06d}.extxyz"), atoms, format="extxyz")
    report = {"structure_count": len(indices), "sample_indices": indices,
              "system_id": str(batch["system_id"].item()), "files": files,
              "individual_extxyz": individual, "generator_output": generator_output,
              "energy_force_labels": "none; these coordinates have not been evaluated"}
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT / "inputs.npz")
    parser.add_argument("--output-dir", type=Path, help="Default: structures/ beside the input archive.")
    parser.add_argument("--positions", type=Path, help="Optional generator Cartesian output, .npy (not noise/latent).")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, help="Maximum number of structures to export; default all.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--format", choices=["extxyz", "traj", "both"], default="both")
    parser.add_argument("--individual", action="store_true", help="Also write one .extxyz per sample.")
    args = parser.parse_args()
    batch = load_batch(args.input)
    if args.positions is not None:
        batch = replace_positions(batch, np.load(args.positions, allow_pickle=False))
    output_dir = args.output_dir or args.input.resolve().parent / "structures"
    report = export_batch(batch, output_dir, start=args.start, count=args.count,
        stride=args.stride, file_format=args.format, individual=args.individual,
        generator_output=args.positions is not None)
    print(json.dumps({"output_dir": str(output_dir.resolve()),
                      "structure_count": report["structure_count"], "files": report["files"]}, indent=2))


if __name__ == "__main__":
    main()
