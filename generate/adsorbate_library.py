"""Portable, pickle-free adsorbate templates (Cartesian coordinates in Angstrom)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

DEFAULT_LIBRARY = Path(__file__).with_name("adsorbates.json")


def validate_entry(entry: dict) -> Atoms:
    numbers = np.asarray(entry["atomic_numbers"])
    positions = np.asarray(entry["positions_A"], dtype=float)
    if (numbers.ndim != 1 or len(numbers) < 1 or not np.isin(numbers, np.arange(1, 119)).all()
            or positions.shape != (len(numbers), 3) or not np.isfinite(positions).all()):
        raise ValueError("Invalid adsorbate atomic numbers or Cartesian coordinates")
    binding = entry.get("binding_indices", [])
    if any(not isinstance(i, int) or i < 0 or i >= len(numbers) for i in binding):
        raise ValueError("Binding indices must be zero-based atom indices")
    atoms = Atoms(numbers=numbers.astype(int), positions=positions, pbc=False)
    if len(atoms) > 1:
        distances = atoms.get_all_distances()
        if np.min(distances[np.triu_indices(len(atoms), 1)]) < 1e-5:
            raise ValueError("Adsorbate template contains coincident atoms")
    return atoms


def read_library(path: Path = DEFAULT_LIBRARY) -> dict:
    library = json.loads(Path(path).read_text())
    if library.get("schema_version") != 1 or library.get("coordinate_units") != "angstrom":
        raise ValueError("Unsupported library schema or units")
    for entry in library["adsorbates"].values():
        validate_entry(entry)
    return library


def get_adsorbate(identifier: str, path: Path = DEFAULT_LIBRARY) -> tuple[Atoms, dict]:
    library = read_library(path)
    entries = library["adsorbates"]
    matches = [identifier] if identifier in entries else [key for key, entry in entries.items() if entry["name"] == identifier]
    if not matches:
        # Convenience only when unambiguous; never identify templates by formula.
        matches = [key for key, entry in entries.items() if entry["name"].lstrip("*") == identifier]
    if len(matches) != 1:
        raise ValueError(f"Adsorbate {identifier!r} is absent or ambiguous; use a library ID or exact name")
    key = matches[0]
    entry = entries[key]
    return validate_entry(entry), dict(library=str(Path(path).resolve()), library_id=key, **entry)


def entry_from_file(path: Path, name: str, binding_indices=()) -> dict:
    atoms = read(path, index=0)
    if atoms.pbc.any():
        raise ValueError("An adsorbate template must be an isolated, contiguous molecule (no PBC); use molecular XYZ")
    positions = atoms.positions - atoms.positions.mean(axis=0)
    entry = dict(name=name, formula=atoms.get_chemical_formula(), atomic_numbers=atoms.numbers.tolist(),
                 positions_A=positions.tolist(), binding_indices=list(binding_indices),
                 source=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    validate_entry(entry)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List template IDs, original names and formulas")
    add = commands.add_parser("add", help="Add an isolated molecule from XYZ/extxyz/etc.; never overwrite an existing ID")
    add.add_argument("--id", required=True)
    add.add_argument("--name", required=True)
    add.add_argument("--structure", type=Path, required=True)
    add.add_argument("--binding-indices", type=int, nargs="*", default=[])
    args = parser.parse_args()
    if args.command == "list":
        for key, entry in read_library(args.library)["adsorbates"].items():
            print(f"{key:>8}  {entry['name']:<18} {entry['formula']}")
        return
    library = read_library(args.library) if args.library.exists() else dict(schema_version=1, coordinate_units="angstrom", adsorbates={})
    if args.id in library["adsorbates"]:
        raise ValueError(f"Refusing to overwrite existing adsorbate ID {args.id!r}")
    library["adsorbates"][args.id] = entry_from_file(args.structure, args.name, args.binding_indices)
    args.library.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.library.with_suffix(args.library.suffix + ".tmp")
    with temporary.open("x") as handle:
        handle.write(json.dumps(library, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.library)
    print(f"Added {args.id}: {args.name} to {args.library}")


if __name__ == "__main__":
    main()
