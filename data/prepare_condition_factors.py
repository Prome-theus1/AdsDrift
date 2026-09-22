"""Create the explicit test_18 condition_factors.json beside one training system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read

from AdsDrift.model.condition import load_factorized_condition


def prepare(args: argparse.Namespace) -> Path:
    system_directory = args.system_directory.expanduser().resolve()
    input_path = system_directory / "random_input" / "inputs.npz"
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    with np.load(input_path, allow_pickle=False) as values:
        system_id = str(np.asarray(values["system_id"]).item())
        all_numbers = np.asarray(values["atomic_numbers"], dtype=np.int64)
        tags = np.asarray(values["tags"], dtype=np.int64)
    adsorbate_numbers = all_numbers[tags == 2]

    primitive = read(args.primitive_structure.expanduser().resolve(), index=args.index)
    conventional = read(
        args.conventional_structure.expanduser().resolve(), index=args.index
    )
    if not primitive.pbc.all():
        raise ValueError("The primitive crystal input must be periodic in all three axes")
    if not conventional.pbc.all():
        raise ValueError("The conventional crystal input must be periodic in all three axes")
    primitive.wrap()
    conventional.wrap()
    fractions = np.mod(primitive.get_scaled_positions(wrap=True), 1.0)
    matrix = np.asarray(args.supercell_matrix, dtype=np.int64).reshape(3, 3)
    record = {
        "schema_version": 1,
        "system_id": system_id,
        "primitive": {
            "atomic_numbers": primitive.numbers.tolist(),
            "fractional_positions": fractions.tolist(),
            "cell": np.asarray(primitive.cell).tolist(),
        },
        "surface": {
            "miller_index": list(args.miller),
            "miller_basis": "provided conventional standard bulk",
            "conventional_cell": np.asarray(conventional.cell).tolist(),
            "termination_shift": args.termination_shift,
            "top": args.top,
        },
        "construction": {
            "supercell_matrix": matrix.tolist(),
            "slab_layers": args.slab_layers,
            "vacuum_A": args.vacuum,
            "strain_voigt": list(args.strain_voigt),
            "adsorbates_per_cell": args.adsorbates_per_cell,
        },
        "adsorbate": {
            "atomic_numbers": adsorbate_numbers.tolist(),
            "identity_source": "ordered tag-2 atoms in random_input/inputs.npz",
        },
        "provenance": {
            "primitive_structure": str(args.primitive_structure.expanduser().resolve()),
            "conventional_structure": str(
                args.conventional_structure.expanduser().resolve()
            ),
            "note": "Construction variables supplied explicitly; none inferred from the final slab.",
        },
    }
    output = system_directory / "condition_factors.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    # Re-open through the training validator before declaring success.
    load_factorized_condition(
        output,
        expected_system_id=system_id,
        expected_adsorbate_atomic_numbers=adsorbate_numbers,
    )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-directory", type=Path, required=True)
    parser.add_argument("--primitive-structure", type=Path, required=True)
    parser.add_argument("--conventional-structure", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--miller", type=int, nargs=3, required=True)
    parser.add_argument("--termination-shift", type=float, required=True)
    side = parser.add_mutually_exclusive_group(required=True)
    side.add_argument("--top", dest="top", action="store_true")
    side.add_argument("--bottom", dest="top", action="store_false")
    parser.add_argument(
        "--supercell-matrix",
        type=int,
        nargs=9,
        required=True,
        metavar=("M11", "M12", "M13", "M21", "M22", "M23", "M31", "M32", "M33"),
    )
    parser.add_argument("--slab-layers", type=int, required=True)
    parser.add_argument("--vacuum", type=float, required=True)
    parser.add_argument("--strain-voigt", type=float, nargs=6, default=[0.0] * 6)
    parser.add_argument("--adsorbates-per-cell", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(prepare(parse_args()))
