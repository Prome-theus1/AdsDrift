"""One-pass inference from a complete 100-structure R0 condition bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from AdsDrift.model.generator import build_generator
from AdsDrift.model.condition import collate_factorized_conditions
from AdsDrift.model.utils.data_loader import load_system_bank, move_batch


def sample(
    checkpoint_path: str | Path,
    system_directory: str | Path,
    output_directory: str | Path,
    device_name: str = "auto",
) -> Path:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    system = load_system_bank(system_directory)
    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    generator = build_generator(checkpoint["generator_config"])
    generator.load_state_dict(checkpoint["generator"])
    generator.to(device).eval()

    numbers = torch.from_numpy(system.atomic_numbers)[None].to(device)
    roles = torch.from_numpy(system.roles)[None].to(device)
    atom_mask = torch.ones_like(numbers, dtype=torch.bool)
    cell = torch.from_numpy(system.cell)[None].to(device)
    r0 = torch.from_numpy(system.r0_positions)[None].to(device)
    condition = move_batch(
        {"condition": collate_factorized_conditions([system.condition])}, device
    )["condition"]
    with torch.inference_mode():
        generated = generator(
            numbers, roles, atom_mask, cell, r0, condition
        )["positions"][0]
    positions = generated.cpu().numpy()

    np.savez_compressed(
        output_directory / "generated_structures.npz",
        schema_version=np.asarray(2, dtype=np.int64),
        system_id=np.asarray(system.system_id),
        positions=positions,
        r0_positions=system.r0_positions,
        atomic_numbers=system.atomic_numbers,
        roles=system.roles,
        cell=system.cell,
    )
    from ase import Atoms
    from ase.constraints import FixAtoms
    from ase.io import write

    tags = np.empty_like(system.roles)
    tags[system.roles == 1] = 0
    tags[system.roles == 2] = 1
    tags[system.roles == 3] = 2
    frames = []
    for index, coordinates in enumerate(positions):
        atoms = Atoms(
            numbers=system.atomic_numbers,
            positions=coordinates,
            cell=system.cell,
            pbc=system.pbc,
            tags=tags,
        )
        atoms.set_constraint(FixAtoms(indices=np.flatnonzero(tags == 0)))
        atoms.info.update(system_id=system.system_id, generated_index=index)
        frames.append(atoms)
    write(output_directory / "generated_structures.extxyz", frames)
    (output_directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "system_id": system.system_id,
                "count": len(positions),
                "checkpoint": str(checkpoint_path),
                "inference": "one generator forward pass; no relaxation",
            },
            indent=2,
        )
        + "\n"
    )
    return output_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--system-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(sample(args.checkpoint, args.system_directory, args.output_directory, args.device))


if __name__ == "__main__":
    main()
