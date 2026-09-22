"""Clean slab/bulk + adsorbate -> R0 -> one-pass AdsDrift -> PDB (no MLFF)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

# Works both as a file and via python -m AdsDrift.generate.generate_pdb.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import write

from AdsDrift.generate.adsorbate_library import DEFAULT_LIBRARY, entry_from_file, get_adsorbate, validate_entry
from AdsDrift.generate.structure_input import combine_reference, prepare_slab
from AdsDrift.model.initialize.generate_random_inputs import generate_batch, coverage_report
from AdsDrift.model.condition import collate_factorized_conditions, load_factorized_condition


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_generator(path, device):
    import torch
    from AdsDrift.model.generator import build_generator
    # Existing training checkpoints also contain optimizer/RNG state. Only load
    # user-owned/trusted checkpoints; the CLI never downloads or invents weights.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not {"generator", "generator_config"} <= checkpoint.keys():
        raise ValueError("Expected a test_18 training checkpoint with generator and generator_config")
    if int(checkpoint.get("format_version", -1)) != 3:
        raise ValueError("Expected checkpoint format_version=3 from test_18")
    config = checkpoint["generator_config"]
    if tuple(config.get("pbc_axes", (True, True, False))) != (True, True, False):
        raise ValueError("This surface CLI requires a checkpoint trained with xy periodic / z nonperiodic graphs")
    model = build_generator(config)
    model.load_state_dict(checkpoint["generator"], strict=True)
    model.to(device).eval()
    metadata = dict(epoch=checkpoint.get("epoch"), step=checkpoint.get("step"), generator_config=config,
                    training_data_roots=checkpoint.get("config", {}).get("data", {}).get("roots", []))
    return model, metadata


def infer(model, batch, condition, device, microbatch_size):
    import torch
    t = lambda value, dtype: torch.as_tensor(value, device=device, dtype=dtype)
    numbers = t(batch["atomic_numbers"], torch.long)[None]
    roles = t(batch["tags"] + 1, torch.long)[None]
    cell = t(batch["cell"], torch.float32)[None]
    mask = torch.ones_like(numbers, dtype=torch.bool)
    condition = {name: value.to(device) for name, value in condition.items()}
    if int(numbers.max()) > model.config.max_atomic_number:
        raise ValueError("Input element is outside the checkpoint element table")
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(batch["positions"]), microbatch_size):
            initial = t(batch["positions"][start:start+microbatch_size], torch.float32)[None]
            predicted = model(
                numbers, roles, mask, cell, initial, condition
            )["positions"][0]
            if not torch.isfinite(predicted).all():
                raise FloatingPointError("Generator returned NaN/Inf; no successful output will be marked")
            fixed = roles[0] == 1
            if not torch.equal(predicted[:, fixed], initial[0, :, fixed]):
                raise ValueError("Generator changed fixed slab atoms")
            outputs.append(predicted.cpu().numpy())
    return np.concatenate(outputs)


def make_frames(batch, positions):
    tags = batch["tags"]
    frames = []
    for i, coordinates in enumerate(positions):
        atoms = Atoms(numbers=batch["atomic_numbers"], positions=coordinates,
                      cell=batch["cell"], pbc=batch["pbc"], tags=tags)
        atoms.set_constraint(FixAtoms(indices=np.flatnonzero(tags == 0)))
        atoms.new_array("residuenames", np.array(["FIX" if t == 0 else "SLB" if t == 1 else "ADS" for t in tags]))
        atoms.new_array("residuenumbers", tags.astype(int)+1)
        atoms.info.update(candidate_index=i, structure_status="generator_prediction_no_relaxation_no_energy_ranking")
        frames.append(atoms)
    return frames


def run(args):
    import torch
    if args.num_samples < 1 or args.microbatch_size < 1 or args.seed < 0 or args.threads < 1:
        raise ValueError("Sample/microbatch/thread counts must be positive and seed nonnegative")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a nonempty output directory: {output}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.adsorbate_file is not None:
        ads_metadata = entry_from_file(args.adsorbate_file, args.adsorbate_file.stem)
        molecule = validate_entry(ads_metadata)
    else:
        molecule, ads_metadata = get_adsorbate(args.adsorbate, args.library)
    vacuum = None if args.keep_cell else (15.0 if args.vacuum is None else args.vacuum)
    slab, slab_metadata = prepare_slab(args.structure, index=args.index, input_kind=args.input_kind,
        cell_override=args.cell, miller=args.miller, layers=args.layers, repeat=args.repeat,
        vacuum=vacuum, fixed_layers=args.fixed_layers, fixed_indices=args.fixed_indices,
        use_input_tags=args.use_input_tags, layer_tolerance=args.layer_tolerance)
    reference = combine_reference(slab, molecule, args.gap_max)
    if len(reference) > 99999:
        raise ValueError("PDB supports at most 99999 atoms per model in this exporter")
    print(f"Slab: {len(slab)} atoms | adsorbate: {len(molecule)} atoms | vacuum/side: {vacuum}", flush=True)
    batch, prior_metadata = generate_batch(reference, count=args.num_samples, seed=args.seed,
        gap_min=args.gap_min, gap_max=args.gap_max, max_attempts_per_sample=args.max_attempts_per_sample,
        system_id=args.structure.stem, sampling="stratified", min_lateral_separation=args.min_lateral_separation)
    template_distances = molecule.get_all_distances()
    for coordinates in batch["positions"][:, batch["adsorbate_mask"]]:
        distances = np.linalg.norm(coordinates[:, None]-coordinates[None, :], axis=-1)
        if not np.allclose(distances, template_distances, atol=1e-5, rtol=0):
            raise ValueError("Cell is too small for the molecular template: PBC unwrapping changed its internal geometry; increase --repeat")
    coverage = coverage_report(batch)
    adsorbate_numbers = batch["atomic_numbers"][batch["adsorbate_mask"]]
    condition_record = load_factorized_condition(
        args.condition_file,
        expected_adsorbate_atomic_numbers=adsorbate_numbers,
    )
    condition = collate_factorized_conditions([condition_record])
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu or a GPU environment")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model, checkpoint_metadata = load_generator(args.checkpoint, device)
    started = time.monotonic()
    positions = infer(model, batch, condition, device, args.microbatch_size)
    elapsed = time.monotonic()-started
    # Keep exported fixed atoms exactly equal to the float64 input; the model
    # check above already established immobility at its fp32 precision.
    positions = positions.astype(np.float64)
    positions[:, batch["fixed_mask"]] = batch["reference_positions"][batch["fixed_mask"]]
    frames = make_frames(batch, positions)
    _, pdb_rotation = reference.cell.standard_form()
    for atoms in frames:
        pdb_positions = atoms.positions @ pdb_rotation.T
        if np.min(pdb_positions) < -999.999 or np.max(pdb_positions) > 9999.999:
            raise ValueError("Coordinates exceed PDB's 8.3 field; recenter the input cell")
    output.mkdir(parents=True, exist_ok=True)
    (output / "pdb").mkdir()
    write(output / "reference.extxyz", reference)
    write(output / "generated.extxyz", frames)
    write(output / "generated.pdb", frames, format="proteindatabank")
    for i, atoms in enumerate(frames):
        write(output / "pdb" / f"candidate_{i:04d}.pdb", atoms, format="proteindatabank")
    np.savez_compressed(output / "initial_structures.npz", **batch)
    np.savez_compressed(output / "generated_structures.npz", positions=positions,
        atomic_numbers=batch["atomic_numbers"], tags=batch["tags"], roles=batch["tags"]+1,
        cell=batch["cell"], pbc=batch["pbc"], fixed_mask=batch["fixed_mask"], sample_ids=batch["sample_ids"])
    save_json(output / "initial_coverage.json", coverage)
    manifest = dict(schema_version=2, complete=True, source_structure=str(args.structure.resolve()),
        source_sha256=sha256(args.structure), checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
        factorized_condition=str(args.condition_file.resolve()),
        factorized_condition_sha256=sha256(args.condition_file),
        checkpoint_metadata=checkpoint_metadata, slab=slab_metadata, adsorbate=ads_metadata,
        initial_sampling=prior_metadata, count=len(frames), seed=args.seed, device=str(device),
        microbatch_size=args.microbatch_size, generator_seconds=elapsed,
        workflow="random_R0 -> one_generator_pass_per_candidate -> export_all_candidates",
        mlff_used=False, energy_ranking=False, numerical_relaxation=False,
        warning="Predicted low-energy candidates only: neither physical minima nor generalization to unseen systems are guaranteed; frequencies are not thermodynamic occupancies.",
        pdb=dict(positions_transform="pdb_positions = npz_positions @ rotation.T", rotation=pdb_rotation.tolist(),
                 coordinate_precision_A=.001, residues={"FIX": "fixed slab", "SLB": "movable slab", "ADS": "adsorbate"},
                 limitations="PDB loses exact cell precision, PBC axes, role tags and constraints; use NPZ/extxyz for numerical reuse."))
    save_json(output / "manifest.json", manifest)
    (output / "_SUCCESS").write_text("generator export complete; no MLFF/DFT validation\n")
    print(json.dumps(dict(output=str(output), count=len(frames), generator_seconds=elapsed, mlff_used=False)), flush=True)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted test_18 training .pt checkpoint")
    parser.add_argument(
        "--condition-file",
        type=Path,
        required=True,
        help="Explicit condition_factors.json; test_18 does not infer primitive construction variables",
    )
    parser.add_argument("--structure", "--slab", type=Path, required=True, help="Clean slab or conventional bulk cell, ASE-readable")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--input-kind", choices=("slab", "bulk"), default="slab")
    parser.add_argument("--cell", type=float, nargs="+", help="3 lengths / 6 cell parameters / 9 row-major matrix entries; never rescale atoms")
    parser.add_argument("--miller", type=int, nargs=3)
    parser.add_argument("--layers", type=int, help="ASE surface layer count for bulk cutting")
    parser.add_argument("--repeat", type=int, nargs=2, default=[1, 1], metavar=("NX", "NY"))
    vacuum = parser.add_mutually_exclusive_group()
    vacuum.add_argument("--vacuum", type=float, help="Vacuum on EACH side along surface normal, default 15 A")
    vacuum.add_argument("--keep-cell", action="store_true", help="Preserve an existing slab cell and coordinates")
    fixed = parser.add_mutually_exclusive_group()
    fixed.add_argument("--fixed-layers", type=int, help="Bottom atomic planes; default preserve FixAtoms else 1; 0 means all movable")
    fixed.add_argument("--fixed-indices", type=int, nargs="+")
    fixed.add_argument("--use-input-tags", action="store_true")
    parser.add_argument("--layer-tolerance", type=float, default=.3)
    adsorbate = parser.add_mutually_exclusive_group(required=True)
    adsorbate.add_argument("--adsorbate", help="Template ID or original name, e.g. 74 or '*NH'")
    adsorbate.add_argument("--adsorbate-file", type=Path, help="Isolated molecular XYZ/etc. instead of a library entry")
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--gap-min", type=float, default=1.2)
    parser.add_argument("--gap-max", type=float, default=3.0)
    parser.add_argument("--max-attempts-per-sample", type=int, default=100)
    parser.add_argument("--min-lateral-separation", type=float, default=.1)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
