#!/usr/bin/env python3
"""Prepare OC20-Dense data: training index, source trajectories, or endpoints.

This consolidates the three historical data scripts. R0 generation and MACE
feature dumping are separate downstream stages, not performed by this entry point.
Run with --help or <subcommand> --help for arguments.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import time

INDEX_SCHEMA = "oc20_dense_drifting_training_v1"
TRAJECTORY_SCHEMA = "oc20_dense_filtered_trajectories_v1"


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def stable_seed(base_seed: int, system_id: str) -> int:
    payload = f"{base_seed}:{system_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def select_clusters(report: dict, cluster_key: str, window_ev: float) -> list[dict]:
    clusters = report.get("clusters", {}).get(cluster_key)
    if clusters is None:
        if int(report.get("accepted_count", 0)) == 0:
            return []
        raise ValueError(f"Missing {cluster_key} in audit for {report['system_id']}")
    return sorted(
        (item for item in clusters if float(item["delta_E_eV"]) <= window_ev + 1e-12),
        key=lambda item: (float(item["delta_E_eV"]), str(item["representative"])),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_one(task: tuple[Path, Path]) -> dict:
    source, destination = task
    source_stat = source.stat()
    if destination.exists():
        destination_stat = destination.stat()
        if (destination_stat.st_size != source_stat.st_size
                or sha256(destination) != sha256(source)):
            raise RuntimeError(
                f"Refusing to replace conflicting destination: {destination}"
            )
        return {"bytes": source_stat.st_size, "copied": 0, "reused": 1}

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    if temporary.stat().st_size != source_stat.st_size:
        raise RuntimeError(f"Incomplete copy: {source} -> {destination}")
    os.replace(temporary, destination)
    return {"bytes": source_stat.st_size, "copied": 1, "reused": 0}



def extract_positives(args: argparse.Namespace) -> None:
    import numpy as np
    from ase.io import read, write

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {args.output_dir}")

    report = json.loads(args.audit_json.read_text())
    system_id = report["system_id"]
    # This is the official, locally downloaded OC20-Dense tag mapping.
    with args.tag_mapping.open("rb") as stream:
        tags = np.asarray(pickle.load(stream)[system_id], dtype=int)
    clusters = select_clusters(report, args.cluster_key, args.window_ev)
    if not clusters:
        raise ValueError("No accepted low-energy endpoints in the requested window")

    structures = []
    manifest = []
    for rank, cluster in enumerate(clusters, start=1):
        config_id = cluster["representative"]
        source = args.system_dir / f"{system_id}_{config_id}.traj"
        atoms = read(source, index=-1)
        if len(atoms) != len(tags):
            raise ValueError(f"Official tag length does not match {source}")
        atoms.set_tags(tags)
        atoms.info.update({
            "system_key": system_id,
            "config_id": config_id,
            "positive_rank": rank,
            "adsorption_energy_eV": float(cluster["energy_adsorption_eV"]),
            "delta_E_eV": float(cluster["delta_E_eV"]),
            "cluster_multiplicity": int(cluster["multiplicity"]),
            "selection_window_eV": float(args.window_ev),
            "cluster_definition": args.cluster_key,
        })
        filename = f"{system_id}_rank{rank:02d}_{config_id}.extxyz"
        structures.append(atoms)
        manifest.append({
            "rank": rank,
            "system_id": system_id,
            "config_id": config_id,
            "file": filename,
            "adsorption_energy_eV": float(cluster["energy_adsorption_eV"]),
            "delta_E_eV": float(cluster["delta_E_eV"]),
            "cluster_multiplicity": int(cluster["multiplicity"]),
            "member_config_ids": cluster["member_config_ids"],
            "source_trajectory": str(source),
        })

    args.output_dir.mkdir(parents=True)
    for atoms, item in zip(structures, manifest):
        write(args.output_dir / item["file"], atoms, format="extxyz")
    window_label = f"{args.window_ev:.2f}".replace(".", "p")
    combined_name = f"{system_id}_positive_{window_label}eV.extxyz"
    write(args.output_dir / combined_name, structures, format="extxyz")
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "system_id": system_id,
        "energy_window_eV": args.window_ev,
        "energy_reference": "best accepted endpoint observed for this system",
        "cluster_key": args.cluster_key,
        "structure_count": len(manifest),
        "combined_file": combined_name,
        "structures": manifest,
    }, indent=2) + "\n")
    with (args.output_dir / "manifest.csv").open("w", newline="") as stream:
        columns = ["rank", "system_id", "config_id", "file", "adsorption_energy_eV",
                   "delta_E_eV", "cluster_multiplicity", "source_trajectory"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in manifest:
            writer.writerow({key: item[key] for key in columns})
    print(json.dumps({"output_dir": str(args.output_dir),
                      "structure_count": len(manifest),
                      "representatives": [item["config_id"] for item in manifest]}))


def prepare_index(args: argparse.Namespace) -> None:
    if args.window_ev < 0 or args.minimum_modes < 1 or args.random_count < 1:
        raise SystemExit("Invalid window, mode threshold, or random count")
    reports = sorted((args.audit_root / "systems").glob("*.json"))
    if not reports:
        raise FileNotFoundError(args.audit_root / "systems")
    if not args.trajectory_root.is_dir() or not args.tag_mapping.is_file():
        raise FileNotFoundError("Trajectory root or official tag mapping is missing")

    rows = []
    excluded = []
    single_mode_allowlist = set(args.single_mode_systems or [])
    unknown_exceptions = single_mode_allowlist - {path.stem for path in reports}
    if unknown_exceptions:
        raise ValueError(f"Single-mode exceptions absent from audit: {sorted(unknown_exceptions)}")
    total_modes = total_endpoints = 0
    for report_path in reports:
        report = json.loads(report_path.read_text())
        system_id = str(report["system_id"])
        selected = select_clusters(report, args.cluster_key, args.window_ev)
        minimum_modes = 1 if system_id in single_mode_allowlist else args.minimum_modes
        if len(selected) < minimum_modes:
            excluded.append({
                "system_id": system_id,
                "reason": "fewer low-energy modes than minimum",
                "mode_count": len(selected),
            })
            continue
        endpoint_count = sum(len(item["member_config_ids"]) for item in selected)
        reference_config = str(selected[0]["representative"])
        reference = args.trajectory_root / system_id / f"{system_id}_{reference_config}.traj"
        if not reference.is_file():
            raise FileNotFoundError(reference)
        row = {
            "system_id": system_id,
            "reference_config": reference_config,
            "seed": stable_seed(args.base_seed, system_id),
            "positive_mode_count": len(selected),
            "positive_endpoint_count": endpoint_count,
        }
        rows.append(row)
        total_modes += len(selected)
        total_endpoints += endpoint_count

    rows.sort(key=lambda item: item["system_id"])
    if args.expected_systems is not None and len(rows) != args.expected_systems:
        raise RuntimeError(f"Expected {args.expected_systems} systems, found {len(rows)}")
    if not rows:
        raise ValueError("No systems satisfy the requested mode threshold")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty index directory: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "systems").mkdir()
    (args.output_root / "logs").mkdir()

    atomic_text(
        args.output_root / "jobs.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )
    header = "system_id\treference_config\tseed\tpositive_mode_count\tpositive_endpoint_count\n"
    atomic_text(
        args.output_root / "jobs.tsv",
        header + "".join(
            f"{row['system_id']}\t{row['reference_config']}\t{row['seed']}\t"
            f"{row['positive_mode_count']}\t{row['positive_endpoint_count']}\n"
            for row in rows
        ),
    )
    atomic_text(
        args.output_root / "excluded_systems.json",
        json.dumps(excluded, ensure_ascii=False, indent=2) + "\n",
    )
    manifest = {
        "schema": INDEX_SCHEMA,
        "complete": False,
        "selection": {
            "energy_window_eV": args.window_ev,
            "cluster_key": args.cluster_key,
            "minimum_distinct_modes": args.minimum_modes,
            "single_mode_system_allowlist": sorted(single_mode_allowlist),
            "positive_endpoint_policy": "all accepted endpoints retained; training samples modes uniformly",
            "random_R0_per_system": args.random_count,
            "random_prior": "stratified lateral coverage, stratified height, uniform SO(3)",
        },
        "counts": {
            "audited_systems": len(reports),
            "included_systems": len(rows),
            "excluded_systems": len(excluded),
            "positive_modes": total_modes,
            "positive_endpoints": total_endpoints,
            "random_R0": len(rows) * args.random_count,
        },
        "source": {
            "audit_root": str(args.audit_root.resolve()),
            "trajectory_root": str(args.trajectory_root.resolve()),
            "tag_mapping": str(args.tag_mapping.resolve()),
        },
        "base_seed": args.base_seed,
        "files": {
            "jobs": "jobs.jsonl",
            "systems": "systems/<system_id>",
            "validation": "validation.json",
            "temperature_calibration": "temperature_calibration.json",
        },
    }
    atomic_text(args.output_root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    atomic_text(args.output_root / "_INCOMPLETE", "dataset preparation in progress\n")
    readme = f"""# OC20-Dense conditional Drifting training dataset

This index selects {len(rows)} fixed surface-adsorbate conditions.
After the structure and feature preparation stages, each condition will have {args.random_count} random R0 structures, every accepted DFT
endpoint within {args.window_ev:.2f} eV, endpoint mode labels, and frozen MACE-MH-1
adsorbate/movable-surface scalar and l=1 features. Conditions with fewer than
{args.minimum_modes} symmetry-distinct low-energy modes are excluded, except for
the explicit single-mode system allowlist in manifest.json.

Index creation alone does not generate R0, endpoint archives, or MACE features.
The manifest remains incomplete until the downstream validation succeeds.

The endpoint archive is not weighted by raw relaxation multiplicity during
training: one endpoint is sampled from every mode, using the same endpoint index
for all four feature branches.
"""
    atomic_text(args.output_root / "README.md", readme)
    print(json.dumps(manifest["counts"], indent=2))


def export_trajectories(args: argparse.Namespace) -> int:
    if args.window_ev < 0 or args.workers < 1:
        raise SystemExit("window-ev must be nonnegative and workers must be positive")

    system_report_dir = args.audit_root / "systems"
    audit_summary = args.audit_root / "summary.json"
    if not system_report_dir.is_dir() or not audit_summary.is_file():
        raise SystemExit(f"Incomplete audit input: {args.audit_root}")
    if not args.trajectory_root.is_dir():
        raise SystemExit(f"Missing trajectory root: {args.trajectory_root}")

    if not args.mapping_root.is_dir():
        raise FileNotFoundError(args.mapping_root)
    mapping_files = sorted(args.mapping_root.glob("*.pkl"))
    if not mapping_files:
        raise FileNotFoundError(f"No official mapping .pkl files in {args.mapping_root}")

    request = {
        "trajectory_root": str(args.trajectory_root.resolve()),
        "audit_root": str(args.audit_root.resolve()),
        "audit_summary_sha256": sha256(audit_summary),
        "mapping_root": str(args.mapping_root.resolve()),
        "window_ev": args.window_ev,
        "cluster_key": args.cluster_key,
    }
    request_file = args.output_dir / "preparation_request.json"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not request_file.is_file() or json.loads(request_file.read_text()) != request:
            raise FileExistsError(f"Output belongs to a different or legacy preparation: {args.output_dir}")
    success = args.output_dir / "_SUCCESS"
    if success.exists():
        print(f"Dataset is already complete: {args.output_dir}", flush=True)
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_text(request_file, json.dumps(request, indent=2) + "\n")
    atomic_text(args.output_dir / "_INCOMPLETE", "copy in progress\n")

    selection_rows: list[dict] = []
    system_rows: list[dict] = []
    copy_tasks: list[tuple[Path, Path]] = []
    excluded_systems: list[dict] = []
    sample_index = 0

    reports = sorted(system_report_dir.glob("*.json"), key=lambda p: p.stem)
    if not reports:
        raise SystemExit(f"No system reports under {system_report_dir}")

    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        system_id = str(report["system_id"])
        selected = select_clusters(report, args.cluster_key, args.window_ev)
        if not selected:
            excluded_systems.append({
                "system_id": system_id,
                "reason": "no accepted representative within energy window",
                "accepted_count": int(report.get("accepted_count", 0)),
            })
            continue

        source_system = args.trajectory_root / system_id
        destination_system = args.output_dir / "trajs" / system_id
        surface_name = f"{system_id}_surface.traj"
        source_surface = source_system / surface_name
        if not source_surface.is_file():
            raise FileNotFoundError(source_surface)
        copy_tasks.append((source_surface, destination_system / surface_name))

        selected = sorted(
            selected,
            key=lambda row: (float(row["energy_adsorption_eV"]), row["representative"]),
        )
        system_sample_indices = []
        for positive_rank, cluster in enumerate(selected):
            config_id = str(cluster["representative"])
            filename = f"{system_id}_{config_id}.traj"
            source = source_system / filename
            destination = destination_system / filename
            if not source.is_file():
                raise FileNotFoundError(source)
            copy_tasks.append((source, destination))
            relative = destination.relative_to(args.output_dir).as_posix()
            selection_rows.append({
                "sample_index": sample_index,
                "system_id": system_id,
                "config_id": config_id,
                "positive_rank": positive_rank,
                "trajectory": relative,
                "adsorption_energy_eV": float(cluster["energy_adsorption_eV"]),
                "delta_E_eV": float(cluster["delta_E_eV"]),
                "cluster_multiplicity": int(cluster["multiplicity"]),
                "member_config_ids": list(cluster["member_config_ids"]),
                "energy_spread_eV": float(cluster["energy_spread_eV"]),
            })
            system_sample_indices.append(sample_index)
            sample_index += 1

        metadata = report.get("metadata", {})
        system_rows.append({
            "system_id": system_id,
            "surface_trajectory": f"trajs/{system_id}/{surface_name}",
            "number_of_positive_trajectories": len(selected),
            "sample_indices": system_sample_indices,
            "surface_formula": report.get("surface_formula"),
            "slab_atoms": report.get("slab_atoms"),
            "adsorbate_atoms": report.get("adsorbate_atoms"),
            "metadata": metadata,
        })

    if not selection_rows:
        raise ValueError("No accepted low-energy trajectories in the requested window")
    expected_samples = args.expected_samples
    if expected_samples is not None and len(selection_rows) != expected_samples:
        raise RuntimeError(
            f"Selection mismatch: expected {expected_samples}, found {len(selection_rows)}"
        )

    selection_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selection_rows)
    systems_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in system_rows)
    atomic_text(args.output_dir / "selection.jsonl", selection_text)
    atomic_text(args.output_dir / "systems.jsonl", systems_text)
    atomic_text(
        args.output_dir / "excluded_systems.json",
        json.dumps(excluded_systems, ensure_ascii=False, indent=2) + "\n",
    )

    metadata_dir = args.output_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    for source in mapping_files:
        copy_tasks.append((source, metadata_dir / source.name))

    print(
        f"Prepared {len(selection_rows)} positive trajectories from "
        f"{len(system_rows)} systems; {len(copy_tasks)} total files to copy",
        flush=True,
    )
    started = time.time()
    totals = {"bytes": 0, "copied": 0, "reused": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for completed, result in enumerate(pool.map(copy_one, copy_tasks), start=1):
            for key in totals:
                totals[key] += result[key]
            if completed % 500 == 0 or completed == len(copy_tasks):
                print(
                    f"files={completed}/{len(copy_tasks)} "
                    f"copied={totals['copied']} reused={totals['reused']} "
                    f"GiB={totals['bytes'] / 2**30:.2f}",
                    flush=True,
                )

    copied_trajectories = list((args.output_dir / "trajs").glob("*/*.traj"))
    expected_trajectories = len(selection_rows) + len(system_rows)
    if len(copied_trajectories) != expected_trajectories:
        raise RuntimeError(
            f"Final file-count mismatch: expected {expected_trajectories}, "
            f"found {len(copied_trajectories)}"
        )

    max_delta = max(row["delta_E_eV"] for row in selection_rows)
    manifest = {
        "schema": TRAJECTORY_SCHEMA,
        "complete": True,
        "selection": {
            "energy_window_eV": args.window_ev,
            "cluster_key": args.cluster_key,
            "representative_rule": "lowest-energy original DFT trajectory per cluster",
            "energy_reference": "accepted observed minimum within each system",
        },
        "counts": {
            "audit_systems": len(reports),
            "included_systems": len(system_rows),
            "excluded_systems": len(excluded_systems),
            "positive_trajectories": len(selection_rows),
            "surface_trajectories": len(system_rows),
            "trajectory_files": expected_trajectories,
        },
        "observed_max_delta_E_eV": max_delta,
        "copied_bytes_including_metadata": totals["bytes"],
        "source": {
            "trajectory_root": str(args.trajectory_root),
            "audit_root": str(args.audit_root),
            "audit_summary_sha256": sha256(audit_summary),
            "mapping_root": str(args.mapping_root),
        },
        "files": {
            "selection": "selection.jsonl",
            "systems": "systems.jsonl",
            "excluded_systems": "excluded_systems.json",
            "trajectories": "trajs/<system_id>/*.traj",
            "official_metadata": "metadata/*.pkl",
        },
        "copy": {
            **totals,
            "workers": args.workers,
            "elapsed_seconds": time.time() - started,
        },
    }
    atomic_text(
        args.output_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    readme = f"""# OC20-Dense {args.window_ev:.2f} eV low-energy trajectories

This directory contains unmodified original ASE `.traj` files selected from
OC20-Dense. It is a file-level training corpus, not a model-specific serialized
dataset.

- Included systems: {len(system_rows)}
- Selected adsorption trajectories: {len(selection_rows)}
- Clean-surface trajectories: {len(system_rows)}
- Excluded systems: {len(excluded_systems)}
- Energy rule: `delta_E <= {args.window_ev:.2f} eV` relative to the accepted
  observed minimum of the same surface-adsorbate system
- Deduplication: `{args.cluster_key}`
- Representative: lowest-energy original DFT trajectory in each cluster

Every selected adsorption file retains its complete relaxation history. No
frame, coordinate, cell, energy, force, or calculator result was rewritten.
The matching clean-surface trajectory is stored in the same system directory.

Use `selection.jsonl` for the selected trajectory list and energies,
`systems.jsonl` for grouping, and `manifest.json` for provenance and counts.
The files in `metadata/` are unchanged copies of the official OC20-Dense
mapping/tag/target/reference-energy tables and may contain entries outside this
filtered subset.
"""
    atomic_text(args.output_dir / "README.md", readme)
    incomplete = args.output_dir / "_INCOMPLETE"
    if incomplete.exists():
        incomplete.unlink()
    atomic_text(success, "complete\n")
    print(json.dumps(manifest["counts"], indent=2), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--window-ev", "--window-eV", "--window", dest="window_ev",
        type=float, default=0.5, help="Energy window in eV (default: 0.5)",
    )
    shared.add_argument("--cluster-key", default="symmetry_rmsd_0.1A")

    index = subcommands.add_parser(
        "index", parents=[shared], help="Build the deterministic multi-system training index",
    )
    index.add_argument("--audit-root", type=Path, required=True)
    index.add_argument("--trajectory-root", type=Path, required=True)
    index.add_argument("--tag-mapping", type=Path, required=True)
    index.add_argument("--output-root", type=Path, required=True)
    index.add_argument("--minimum-modes", type=int, default=2)
    index.add_argument(
        "--single-mode-systems", nargs="*", default=[],
        help="Explicit system IDs allowed to have one low-energy mode; other conditions keep the normal threshold",
    )
    index.add_argument("--random-count", type=int, default=100)
    index.add_argument("--base-seed", type=int, default=20260906)
    index.add_argument("--expected-systems", type=int, help="Optional full-corpus count assertion")
    index.set_defaults(handler=prepare_index)

    trajectories = subcommands.add_parser(
        "trajectories", parents=[shared],
        help="Copy one original trajectory per mode plus clean surfaces and metadata",
    )
    trajectories.add_argument("--audit-root", type=Path, required=True)
    trajectories.add_argument("--trajectory-root", type=Path, required=True)
    trajectories.add_argument("--mapping-root", type=Path, required=True)
    trajectories.add_argument("--output-dir", type=Path, required=True)
    trajectories.add_argument("--workers", type=int, default=8)
    trajectories.add_argument("--expected-samples", type=int, help="Optional representative-count assertion")
    trajectories.set_defaults(handler=export_trajectories)

    positives = subcommands.add_parser(
        "positives", parents=[shared], help="Export one final-frame representative per mode for one system",
    )
    positives.add_argument("--system-dir", type=Path, required=True)
    positives.add_argument("--audit-json", type=Path, required=True)
    positives.add_argument("--tag-mapping", type=Path, required=True)
    positives.add_argument("--output-dir", type=Path, required=True)
    positives.set_defaults(handler=extract_positives)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.window_ev) or args.window_ev < 0:
        parser.error("The energy window must be finite and nonnegative")
    for name in ("minimum_modes", "random_count", "workers", "expected_systems", "expected_samples"):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args.handler(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
