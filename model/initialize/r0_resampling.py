"""Deterministic per-epoch R0 resampling from each system's saved provenance."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from AdsDrift.model.generator import ROLE_ADSORBATE, ROLE_FIXED, ROLE_SURFACE
from AdsDrift.model.utils.data_loader import SystemFeatureBank
from AdsDrift.model.initialize.generate_random_inputs import (
    generate_batch,
    load_reference,
    load_tags_pickle,
)


@dataclass(frozen=True)
class OnlineR0Config:
    enabled: bool = False
    base_seed: int = 20260907
    max_attempts_per_sample: int = 100
    max_bank_retries: int = 4

    @classmethod
    def from_dict(cls, values: dict | None) -> "OnlineR0Config":
        values = dict(values or {})
        known = set(cls.__dataclass_fields__)
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown online R0 settings: {sorted(unknown)}")
        config = cls(**values)
        if (
            config.base_seed < 0
            or config.max_attempts_per_sample < 1
            or config.max_bank_retries < 1
        ):
            raise ValueError("Online R0 seeds must be nonnegative and attempts positive")
        return config


def epoch_seed(base_seed: int, system_id: str, epoch: int) -> int:
    """Return a deterministic, practically collision-free seed for one epoch.

    A 64-bit system-specific base plus the epoch index makes seeds unique for
    consecutive epochs and independent of dataloader/RNG resume state.
    """
    if base_seed < 0 or epoch < 0:
        raise ValueError("Seeds and epochs must be nonnegative")
    digest = hashlib.sha256(f"{base_seed}:{system_id}".encode()).digest()
    system_base = int.from_bytes(digest[:8], "little") | (1 << 63)
    return system_base + epoch


def _tags_from_roles(roles: np.ndarray) -> np.ndarray:
    tags = np.full(roles.shape, -1, dtype=np.int64)
    tags[roles == ROLE_FIXED] = 0
    tags[roles == ROLE_SURFACE] = 1
    tags[roles == ROLE_ADSORBATE] = 2
    if np.any(tags < 0):
        raise ValueError("System has unsupported atom roles")
    return tags


class OnlineR0Resampler:
    """Recreate the original valid prior with a new seed every epoch."""

    def __init__(self, systems: list[SystemFeatureBank], config: OnlineR0Config):
        self.config = config
        self.references = {}
        self.provenance = {}
        if not config.enabled:
            return
        for system in systems:
            input_path = system.directory / "random_input" / "inputs.npz"
            with np.load(input_path, allow_pickle=False) as values:
                if "metadata_json" not in values.files:
                    raise ValueError(f"Online R0 resampling needs metadata_json in {input_path}")
                metadata = json.loads(str(values["metadata_json"].item()))
                embedded_reference = {
                    "positions": np.asarray(values["reference_positions"], dtype=np.float64),
                    "atomic_numbers": np.asarray(values["atomic_numbers"], dtype=np.int64),
                    "tags": np.asarray(values["tags"], dtype=np.int64),
                    "cell": np.asarray(values["cell"], dtype=np.float64),
                    "pbc": np.asarray(values["pbc"], dtype=bool),
                }
            reference_path = Path(metadata["reference"]).expanduser()
            if reference_path.is_file():
                actual_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
                if actual_sha != metadata.get("reference_sha256"):
                    raise ValueError(f"Reference SHA256 differs for {system.system_id}")
                tags_from = metadata.get("tags_from")
                tags_pickle = metadata.get("tags_pickle")
                tags_array = (
                    load_tags_pickle(Path(tags_pickle), system.system_id)
                    if tags_pickle
                    else None
                )
                reference = load_reference(
                    reference_path,
                    int(metadata.get("reference_index", 0)),
                    Path(tags_from) if tags_from else None,
                    tags_array,
                )
                metadata["online_r0_reference_source"] = str(reference_path)
            else:
                # The prepared dataset is portable even when provenance keeps
                # the source trajectory's absolute path from another cluster.
                # inputs.npz stores the exact validated reference geometry used
                # to create the original R0 bank, so reconstruct from it.
                from ase import Atoms
                from ase.constraints import FixAtoms

                reference = Atoms(
                    numbers=embedded_reference["atomic_numbers"],
                    positions=embedded_reference["positions"],
                    cell=embedded_reference["cell"],
                    pbc=embedded_reference["pbc"],
                    tags=embedded_reference["tags"],
                )
                reference.set_constraint(
                    FixAtoms(indices=np.flatnonzero(embedded_reference["tags"] == 0))
                )
                metadata["online_r0_reference_source"] = (
                    "random_input/inputs.npz::reference_positions"
                )
            if not np.array_equal(reference.numbers, system.atomic_numbers):
                raise ValueError(f"Reference atom order differs for {system.system_id}")
            if not np.array_equal(reference.get_tags(), _tags_from_roles(system.roles)):
                raise ValueError(f"Reference tags differ for {system.system_id}")
            if not np.allclose(reference.cell.array, system.cell, atol=1e-5, rtol=0):
                raise ValueError(f"Reference cell differs for {system.system_id}")
            self.references[system.system_id] = reference
            self.provenance[system.system_id] = metadata

    def sample(
        self, system: SystemFeatureBank, epoch: int, count: int
    ) -> tuple[np.ndarray, dict]:
        if not self.config.enabled:
            raise RuntimeError("Online R0 resampling is disabled")
        metadata = self.provenance[system.system_id]
        base = epoch_seed(self.config.base_seed, system.system_id, epoch)
        failure = None
        for retry in range(self.config.max_bank_retries):
            # Retry streams cannot overlap adjacent epoch streams.
            seed = base + retry * (1 << 48)
            if seed == int(metadata["seed"]):
                continue
            try:
                batch, report = generate_batch(
                    self.references[system.system_id],
                    count=count,
                    seed=seed,
                    gap_min=float(metadata["gap_range_angstrom"][0]),
                    gap_max=float(metadata["gap_range_angstrom"][1]),
                    covalent_factor=float(metadata["covalent_factor"]),
                    min_distance=float(metadata["min_distance_angstrom"]),
                    vacuum_margin=float(metadata["vacuum_margin_angstrom"]),
                    max_attempts_per_sample=self.config.max_attempts_per_sample,
                    system_id=system.system_id,
                    sampling=str(metadata["sampling"]),
                    min_lateral_separation=float(
                        metadata["min_lateral_separation_angstrom"]
                    ),
                )
                break
            except ValueError as error:
                if "Could not fill all lateral and height strata" not in str(error):
                    raise
                failure = error
        else:
            raise RuntimeError(
                f"Could not resample {system.system_id} epoch {epoch} after "
                f"{self.config.max_bank_retries} deterministic streams"
            ) from failure
        positions = batch["positions"].astype(np.float32, copy=False)
        if positions.shape != (count, system.atom_count, 3):
            raise ValueError("Online R0 batch has an unexpected shape")
        diagnostics = {
            "r0_source": "online_epoch_resample",
            "r0_seed": seed,
            "r0_bank_retry": retry,
            "r0_proposal_count": int(report["proposal_count"]),
            "r0_rejected_contact": int(report["rejected_contact"]),
            "r0_rejected_vacuum": int(report["rejected_vacuum"]),
            "r0_rejected_lateral_duplicate": int(
                report["rejected_lateral_duplicate"]
            ),
            "r0_height_lateral_assignment_attempts": int(
                report["height_lateral_assignment_attempts"]
            ),
            "r0_adsorbate_unwrapped_before_rotation": bool(
                report["adsorbate_unwrapped_before_rotation"]
            ),
            "r0_adsorbate_covalent_bond_count": int(
                report["adsorbate_covalent_bond_count"]
            ),
            "r0_adsorbate_covalent_component_count": int(
                report["adsorbate_covalent_component_count"]
            ),
            "r0_adsorbate_component_bridge_count": int(
                report["adsorbate_component_bridge_count"]
            ),
            "r0_adsorbate_unwrap_max_displacement_A": float(
                report["adsorbate_unwrap_max_displacement_angstrom"]
            ),
            "r0_adsorbate_unwrap_max_bond_error_A": float(
                report["adsorbate_unwrap_max_bond_error_angstrom"]
            ),
        }
        return positions, diagnostics
