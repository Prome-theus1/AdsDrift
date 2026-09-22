"""Validated condition banks for conditional adsorption-structure training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

from AdsDrift.model.generator import (
    ROLE_ADSORBATE,
    ROLE_FIXED,
    ROLE_PADDING,
    ROLE_SURFACE,
)
from AdsDrift.model.condition import (
    FactorizedCondition,
    collate_factorized_conditions,
    load_factorized_condition,
)


ADS_FEATURE_NAMES = ("scalar_ads", "message_l1_ads")
MOVABLE_FEATURE_FILES = {
    "scalar_movable": "scalar_features.npz",
    "message_l1_movable": "l1_norm_features.npz",
}
FEATURE_NAMES = (*ADS_FEATURE_NAMES, *MOVABLE_FEATURE_FILES)


@dataclass(frozen=True)
class SystemFeatureBank:
    """One fixed chemical condition and its R0/positive feature distributions."""

    system_id: str
    directory: Path
    atomic_numbers: np.ndarray
    roles: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray
    r0_positions: np.ndarray
    positive_features: dict[str, np.ndarray]
    movable_feature_paths: dict[str, Path]
    positive_modes: np.ndarray
    condition: FactorizedCondition

    @property
    def atom_count(self) -> int:
        return int(self.atomic_numbers.shape[0])

    @property
    def random_count(self) -> int:
        return int(self.r0_positions.shape[0])

    @property
    def positive_count(self) -> int:
        return int(self.positive_modes.shape[0])

    @property
    def mode_count(self) -> int:
        return int(np.unique(self.positive_modes).size)


def _roles_from_oc20_tags(tags: np.ndarray) -> np.ndarray:
    roles = np.full(tags.shape, ROLE_PADDING, dtype=np.int64)
    roles[tags == 0] = ROLE_FIXED
    roles[tags == 1] = ROLE_SURFACE
    roles[tags == 2] = ROLE_ADSORBATE
    if np.any(roles == ROLE_PADDING):
        raise ValueError("OC20 tags must be exactly 0=fixed, 1=surface, 2=adsorbate")
    return roles


def load_system_bank(
    directory: str | Path,
    movable_feature_root: str | Path | None = None,
) -> SystemFeatureBank:
    directory = Path(directory).resolve()
    input_path = directory / "random_input" / "inputs.npz"
    manifest_path = directory / "structure_manifest.json"
    feature_directory = directory / "features" / "mace"
    for required in (input_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    with np.load(input_path, allow_pickle=False) as values:
        atomic_numbers = np.asarray(values["atomic_numbers"], dtype=np.int64)
        tags = np.asarray(values["tags"], dtype=np.int64)
        cell = np.asarray(values["cell"], dtype=np.float32)
        pbc = np.asarray(values["pbc"], dtype=bool)
        r0_positions = np.asarray(values["positions"], dtype=np.float32)
        stored_system_id = str(np.asarray(values["system_id"]).item())
    manifest = json.loads(manifest_path.read_text())
    system_id = str(manifest.get("system_id", stored_system_id))
    if stored_system_id != system_id:
        raise ValueError(f"System ID mismatch in {directory}")

    if atomic_numbers.ndim != 1 or not np.all((1 <= atomic_numbers) & (atomic_numbers <= 118)):
        raise ValueError(f"Invalid atomic numbers in {directory}")
    if tags.shape != atomic_numbers.shape:
        raise ValueError(f"Tag shape mismatch in {directory}")
    if cell.shape != (3, 3) or abs(float(np.linalg.det(cell))) < 1e-8:
        raise ValueError(f"Invalid cell in {directory}")
    if pbc.shape != (3,) or not pbc[:2].all():
        raise ValueError(f"Invalid slab periodicity in {directory}")
    if r0_positions.ndim != 3 or r0_positions.shape[1:] != (len(tags), 3):
        raise ValueError(f"R0 coordinates have the wrong shape in {directory}")
    if not np.isfinite(r0_positions).all() or not np.isfinite(cell).all():
        raise ValueError(f"Non-finite geometry in {directory}")
    roles = _roles_from_oc20_tags(tags)
    if not np.any(roles == ROLE_ADSORBATE) or not np.any(roles != ROLE_ADSORBATE):
        raise ValueError(f"Condition {system_id} needs slab and adsorbate atoms")
    if not np.any(roles == ROLE_FIXED):
        raise ValueError(f"Condition {system_id} needs a fixed tag-0 reference layer")
    condition = load_factorized_condition(
        directory / "condition_factors.json",
        expected_system_id=system_id,
        expected_adsorbate_atomic_numbers=atomic_numbers[roles == ROLE_ADSORBATE],
    )

    positives = [sample for sample in manifest["samples"]
                 if sample.get("split") == "positive_endpoint"]
    positives.sort(key=lambda sample: int(sample["positive_sample_index"]))
    expected_indices = np.arange(len(positives))
    actual_indices = np.asarray(
        [int(sample["positive_sample_index"]) for sample in positives], dtype=np.int64
    )
    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError(f"Positive sample indices are not contiguous in {directory}")
    positive_modes = np.asarray(
        [int(sample["positive_rank"]) - 1 for sample in positives], dtype=np.int64
    )
    if len(positive_modes) == 0 or np.any(positive_modes < 0):
        raise ValueError(f"No valid positive modes in {directory}")

    positive_features: dict[str, np.ndarray] = {}
    reference_shape: tuple[int, ...] | None = None
    for name in ADS_FEATURE_NAMES:
        path = feature_directory / f"{name}__positive.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        feature = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        if feature.ndim != 3 or feature.shape[0] != len(positive_modes):
            raise ValueError(f"Unexpected {name} shape {feature.shape} in {directory}")
        if not np.isfinite(feature).all():
            raise ValueError(f"Non-finite {name} in {directory}")
        if reference_shape is not None and feature.shape != reference_shape:
            raise ValueError(f"Selected MACE branches have inconsistent shapes in {directory}")
        reference_shape = feature.shape
        positive_features[name] = feature

    colocated = {
        name: feature_directory / f"{name}__positive.npy"
        for name in MOVABLE_FEATURE_FILES
    }
    if movable_feature_root is not None:
        movable_directory = Path(movable_feature_root).expanduser().resolve() / system_id
        movable_feature_paths = {
            name: movable_directory / filename
            for name, filename in MOVABLE_FEATURE_FILES.items()
        }
    elif all(path.is_file() for path in colocated.values()):
        movable_feature_paths = colocated
    else:
        movable_directory = directory.parent.parent / "movable_slab20" / system_id
        movable_feature_paths = {
            name: movable_directory / filename
            for name, filename in MOVABLE_FEATURE_FILES.items()
        }
    movable_count = int(np.count_nonzero(roles == ROLE_SURFACE))
    if movable_count == 0:
        raise ValueError(f"Condition {system_id} has no tag-1 movable surface atoms")
    for name, path in movable_feature_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix == ".npy":
            shape = np.load(path, allow_pickle=False, mmap_mode="r").shape
        else:
            with np.load(path, allow_pickle=False) as values:
                cached_modes = np.asarray(values["modes"], dtype=np.int64)
                shape = values["positive"].shape
            if not np.array_equal(cached_modes, positive_modes):
                raise ValueError(f"Positive endpoint order differs in {path}")
        if shape != (len(positive_modes), 2, movable_count, reference_shape[-1]):
            raise ValueError(f"Unexpected identity-preserving {name} shape {shape} in {path}")

    declared = int(manifest.get("positive_count", len(positive_modes)))
    if declared != len(positive_modes):
        raise ValueError(f"Manifest positive count mismatch in {directory}")
    return SystemFeatureBank(
        system_id=system_id,
        directory=directory,
        atomic_numbers=atomic_numbers,
        roles=roles,
        cell=cell,
        pbc=pbc,
        r0_positions=r0_positions,
        positive_features=positive_features,
        movable_feature_paths=movable_feature_paths,
        positive_modes=positive_modes,
        condition=condition,
    )


def discover_system_directories(roots: Iterable[str | Path]) -> list[Path]:
    """Find completed feature banks without recursively selecting nested folders."""
    found: dict[str, Path] = {}
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        candidates = [root] if (root / "structure_manifest.json").is_file() else []
        if root.is_dir():
            candidates.extend(path.parent for path in root.glob("*/structure_manifest.json"))
        for path in candidates:
            if not (path / "random_input" / "inputs.npz").is_file():
                continue
            manifest = json.loads((path / "structure_manifest.json").read_text())
            system_id = str(manifest.get("system_id", path.name))
            if system_id in found and found[system_id] != path:
                raise ValueError(f"Duplicate condition {system_id}: {found[system_id]} and {path}")
            found[system_id] = path
    if not found:
        raise FileNotFoundError(f"No completed system feature banks found under {list(roots)}")
    return [found[name] for name in sorted(found)]


class ConditionBank:
    """In-memory bank and mode-balanced sampler with explicit construction factors."""

    def __init__(
        self,
        roots: Sequence[str | Path],
        movable_feature_root: str | Path | None = None,
    ) -> None:
        self.systems = [
            load_system_bank(path, movable_feature_root)
            for path in discover_system_directories(roots)
        ]
        shapes = {
            tuple(system.positive_features[ADS_FEATURE_NAMES[0]].shape[1:])
            for system in self.systems
        }
        if len(shapes) != 1:
            raise ValueError(f"Feature shapes differ across conditions: {sorted(shapes)}")
        self.feature_shape = shapes.pop()

    def __len__(self) -> int:
        return len(self.systems)

    @property
    def system_ids(self) -> list[str]:
        return [system.system_id for system in self.systems]

    def sample_conditions(self, count: int, rng: np.random.Generator) -> list[SystemFeatureBank]:
        if count < 1:
            raise ValueError("Condition batch size must be positive")
        indices = rng.choice(len(self), size=count, replace=count > len(self))
        return [self.systems[int(index)] for index in np.atleast_1d(indices)]

    @staticmethod
    def _sample_positive_indices(
        system: SystemFeatureBank,
        positives_per_mode: int,
        max_modes: int | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if positives_per_mode < 1:
            raise ValueError("positives_per_mode must be positive")
        modes = np.unique(system.positive_modes)
        if max_modes is not None and len(modes) > max_modes:
            modes = np.sort(rng.choice(modes, size=max_modes, replace=False))
        selected = []
        for mode in modes:
            members = np.flatnonzero(system.positive_modes == mode)
            selected.extend(
                int(index) for index in rng.choice(
                    members,
                    size=positives_per_mode,
                    replace=positives_per_mode > len(members),
                )
            )
        return np.asarray(selected, dtype=np.int64)

    def make_batch(
        self,
        systems: Sequence[SystemFeatureBank],
        generated_per_condition: int,
        positives_per_mode: int,
        rng: np.random.Generator,
        max_positive_modes: int | None = None,
    ) -> dict[str, Tensor | list[str]]:
        if not systems or generated_per_condition < 1:
            raise ValueError("Need conditions and at least one generated sample")
        if len(systems) != 1:
            raise ValueError("The test_18 training unit is exactly one condition per batch")
        batch_size = len(systems)
        max_atoms = max(system.atom_count for system in systems)
        numbers = np.zeros((batch_size, max_atoms), dtype=np.int64)
        roles = np.full((batch_size, max_atoms), ROLE_PADDING, dtype=np.int64)
        atom_mask = np.zeros((batch_size, max_atoms), dtype=bool)
        cells = np.zeros((batch_size, 3, 3), dtype=np.float32)
        r0 = np.zeros((batch_size, generated_per_condition, max_atoms, 3), dtype=np.float32)
        positive_conditions = []
        positive_modes = []
        positive = {name: [] for name in FEATURE_NAMES}

        for condition, system in enumerate(systems):
            n_atoms = system.atom_count
            numbers[condition, :n_atoms] = system.atomic_numbers
            roles[condition, :n_atoms] = system.roles
            atom_mask[condition, :n_atoms] = True
            cells[condition] = system.cell
            if generated_per_condition == system.random_count:
                # A full-system batch consumes the complete, fixed R0 bank.
                r0_indices = np.arange(system.random_count)
            else:
                r0_indices = rng.choice(
                    system.random_count,
                    size=generated_per_condition,
                    replace=generated_per_condition > system.random_count,
                )
            r0[condition, :, :n_atoms] = system.r0_positions[r0_indices]
            target_indices = self._sample_positive_indices(
                system, positives_per_mode, max_positive_modes, rng
            )
            for name in ADS_FEATURE_NAMES:
                positive[name].append(system.positive_features[name][target_indices])
            # Compressed diagnostic archives are opened lazily so hundreds of
            # identity-preserving surface tensors do not stay resident in RAM.
            for name, path in system.movable_feature_paths.items():
                if path.suffix == ".npy":
                    values = np.load(path, allow_pickle=False, mmap_mode="r")
                    feature = np.asarray(values[target_indices], dtype=np.float32)
                else:
                    with np.load(path, allow_pickle=False) as values:
                        feature = values["positive"][target_indices].astype(
                            np.float32, copy=False
                        )
                positive[name].append(feature)
            positive_conditions.extend([condition] * len(target_indices))
            positive_modes.extend(system.positive_modes[target_indices].tolist())

        batch: dict[str, Tensor | list[str]] = {
            "atomic_numbers": torch.from_numpy(numbers),
            "roles": torch.from_numpy(roles),
            "atom_mask": torch.from_numpy(atom_mask),
            "cell": torch.from_numpy(cells),
            "r0_positions": torch.from_numpy(r0),
            "positive_condition": torch.tensor(positive_conditions, dtype=torch.long),
            "positive_modes": torch.tensor(positive_modes, dtype=torch.long),
            "system_ids": [system.system_id for system in systems],
            "condition": collate_factorized_conditions(
                [system.condition for system in systems]
            ),
        }
        for name in FEATURE_NAMES:
            batch[f"positive_{name}"] = torch.from_numpy(np.concatenate(positive[name], axis=0))
        return batch


def move_batch(batch: dict, device: torch.device | str) -> dict:
    def move(value):
        if isinstance(value, Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value

    return {key: move(value) for key, value in batch.items()}
