"""Differentiable frozen MACE-MH-1 adsorbate feature encoder.

The radius-graph membership is discrete and rebuilt from detached coordinates.
Wrapped coordinates and periodic shifts remain Torch expressions, so gradients
from both selected feature branches propagate back to generated coordinates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from AdsDrift.model.generator import ROLE_ADSORBATE, ROLE_SURFACE


DEFAULT_MACE_CHECKPOINT = Path(
    os.environ.get(
        "ADSDRIFT_MACE_CHECKPOINT",
        Path(__file__).resolve().parent / "mace_pt" / "macemh1model.pt",
    )
).expanduser()


@dataclass(frozen=True)
class MACEFeatureConfig:
    checkpoint: str = str(DEFAULT_MACE_CHECKPOINT)
    sha256: str = "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47"
    head: str = "oc20_usemppbe"
    microbatch_size: int = 8
    min_edge_distance_A: float = 1e-8
    max_edges_per_structure: int = 2_000_000
    variance_epsilon: float = 1e-12
    activation_checkpointing: bool = True

    @classmethod
    def from_dict(cls, values: dict | None = None) -> "MACEFeatureConfig":
        values = dict(values or {})
        known = set(cls.__dataclass_fields__)
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown MACE feature settings: {sorted(unknown)}")
        config = cls(**values)
        if config.microbatch_size < 1 or config.max_edges_per_structure < 1:
            raise ValueError("MACE batch and edge limits must be positive")
        if min(config.min_edge_distance_A, config.variance_epsilon) <= 0:
            raise ValueError("MACE numerical floors must be positive")
        return config


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_backbone(config: MACEFeatureConfig) -> nn.Module:
    path = Path(config.checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"MACE-MH-1 checkpoint not found at {path}. The checkpoint is not "
            "distributed with AdsDrift. Set ADSDRIFT_MACE_CHECKPOINT to a "
            "separately obtained checkpoint or pass mace.checkpoint in your "
            "configuration, and comply with the upstream model license."
        )
    actual = _file_sha256(path)
    if config.sha256 and actual != config.sha256:
        raise ValueError(f"MACE checkpoint SHA256 mismatch: {actual}")
    # The official release is a serialized module rather than a state dict.
    import mace.modules  # noqa: F401

    backbone = torch.load(path, map_location="cpu", weights_only=False).float()
    return backbone.requires_grad_(False).eval()


class FrozenMACEInterfaceFeatures(nn.Module):
    """Return pooled adsorbate and identity-preserving movable-layer features.

    Adsorbate branches have shape [S,4,512] (mean/std for two interactions).
    Movable branches have shape [S,2,M,512], retaining the fixed tag-1 atom
    ordering within the condition.
    """

    def __init__(self, config: MACEFeatureConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = _load_backbone(config)
        self.elements = [int(value) for value in self.backbone.atomic_numbers]
        element_lookup = torch.full((119,), -1, dtype=torch.long)
        element_lookup[self.elements] = torch.arange(len(self.elements))
        self.register_buffer("element_lookup", element_lookup, persistent=False)
        self.heads = list(self.backbone.heads)
        if config.head not in self.heads:
            raise ValueError(f"MACE head {config.head!r} is absent; choices: {self.heads}")
        self.head_index = self.heads.index(config.head)
        self.cutoff = float(self.backbone.r_max)
        self.layers = int(self.backbone.num_interactions)
        if self.layers != 2 or len(self.backbone.interactions) != 2:
            raise ValueError("The selected v1 readout requires exactly two MACE interactions")

        from e3nn import o3

        irreps = o3.Irreps(str(self.backbone.products[0].linear.irreps_out))
        self.lmax = irreps.lmax
        self.width = irreps.dim // (self.lmax + 1) ** 2
        if self.width != 512:
            raise ValueError(f"Expected 512 MACE channels, found {self.width}")
        expected_message = "+".join(
            f"{self.width}x{ell}{'e' if ell % 2 == 0 else 'o'}"
            for ell in range(4)
        )
        for interaction in self.backbone.interactions:
            if str(interaction.irreps_out) != expected_message:
                raise ValueError(f"Unexpected interaction irreps: {interaction.irreps_out}")
        self._messages: list[Tensor] = []
        self._hooks = [
            interaction.register_forward_hook(self._capture_message)
            for interaction in self.backbone.interactions
        ]
        self.eval()

    def _capture_message(self, module, inputs, output) -> None:
        del module, inputs
        self._messages.append(output[0])

    def train(self, mode: bool = True):
        # Frozen feature statistics must never switch to training behavior.
        super().train(False)
        return self

    def _graph(
        self,
        atomic_numbers: Tensor,
        positions: Tensor,
        cell: Tensor,
        atom_mask: Tensor,
    ) -> dict[str, Tensor]:
        from matscipy.neighbours import neighbour_list

        device = positions.device
        dtype = positions.dtype
        packed_positions = []
        packed_cells = []
        edges = []
        unit_shifts = []
        shifts = []
        attributes = []
        counts = []
        offset = 0
        for structure in range(len(positions)):
            valid = atom_mask[structure].bool()
            pos = positions[structure, valid]
            numbers = atomic_numbers[structure, valid]
            box = cell[structure]
            if len(pos) == 0:
                raise ValueError("MACE received an empty structure")
            if not torch.isfinite(pos).all() or not torch.isfinite(box).all():
                raise FloatingPointError("MACE received non-finite geometry")
            species = self.element_lookup[numbers]
            if (species < 0).any():
                bad = numbers[species < 0].detach().cpu().tolist()
                raise ValueError(f"MACE checkpoint does not support atomic numbers {bad}")

            fractional = pos @ torch.linalg.inv(box)
            images = torch.floor(fractional.detach()) * pos.new_tensor([1.0, 1.0, 0.0])
            wrapped = pos - images @ box
            numpy_positions = wrapped.detach().double().cpu().numpy()
            numpy_cell = box.detach().double().cpu().numpy()
            i, j, lattice_shift = neighbour_list(
                "ijS",
                pbc=[True, True, False],
                cell=numpy_cell,
                positions=numpy_positions,
                cutoff=self.cutoff,
            )
            displacement = (
                numpy_positions[j] - numpy_positions[i] + lattice_shift @ numpy_cell
            )
            distance = np.linalg.norm(displacement, axis=1)
            keep = distance > self.config.min_edge_distance_A
            i, j, lattice_shift = i[keep], j[keep], lattice_shift[keep]
            if len(i) > self.config.max_edges_per_structure:
                raise ValueError(
                    "MACE radius graph exceeds max_edges_per_structure; "
                    "neighbors were not truncated"
                )
            if len(lattice_shift) and np.any(lattice_shift[:, 2]):
                raise ValueError("Unexpected MACE periodic edge through the vacuum axis")

            unit = torch.as_tensor(lattice_shift, device=device, dtype=dtype)
            edge = torch.as_tensor(
                np.stack([i, j]) + offset, device=device, dtype=torch.long
            )
            packed_positions.append(wrapped)
            packed_cells.append(box)
            edges.append(edge)
            unit_shifts.append(unit)
            shifts.append(unit @ box)
            attributes.append(
                torch.nn.functional.one_hot(species, len(self.elements)).to(dtype)
            )
            counts.append(len(pos))
            offset += len(pos)

        ptr = torch.tensor([0, *np.cumsum(counts).tolist()], device=device)
        return {
            "positions": torch.cat(packed_positions),
            "cell": torch.stack(packed_cells),
            "edge_index": torch.cat(edges, dim=1),
            "node_attrs": torch.cat(attributes),
            "shifts": torch.cat(shifts),
            "unit_shifts": torch.cat(unit_shifts),
            "ptr": ptr,
            "batch": torch.repeat_interleave(
                torch.arange(len(counts), device=device),
                torch.tensor(counts, device=device),
            ),
            "head": torch.full(
                (len(counts),), self.head_index, device=device, dtype=torch.long
            ),
        }

    def _pool(self, node_layers: list[Tensor], mask: Tensor) -> Tensor:
        groups = []
        for values in node_layers:
            selected = values[mask]
            if len(selected) == 0:
                raise ValueError("Cannot pool an empty adsorbate")
            variance = selected.double().var(dim=0, unbiased=False)
            groups.extend(
                [selected.mean(dim=0),
                 (variance + self.config.variance_epsilon).sqrt().to(values.dtype)]
            )
        return torch.stack(groups)

    def _forward_chunk(
        self,
        atomic_numbers: Tensor,
        positions: Tensor,
        cell: Tensor,
        atom_mask: Tensor,
        roles: Tensor,
    ) -> dict[str, Tensor]:
        from mace.modules.utils import extract_invariant

        graph = self._graph(atomic_numbers, positions, cell, atom_mask)
        self._messages = []
        output = self.backbone(
            graph,
            training=False,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
        )
        if len(self._messages) != self.layers:
            raise RuntimeError("Not all MACE interaction hooks ran")
        invariant = extract_invariant(
            output["node_feats"],
            num_layers=self.layers,
            num_features=self.width,
            l_max=self.lmax,
        )
        scalar_layers = list(invariant.split(self.width, dim=-1))
        angular_layers = []
        for message in self._messages:
            if tuple(message.shape[1:]) != (self.width, 16):
                raise ValueError(f"Unexpected MACE message shape {tuple(message.shape)}")
            angular_layers.append(
                message[:, :, 1:4].double().square().sum(dim=-1)
                .clamp_min(1e-24).sqrt().to(message.dtype)
            )

        scalar_output = []
        angular_output = []
        movable_scalar_output = []
        movable_angular_output = []
        offset = 0
        for structure in range(len(positions)):
            valid = atom_mask[structure].bool()
            count = int(valid.sum())
            adsorbate = roles[structure, valid] == ROLE_ADSORBATE
            movable = roles[structure, valid] == ROLE_SURFACE
            if not movable.any():
                raise ValueError("Cannot read out an empty movable surface layer")
            scalar_output.append(
                self._pool([values[offset : offset + count] for values in scalar_layers], adsorbate)
            )
            angular_output.append(
                self._pool([values[offset : offset + count] for values in angular_layers], adsorbate)
            )
            movable_scalar_output.append(
                torch.stack(
                    [values[offset : offset + count][movable] for values in scalar_layers]
                )
            )
            movable_angular_output.append(
                torch.stack(
                    [values[offset : offset + count][movable] for values in angular_layers]
                )
            )
            offset += count
        self._messages = []
        return {
            "scalar_ads": torch.stack(scalar_output),
            "message_l1_ads": torch.stack(angular_output),
            "scalar_movable": torch.stack(movable_scalar_output),
            "message_l1_movable": torch.stack(movable_angular_output),
        }

    def forward(
        self,
        atomic_numbers: Tensor,
        positions: Tensor,
        cell: Tensor,
        atom_mask: Tensor,
        roles: Tensor,
    ) -> dict[str, Tensor]:
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("MACE positions must have shape [S,N,3]")
        expected = positions.shape[:2]
        if atomic_numbers.shape != expected or atom_mask.shape != expected or roles.shape != expected:
            raise ValueError("MACE atom arrays have inconsistent shapes")
        if cell.shape != (len(positions), 3, 3):
            raise ValueError("MACE cells must have shape [S,3,3]")
        outputs = {
            "scalar_ads": [],
            "message_l1_ads": [],
            "scalar_movable": [],
            "message_l1_movable": [],
        }
        size = self.config.microbatch_size
        # Autocast remains off: the verified checkpoint/readout was evaluated in fp32.
        with torch.autocast(device_type=positions.device.type, enabled=False):
            for start in range(0, len(positions), size):
                stop = min(start + size, len(positions))
                arguments = (
                    atomic_numbers[start:stop],
                    positions[start:stop].float(),
                    cell[start:stop].float(),
                    atom_mask[start:stop],
                    roles[start:stop],
                )
                if self.config.activation_checkpointing and torch.is_grad_enabled() \
                        and arguments[1].requires_grad:
                    def checkpointed(*chunk_arguments):
                        values = self._forward_chunk(*chunk_arguments)
                        return tuple(values[name] for name in outputs)

                    # Full recomputation keeps hook state deterministic and avoids
                    # retaining 100 copies of MACE activations until backward.
                    with set_checkpoint_early_stop(False):
                        checkpointed_values = checkpoint(
                            checkpointed, *arguments, use_reentrant=False
                        )
                    result = dict(zip(outputs, checkpointed_values))
                else:
                    result = self._forward_chunk(*arguments)
                for name in outputs:
                    outputs[name].append(result[name])
        return {name: torch.cat(values) for name, values in outputs.items()}


def default_mace_config() -> dict:
    """Fixed training encoder settings; diagnostic scripts may copy and override."""
    return asdict(MACEFeatureConfig())


def build_mace_features(config: dict | None = None) -> FrozenMACEInterfaceFeatures:
    return FrozenMACEInterfaceFeatures(MACEFeatureConfig.from_dict(config))
