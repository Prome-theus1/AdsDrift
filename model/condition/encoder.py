"""Invariant encoder for factorized crystal construction variables."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weight = mask.to(values.dtype)
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    return (values * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def _cell_signature(cell: Tensor, atom_count: Tensor | None = None) -> Tensor:
    """Rotation-invariant lengths, angles, and intensive log-volume."""
    lengths = torch.linalg.vector_norm(cell, dim=-1).clamp_min(1.0e-8)
    normalized = cell / lengths[..., None]
    cosines = torch.stack(
        (
            (normalized[:, 0] * normalized[:, 1]).sum(-1),
            (normalized[:, 0] * normalized[:, 2]).sum(-1),
            (normalized[:, 1] * normalized[:, 2]).sum(-1),
        ),
        dim=-1,
    )
    volume = torch.linalg.det(cell).abs().clamp_min(1.0e-8)
    if atom_count is not None:
        volume = volume / atom_count.to(volume.dtype).clamp_min(1.0)
    return torch.cat((torch.log(lengths), cosines, torch.log(volume)[:, None]), dim=-1)


class FactorizedConditionEncoder(nn.Module):
    """Encode chemistry and construction factors without a system-ID embedding.

    All outputs are l=0 scalars.  Conditioning an equivariant trunk with these
    values therefore preserves its rotation covariance.
    """

    def __init__(
        self,
        *,
        num_channels: int,
        num_layers: int,
        hidden_channels: int,
        primitive_message_layers: int,
        num_radial_basis: int,
        max_distance_A: float,
        max_atomic_number: int,
    ) -> None:
        super().__init__()
        if min(num_channels, num_layers, hidden_channels, primitive_message_layers, num_radial_basis) < 1:
            raise ValueError("Condition encoder dimensions must be positive")
        if max_distance_A <= 0:
            raise ValueError("condition_max_distance_A must be positive")
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.num_radial_basis = num_radial_basis
        self.max_distance_A = max_distance_A
        self.element_embedding = nn.Embedding(
            max_atomic_number + 1, hidden_channels, padding_idx=0
        )
        centers = torch.linspace(0.0, max_distance_A, num_radial_basis)
        self.register_buffer("radial_centers", centers, persistent=False)
        spacing = max_distance_A / max(num_radial_basis - 1, 1)
        self.radial_gamma = 1.0 / max(spacing, 1.0e-3) ** 2
        pair_width = 2 * hidden_channels + num_radial_basis + 2
        self.primitive_messages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(pair_width, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                for _ in range(primitive_message_layers)
            ]
        )
        self.primitive_updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(2 * hidden_channels),
                    nn.Linear(2 * hidden_channels, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                for _ in range(primitive_message_layers)
            ]
        )
        # primitive pool + adsorbate composition pool + 54 geometric scalars
        factor_width = 2 * hidden_channels + 54
        self.factor_mlp = nn.Sequential(
            nn.LayerNorm(factor_width),
            nn.Linear(factor_width, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
        )
        self.initial_role_bias = nn.Linear(hidden_channels, 3 * num_channels)
        self.layer_modulation = nn.Linear(
            hidden_channels, num_layers * 3 * 2 * num_channels
        )

    def _primitive_embedding(self, condition: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        numbers = condition["primitive_atomic_numbers"]
        fractions = condition["primitive_fractional_positions"]
        mask = condition["primitive_atom_mask"].bool()
        cell = condition["primitive_cell"].to(fractions.dtype)
        conventional_cell = condition["miller_conventional_cell"].to(
            fractions.dtype
        )
        miller = condition["miller_index"].to(fractions.dtype)
        hidden = self.element_embedding(numbers)

        delta_fractional = fractions[:, None, :, :] - fractions[:, :, None, :]
        delta_fractional = delta_fractional - torch.round(delta_fractional)
        delta = torch.einsum("bijc,bcd->bijd", delta_fractional, cell)
        distance = torch.linalg.vector_norm(delta, dim=-1)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        eye = torch.eye(mask.shape[1], device=mask.device, dtype=torch.bool)
        pair_mask = pair_mask & ~eye[None] & (distance < self.max_distance_A)

        reciprocal_normal = torch.bmm(
            miller[:, None, :],
            torch.linalg.inv(conventional_cell).transpose(1, 2),
        ).squeeze(1)
        normal_length = torch.linalg.vector_norm(
            reciprocal_normal, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        normal = reciprocal_normal / normal_length
        normal_projection = torch.einsum("bijd,bd->bij", delta, normal)
        lateral = (
            distance.square() - normal_projection.square()
        ).clamp_min(0.0).sqrt()
        radial = torch.exp(
            -self.radial_gamma
            * (distance[..., None] - self.radial_centers.to(distance.dtype)).square()
        )
        geometry = torch.cat(
            (
                radial,
                (normal_projection / self.max_distance_A)[..., None],
                (lateral / self.max_distance_A)[..., None],
            ),
            dim=-1,
        )
        for message_mlp, update_mlp in zip(
            self.primitive_messages, self.primitive_updates
        ):
            target = hidden[:, :, None, :].expand(-1, -1, hidden.shape[1], -1)
            source = hidden[:, None, :, :].expand(-1, hidden.shape[1], -1, -1)
            messages = message_mlp(torch.cat((target, source, geometry), dim=-1))
            messages = messages * pair_mask[..., None]
            denominator = pair_mask.sum(dim=2, keepdim=True).clamp_min(1)
            aggregate = messages.sum(dim=2) / denominator
            hidden = hidden + update_mlp(torch.cat((hidden, aggregate), dim=-1))
            hidden = hidden * mask[..., None]
        return _masked_mean(hidden, mask, dim=1), normal_length

    def forward(
        self,
        condition: dict[str, Tensor],
        atomic_numbers: Tensor,
        roles: Tensor,
        atom_mask: Tensor,
        slab_cell: Tensor,
    ) -> tuple[Tensor, Tensor]:
        primitive, reciprocal_normal_length = self._primitive_embedding(condition)
        ads_mask = atom_mask.bool() & (roles == 3)
        adsorbate = _masked_mean(
            self.element_embedding(atomic_numbers), ads_mask, dim=1
        )
        primitive_count = condition["primitive_atom_mask"].sum(dim=1)
        primitive_cell = condition["primitive_cell"].to(slab_cell.dtype)
        conventional_cell = condition["miller_conventional_cell"].to(
            slab_cell.dtype
        )
        matrix = condition["supercell_matrix"].to(slab_cell.dtype)
        supercell = torch.bmm(matrix, primitive_cell)
        miller = condition["miller_index"].to(slab_cell.dtype)
        miller = miller / torch.linalg.vector_norm(miller, dim=-1, keepdim=True).clamp_min(1.0)
        shift = condition["termination_shift"].to(slab_cell.dtype)
        surface = torch.cat(
            (
                miller,
                torch.log(reciprocal_normal_length.to(slab_cell.dtype)),
                torch.sin(2.0 * math.pi * shift),
                torch.cos(2.0 * math.pi * shift),
                2.0 * condition["top"].to(slab_cell.dtype) - 1.0,
            ),
            dim=-1,
        )
        matrix_continuous = torch.sign(matrix) * torch.log1p(matrix.abs())
        matrix_det = torch.linalg.det(matrix).abs().clamp_min(1.0)
        construction = torch.cat(
            (
                torch.log1p(condition["slab_layers"].to(slab_cell.dtype)),
                torch.log1p(condition["vacuum_A"].to(slab_cell.dtype)),
                condition["strain_voigt"].to(slab_cell.dtype),
                torch.log1p(condition["adsorbates_per_cell"].to(slab_cell.dtype)),
            ),
            dim=-1,
        )
        factors = torch.cat(
            (
                primitive,
                adsorbate,
                _cell_signature(primitive_cell, primitive_count),
                _cell_signature(conventional_cell),
                _cell_signature(slab_cell),
                surface,
                matrix_continuous.flatten(1),
                _cell_signature(supercell),
                torch.log(matrix_det)[:, None],
                construction,
            ),
            dim=-1,
        )
        embedding = self.factor_mlp(factors)
        initial = self.initial_role_bias(embedding).view(
            -1, 3, self.num_channels
        )
        modulation = self.layer_modulation(embedding).view(
            -1, self.num_layers, 3, 2, self.num_channels
        )
        return initial, modulation
