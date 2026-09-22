"""Pure-MACE adsorbate coordinate-gradient projection and balancing.

The frozen feature-space loss is differentiated once with respect to an
isolated coordinate leaf.  Its adsorbate gradient is then split into mutually
orthogonal rigid-translation, infinitesimal rigid-rotation, and internal
components.  Only their relative magnitudes are changed; their MACE-derived
directions and the original per-candidate total gradient energy are retained.

All operations in this module run under ``no_grad``.  The balanced field is
injected into the generator through a linear straight-through surrogate, so
no Hessian or second backward graph through MACE is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from AdsDrift.model.generator import ROLE_ADSORBATE


@dataclass(frozen=True)
class CoordinateGradientBalancingConfig:
    """YAML-controlled amplitudes for [translation, rotation, internal]."""

    component_energy_fractions: tuple[float, float, float] = (0.2, 0.2, 0.6)
    energy_floor: float = 1.0e-20
    maximum_component_rescale: float = 10000.0
    preserve_total_energy: bool = True

    @classmethod
    def from_dict(
        cls, values: dict | None
    ) -> "CoordinateGradientBalancingConfig":
        values = dict(values or {})
        if "component_energy_fractions" in values:
            values["component_energy_fractions"] = tuple(
                float(value) for value in values["component_energy_fractions"]
            )
        known = set(cls.__dataclass_fields__)
        unknown = set(values) - known
        if unknown:
            raise ValueError(
                f"Unknown coordinate-gradient balancing settings: {sorted(unknown)}"
            )
        config = cls(**values)
        if len(config.component_energy_fractions) != 3:
            raise ValueError(
                "component_energy_fractions must contain "
                "[translation, rotation, internal]"
            )
        if any(value < 0.0 for value in config.component_energy_fractions):
            raise ValueError(
                "Coordinate-gradient component energy fractions must be non-negative"
            )
        if abs(sum(config.component_energy_fractions) - 1.0) > 1.0e-6:
            raise ValueError(
                "Coordinate-gradient component energy fractions must sum to one"
            )
        if config.energy_floor <= 0.0 or config.maximum_component_rescale <= 0.0:
            raise ValueError("Coordinate-gradient numerical limits must be positive")
        return config


def _gather_adsorbate(values: Tensor, mask: Tensor) -> Tensor:
    counts = mask.sum(dim=1)
    if len(counts) == 0 or int(counts.min()) < 1:
        raise ValueError("Every structure needs at least one adsorbate atom")
    if not torch.equal(counts, counts[:1].expand_as(counts)):
        raise ValueError("A coordinate-gradient batch needs a common adsorbate size")
    return values[mask].reshape(len(values), int(counts[0]), 3)


def rigid_gradient_components(
    positions: Tensor,
    gradient: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Orthogonally split a [S,A,3] field into translation/rotation/internal."""

    if positions.shape != gradient.shape or positions.ndim != 3:
        raise ValueError("Positions and gradient must both have shape [S,A,3]")
    if positions.shape[-1] != 3:
        raise ValueError("The final coordinate dimension must be three")
    relative = positions - positions.mean(dim=1, keepdim=True)
    translation = gradient.mean(dim=1, keepdim=True).expand_as(gradient)
    residual = gradient - translation
    identity = torch.eye(3, device=positions.device, dtype=positions.dtype)
    inertia = (
        relative.square().sum(dim=-1)[:, :, None, None] * identity
        - relative[:, :, :, None] * relative[:, :, None, :]
    ).sum(dim=1)
    torque = torch.linalg.cross(relative, residual, dim=-1).sum(dim=1)
    angular = torch.bmm(torch.linalg.pinv(inertia), torque[:, :, None]).squeeze(-1)
    rotation = torch.linalg.cross(angular[:, None], relative, dim=-1)
    internal = residual - rotation
    return translation, rotation, internal


def _energy(values: Tensor) -> Tensor:
    return values.square().sum(dim=(1, 2))


def _mean_fraction(component: Tensor, total: Tensor, floor: float) -> Tensor:
    total_energy = _energy(total)
    valid = total_energy > floor
    if not valid.any():
        return total_energy.new_zeros(())
    return (_energy(component)[valid] / total_energy[valid]).mean()


class CoordinateGradientBalancer(nn.Module):
    """Reweight orthogonal pure-MACE adsorbate coordinate directions."""

    def __init__(self, config: CoordinateGradientBalancingConfig) -> None:
        super().__init__()
        self.config = config

    @torch.no_grad()
    def forward(
        self,
        positions: Tensor,
        coordinate_gradient: Tensor,
        roles: Tensor,
        atom_mask: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if positions.shape != coordinate_gradient.shape:
            raise ValueError("Position and coordinate-gradient tensors must match")
        if positions.ndim != 4 or positions.shape[-1] != 3:
            raise ValueError("Generated positions must have shape [B,G,N,3]")
        batch, generated, atoms = positions.shape[:3]
        flat_positions = positions.reshape(batch * generated, atoms, 3)
        flat_gradient = coordinate_gradient.reshape(batch * generated, atoms, 3)
        flat_roles = roles.repeat_interleave(generated, dim=0)
        flat_mask = atom_mask.repeat_interleave(generated, dim=0)
        adsorbate_mask = flat_mask & (flat_roles == ROLE_ADSORBATE)

        ads_positions = _gather_adsorbate(flat_positions, adsorbate_mask)
        ads_gradient = _gather_adsorbate(flat_gradient, adsorbate_mask)
        components = rigid_gradient_components(ads_positions, ads_gradient)
        component_energy = torch.stack([_energy(value) for value in components], dim=1)
        total_energy = component_energy.sum(dim=1)

        configured_fraction = ads_gradient.new_tensor(
            self.config.component_energy_fractions
        )
        requested_energy = configured_fraction[None].expand_as(component_energy)
        active = component_energy > self.config.energy_floor
        requested_energy = requested_energy * active
        requested_sum = requested_energy.sum(dim=1, keepdim=True)
        target_fraction = torch.where(
            requested_sum > 0.0,
            requested_energy / requested_sum.clamp_min(self.config.energy_floor),
            torch.zeros_like(requested_energy),
        )
        raw_rescale = (
            target_fraction
            * total_energy[:, None]
            / component_energy.clamp_min(self.config.energy_floor)
        ).clamp_min(0.0).sqrt()
        raw_rescale = torch.where(active, raw_rescale, torch.zeros_like(raw_rescale))
        rescale = raw_rescale.clamp_max(self.config.maximum_component_rescale)
        scaled_components = tuple(
            component * rescale[:, index, None, None]
            for index, component in enumerate(components)
        )
        balanced_adsorbate = sum(scaled_components)

        if self.config.preserve_total_energy:
            balanced_energy = _energy(balanced_adsorbate)
            total_rescale = torch.where(
                balanced_energy > self.config.energy_floor,
                (total_energy / balanced_energy.clamp_min(self.config.energy_floor)).sqrt(),
                torch.zeros_like(balanced_energy),
            )
            balanced_adsorbate = balanced_adsorbate * total_rescale[:, None, None]
            scaled_components = tuple(
                value * total_rescale[:, None, None] for value in scaled_components
            )

        balanced_flat = flat_gradient.clone()
        balanced_flat[adsorbate_mask] = balanced_adsorbate.reshape(-1, 3)
        balanced = balanced_flat.reshape_as(positions).detach()
        original_total = sum(components)

        names = ("translation", "rotation", "internal")
        metrics: dict[str, Tensor] = {
            "coordinate_balancing_enabled": positions.new_ones(()),
            "coordinate_gradient_rms_before": coordinate_gradient.square().mean().sqrt(),
            "coordinate_gradient_rms_after": balanced.square().mean().sqrt(),
            "ads_coordinate_gradient_rms_before": ads_gradient.square().mean().sqrt(),
            "ads_coordinate_gradient_rms_after": balanced_adsorbate.square().mean().sqrt(),
        }
        for index, name in enumerate(names):
            metrics[f"mace_ads_{name}_energy_fraction"] = _mean_fraction(
                components[index], original_total, self.config.energy_floor
            )
            metrics[f"balanced_ads_{name}_energy_fraction"] = _mean_fraction(
                scaled_components[index], balanced_adsorbate, self.config.energy_floor
            )
            metrics[f"target_ads_{name}_energy_fraction"] = target_fraction[:, index].mean()
            metrics[f"ads_{name}_rescale_mean"] = rescale[:, index].mean()
            metrics[f"ads_{name}_rescale_max"] = rescale[:, index].max()
            metrics[f"ads_{name}_rescale_clipped_fraction"] = (
                raw_rescale[:, index] > self.config.maximum_component_rescale
            ).to(positions.dtype).mean()
        return balanced, metrics


def value_preserving_coordinate_surrogate(
    value: Tensor,
    positions: Tensor,
    coordinate_gradient: Tensor,
) -> Tensor:
    """Keep ``value`` numerically while prescribing d(value)/d(positions)."""

    if value.ndim != 0:
        raise ValueError("The reported loss value must be scalar")
    return value.detach() + (
        (positions - positions.detach()) * coordinate_gradient.detach()
    ).sum()


def build_coordinate_gradient_balancer(
    config: dict | None,
) -> CoordinateGradientBalancer:
    return CoordinateGradientBalancer(
        CoordinateGradientBalancingConfig.from_dict(config)
    )
