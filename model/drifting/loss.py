"""Paper-faithful conditional feature-space Drifting objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


ADSORBATE_FEATURE_BRANCHES = frozenset(("scalar_ads", "message_l1_ads"))
SURFACE_FEATURE_BRANCHES = frozenset(("scalar_movable", "message_l1_movable"))


def _branch_family(name: str) -> str:
    if name in ADSORBATE_FEATURE_BRANCHES:
        return "adsorbate"
    if name in SURFACE_FEATURE_BRANCHES:
        return "surface"
    raise ValueError(f"Unknown feature branch for coordinate routing: {name}")


@dataclass(frozen=True)
class DriftingConfig:
    temperatures: tuple[float, ...] = (0.02, 0.05, 0.2)
    feature_scale_floor: float = 1e-8
    drift_scale_floor: float = 1e-10

    @classmethod
    def from_dict(cls, values: dict) -> "DriftingConfig":
        values = dict(values)
        if "temperatures" in values:
            values["temperatures"] = tuple(float(value) for value in values["temperatures"])
        known = set(cls.__dataclass_fields__)
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown Drifting settings: {sorted(unknown)}")
        config = cls(**values)
        if not config.temperatures or min(config.temperatures) <= 0:
            raise ValueError("Drifting temperatures must be positive")
        if min(config.feature_scale_floor, config.drift_scale_floor) <= 0:
            raise ValueError("Drifting numerical floors must be positive")
        return config


@torch.no_grad()
def bidirectional_weights(
    generated: Tensor,
    positive: Tensor,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """Algorithm 2 weights with the generated batch as self-masked negatives."""
    if generated.ndim != 2 or positive.ndim != 2:
        raise ValueError("Generated and positive features must be rank two")
    if generated.shape[1] != positive.shape[1]:
        raise ValueError("Generated and positive features need equal channels")
    if len(generated) < 2 or len(positive) < 1:
        raise ValueError("Drifting needs at least two generated and one positive sample")
    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    positive_distance = torch.cdist(generated.float(), positive.float())
    negative_distance = torch.cdist(generated.float(), generated.float())
    negative_distance.fill_diagonal_(torch.inf)
    logits = -torch.cat([positive_distance, negative_distance], dim=1) / temperature
    # sqrt(softmax_rows * softmax_columns), evaluated in log space.
    affinity = (
        0.5 * (logits.log_softmax(dim=1) + logits.log_softmax(dim=0))
    ).exp()
    positive_affinity = affinity[:, : len(positive)]
    negative_affinity = affinity[:, len(positive) :]
    positive_weight = positive_affinity * negative_affinity.sum(dim=1, keepdim=True)
    negative_weight = negative_affinity * positive_affinity.sum(dim=1, keepdim=True)
    return positive_weight, negative_weight


@torch.no_grad()
def bidirectional_drift(
    generated: Tensor,
    positive: Tensor,
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Algorithm 2 field for one feature tensor."""
    positive_weight, negative_weight = bidirectional_weights(
        generated, positive, temperature
    )
    field = positive_weight @ positive - negative_weight @ generated
    return field, positive_weight, negative_weight


@torch.no_grad()
def feature_scale(generated: Tensor, positive: Tensor) -> Tensor:
    """Eq. 21 Monte-Carlo scale, including the generated negative batch."""
    channels = generated.shape[1]
    distance = torch.cat(
        [torch.cdist(generated.float(), positive.float()),
         torch.cdist(generated.float(), generated.float())],
        dim=1,
    )
    return distance.mean() / channels**0.5


class FeatureDriftingLoss(nn.Module):
    """A shared joint kernel with independently normalized feature fields.

    One positive endpoint index supplies all adsorbate and movable-layer
    branches. Each group has equal influence on the joint distance regardless
    of channel count, so a surface with many movable atoms cannot swamp the
    adsorbate representation.
    """

    def __init__(self, config: DriftingConfig) -> None:
        super().__init__()
        self.config = config

    @staticmethod
    def _temperature_metric_name(temperature: float) -> str:
        value = f"{temperature:g}".replace("-", "m").replace(".", "p")
        return f"raw_drift_loss_tau_{value}"

    def _condition_loss(
        self,
        generated_features: dict[str, Tensor],
        positive_features: dict[str, Tensor],
        positive_modes: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        groups = []
        skipped_groups = 0
        for name, generated_branch in generated_features.items():
            positive_branch = positive_features[name]
            if generated_branch.ndim < 3 or positive_branch.ndim < 2:
                raise ValueError(f"Invalid feature ranks in branch {name}")
            if generated_branch.shape[1:] != positive_branch.shape[1:]:
                raise ValueError(f"Generated/positive shape mismatch in {name}")
            for group_index in range(generated_branch.shape[1]):
                generated = generated_branch[:, group_index].reshape(
                    len(generated_branch), -1
                )
                positive = positive_branch[:, group_index].reshape(
                    len(positive_branch), -1
                )
                with torch.no_grad():
                    scale = feature_scale(generated.detach(), positive.detach())
                if not torch.isfinite(scale) or scale <= self.config.feature_scale_floor:
                    skipped_groups += 1
                    continue
                groups.append(
                    {
                        "name": f"{name}:{group_index}",
                        "family": _branch_family(name),
                        "generated": generated / scale,
                        "positive": positive / scale,
                        "channels": generated.shape[1],
                        "feature_scale": scale.detach(),
                    }
                )
        if not groups:
            raise FloatingPointError("All selected feature groups are degenerate")

        # Equivalent to concatenating equal-RMS group coordinates. The kernel
        # temperature is therefore the dimensionless tau itself.
        joint_group_count = len(groups)
        joint_generated = torch.cat(
            [group["generated"].detach() / group["channels"] ** 0.5 for group in groups],
            dim=1,
        ) / joint_group_count**0.5
        joint_positive = torch.cat(
            [group["positive"].detach() / group["channels"] ** 0.5 for group in groups],
            dim=1,
        ) / joint_group_count**0.5

        fields_by_group: list[list[Tensor]] = [[] for _ in groups]
        raw_scales = []
        raw_losses_by_temperature: list[list[Tensor]] = [
            [] for _ in self.config.temperatures
        ]
        active_temperature_count = 0
        mode_scores = []
        with torch.no_grad():
            for temperature_index, ratio in enumerate(self.config.temperatures):
                positive_weight, negative_weight = bidirectional_weights(
                    joint_generated,
                    joint_positive,
                    float(ratio),
                )
                active_this_temperature = False
                for group_index, group in enumerate(groups):
                    field = (
                        positive_weight @ group["positive"].detach()
                        - negative_weight @ group["generated"].detach()
                    )
                    drift_scale = (
                        field.square().sum(dim=1).mean() / group["channels"]
                    ).sqrt()
                    raw_scales.append(drift_scale)
                    if torch.isfinite(drift_scale):
                        # This is the paper's pre-normalization ||V||^2 loss.
                        raw_losses_by_temperature[temperature_index].append(
                            drift_scale.square()
                        )
                    if torch.isfinite(drift_scale) and \
                            drift_scale > self.config.drift_scale_floor:
                        fields_by_group[group_index].append(field / drift_scale)
                        active_this_temperature = True
                if not active_this_temperature:
                    continue
                active_temperature_count += 1
                positive_mass = positive_weight.sum(dim=1)
                valid = torch.isfinite(positive_mass) & (positive_mass > 0)
                if not valid.any():
                    continue
                per_positive = positive_weight[valid] / positive_mass[valid, None]
                unique_modes = torch.unique(positive_modes, sorted=True)
                mode_scores.append(
                    torch.stack(
                        [per_positive[:, positive_modes == mode].sum(dim=1)
                         for mode in unique_modes],
                        dim=1,
                    ).mean(dim=0)
                )

        group_losses = []
        component_group_losses: dict[str, list[Tensor]] = {
            "adsorbate": [],
            "surface": [],
        }
        normalized_drifts = []
        active_groups = 0
        for group, fields in zip(groups, fields_by_group):
            if not fields:
                skipped_groups += 1
                continue
            active_groups += 1
            total_field = torch.stack(fields).sum(dim=0)
            target = (group["generated"].detach() + total_field).detach()
            group_loss = (group["generated"] - target).square().mean()
            group_losses.append(group_loss)
            component_group_losses[group["family"]].append(group_loss)
            normalized_drifts.append(
                (total_field.square().sum(dim=1).mean() / group["channels"]).sqrt()
            )
        if not group_losses or not mode_scores:
            raise FloatingPointError("The shared joint kernel produced no active drift")
        mean_mode_score = torch.stack(mode_scores).mean(dim=0)
        relative_minimum = mean_mode_score.min() * len(mean_mode_score)
        temperature_raw_losses = []
        for values in raw_losses_by_temperature:
            if not values:
                raise FloatingPointError("A temperature produced no finite raw drift loss")
            temperature_raw_losses.append(torch.stack(values).mean())
        metrics = {
            "active_feature_groups": torch.tensor(active_groups, device=joint_generated.device),
            "skipped_feature_groups": torch.tensor(skipped_groups, device=joint_generated.device),
            "joint_kernel_groups": torch.tensor(joint_group_count, device=joint_generated.device),
            "active_joint_temperatures": torch.tensor(
                active_temperature_count, device=joint_generated.device
            ),
            "mean_feature_scale": torch.stack(
                [group["feature_scale"] for group in groups]
            ).mean(),
            "mean_raw_drift_rms": torch.stack(raw_scales).mean(),
            # Mean over feature groups and temperatures, matching the paper's
            # loss_R diagnostic before drift normalization.
            "raw_drift_loss": torch.stack(temperature_raw_losses).mean(),
            "mean_normalized_drift_rms": torch.stack(normalized_drifts).mean(),
            "minimum_mode_attraction_relative_uniform": relative_minimum,
        }
        metrics.update({
            self._temperature_metric_name(float(temperature)): raw_loss
            for temperature, raw_loss in zip(
                self.config.temperatures, temperature_raw_losses
            )
        })
        missing = [
            name for name, values in component_group_losses.items() if not values
        ]
        if missing:
            raise FloatingPointError(
                f"No active Drifting feature groups for components: {missing}"
            )
        component_losses = {
            name: torch.stack(values).sum()
            for name, values in component_group_losses.items()
        }
        return torch.stack(group_losses).sum(), component_losses, metrics

    def forward_with_components(
        self,
        generated_features: dict[str, Tensor],
        positive_features: dict[str, Tensor],
        positive_condition: Tensor,
        positive_modes: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        if generated_features.keys() != positive_features.keys():
            raise ValueError("Generated and positive feature branches differ")
        first = next(iter(generated_features.values()))
        if first.ndim < 4:
            raise ValueError("Generated features must have shape [B,G,Q,...]")
        batch_size, generated_count = first.shape[:2]
        if generated_count < 2:
            raise ValueError("At least two generated samples per condition are required")
        if positive_condition.shape != positive_modes.shape:
            raise ValueError("Positive condition and mode arrays must have equal shape")

        condition_losses = []
        condition_component_losses = []
        condition_metrics = []
        for condition in range(batch_size):
            selected = positive_condition == condition
            if not selected.any():
                raise ValueError(f"Condition {condition} has no positive samples")
            for name, generated_branch in generated_features.items():
                if generated_branch.shape[0] != batch_size:
                    raise ValueError(f"Branch {name} has inconsistent batch size")
            condition_loss, component_losses, metrics = self._condition_loss(
                {name: values[condition] for name, values in generated_features.items()},
                {name: values[selected] for name, values in positive_features.items()},
                positive_modes[selected],
            )
            condition_losses.append(condition_loss)
            condition_component_losses.append(component_losses)
            condition_metrics.append(metrics)

        loss = torch.stack(condition_losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite Drifting loss")
        component_losses = {
            name: torch.stack(
                [values[name] for values in condition_component_losses]
            ).mean()
            for name in ("adsorbate", "surface")
        }
        device = loss.device
        metrics = {
            "loss": loss.detach(),
            "adsorbate_feature_loss": component_losses["adsorbate"].detach(),
            "surface_feature_loss": component_losses["surface"].detach(),
            "active_feature_groups": torch.stack(
                [values["active_feature_groups"] for values in condition_metrics]
            ).sum(),
            "skipped_feature_groups": torch.stack(
                [values["skipped_feature_groups"] for values in condition_metrics]
            ).sum(),
            "joint_kernel_groups": torch.stack(
                [values["joint_kernel_groups"] for values in condition_metrics]
            ).sum(),
            "active_joint_temperatures": torch.stack(
                [values["active_joint_temperatures"] for values in condition_metrics]
            ).sum(),
            "mean_feature_scale": torch.stack(
                [values["mean_feature_scale"] for values in condition_metrics]
            ).mean(),
            "mean_raw_drift_rms": torch.stack(
                [values["mean_raw_drift_rms"] for values in condition_metrics]
            ).mean(),
            "raw_drift_loss": torch.stack(
                [values["raw_drift_loss"] for values in condition_metrics]
            ).mean(),
            "mean_normalized_drift_rms": torch.stack(
                [values["mean_normalized_drift_rms"] for values in condition_metrics]
            ).mean(),
            "minimum_mode_attraction_relative_uniform": torch.stack(
                [values["minimum_mode_attraction_relative_uniform"]
                 for values in condition_metrics]
            ).min(),
        }
        metrics.update({
            key: torch.stack([values[key] for values in condition_metrics]).mean()
            for key in condition_metrics[0]
            if key.startswith("raw_drift_loss_tau_")
        })
        return loss, component_losses, metrics

    def forward(
        self,
        generated_features: dict[str, Tensor],
        positive_features: dict[str, Tensor],
        positive_condition: Tensor,
        positive_modes: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        loss, _, metrics = self.forward_with_components(
            generated_features,
            positive_features,
            positive_condition,
            positive_modes,
        )
        return loss, metrics


def build_drifting_loss(config: dict) -> FeatureDriftingLoss:
    return FeatureDriftingLoss(DriftingConfig.from_dict(config))
