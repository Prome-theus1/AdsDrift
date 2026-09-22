"""End-to-end one-pass generator plus frozen-feature Drifting objective."""

from __future__ import annotations

# Allow both 'python -m AdsDrift.model.model' and direct file execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    script_directory = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != script_directory]
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import Tensor, nn

from AdsDrift.model.generator import (
    EquiformerV3AdsorptionGenerator,
    ROLE_ADSORBATE,
    ROLE_FIXED,
    ROLE_SURFACE,
)
from AdsDrift.model.drifting.loss import FeatureDriftingLoss
from AdsDrift.model.drifting.mace_features import FrozenMACEInterfaceFeatures
from AdsDrift.model.drifting.coordinate_gradient_balancing import (
    CoordinateGradientBalancer,
    value_preserving_coordinate_surrogate,
)


class AdsorptionDriftingObjective(nn.Module):
    def __init__(
        self,
        generator: EquiformerV3AdsorptionGenerator,
        feature_encoder: FrozenMACEInterfaceFeatures,
        drifting_loss: FeatureDriftingLoss,
        coordinate_gradient_balancer: CoordinateGradientBalancer,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.feature_encoder = feature_encoder
        self.drifting_loss = drifting_loss
        self.coordinate_gradient_balancer = coordinate_gradient_balancer

    def forward(self, batch: dict) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        generated = self.generator(
            atomic_numbers=batch["atomic_numbers"],
            roles=batch["roles"],
            atom_mask=batch["atom_mask"],
            cell=batch["cell"],
            r0_positions=batch["r0_positions"],
            condition=batch["condition"],
        )
        batch_size, generated_count, atoms = generated["positions"].shape[:3]
        generated_positions = generated["positions"]
        # Isolate the MACE graph from the generator graph.  MACE is traversed
        # exactly once to obtain dL/dR; the balanced detached field is then
        # passed through a linear surrogate into the generator.
        feature_positions = generated_positions.detach().requires_grad_(True)
        flat_positions = feature_positions.reshape(
            batch_size * generated_count, atoms, 3
        )
        flat_numbers = batch["atomic_numbers"].repeat_interleave(generated_count, dim=0)
        flat_roles = batch["roles"].repeat_interleave(generated_count, dim=0)
        flat_mask = batch["atom_mask"].repeat_interleave(generated_count, dim=0)
        flat_cell = batch["cell"].repeat_interleave(generated_count, dim=0)
        flat_features = self.feature_encoder(
            flat_numbers, flat_positions, flat_cell, flat_mask, flat_roles
        )
        generated_features = {
            name: values.reshape(batch_size, generated_count, *values.shape[1:])
            for name, values in flat_features.items()
        }
        positive_features = {
            "scalar_ads": batch["positive_scalar_ads"],
            "message_l1_ads": batch["positive_message_l1_ads"],
            "scalar_movable": batch["positive_scalar_movable"],
            "message_l1_movable": batch["positive_message_l1_movable"],
        }
        feature_loss, metrics = self.drifting_loss(
            generated_features,
            positive_features,
            batch["positive_condition"],
            batch["positive_modes"],
        )
        coordinate_gradient = torch.autograd.grad(
            feature_loss,
            feature_positions,
            retain_graph=False,
            create_graph=False,
        )[0].detach()
        balanced_gradient, balancing_metrics = self.coordinate_gradient_balancer(
            generated_positions.detach(),
            coordinate_gradient,
            batch["roles"],
            batch["atom_mask"],
        )
        loss = value_preserving_coordinate_surrogate(
            feature_loss,
            generated_positions,
            balanced_gradient,
        )
        metrics.update(balancing_metrics)
        metrics["coordinate_routing_enabled"] = loss.detach().new_ones(())
        fixed = (batch["roles"] == ROLE_FIXED) & batch["atom_mask"]
        fixed_error = (
            generated["positions"] - batch["r0_positions"]
        ).abs().masked_select(fixed[:, None, :, None]).max()
        surface_mask = (
            (batch["roles"] == ROLE_SURFACE) & batch["atom_mask"]
        )[:, None, :, None]
        adsorbate_mask = (
            (batch["roles"] == ROLE_ADSORBATE) & batch["atom_mask"]
        )[:, None, :, None]
        surface_rms = (
            generated["surface_displacement"].square().sum()
            / surface_mask.expand_as(generated["surface_displacement"]).sum().clamp_min(1)
        ).sqrt()
        ads_internal_rms = (
            generated["ads_internal_displacement"].square().sum()
            / adsorbate_mask.expand_as(generated["ads_internal_displacement"]).sum().clamp_min(1)
        ).sqrt()
        metrics.update(
            fixed_coordinate_max_error_A=fixed_error.detach(),
            surface_displacement_rms_A=surface_rms.detach(),
            ads_center_displacement_rms_A=generated["ads_center_displacement"]
            .square().mean().sqrt().detach(),
            ads_internal_displacement_rms_A=ads_internal_rms.detach(),
        )
        return loss, metrics, generated


def build_model(
    generator_config: dict,
    drifting_config: dict,
    coordinate_gradient_balancing_config: dict | None = None,
) -> AdsorptionDriftingObjective:
    """Assemble a generator and loss with the fixed, package-local MACE encoder."""
    from AdsDrift.model.generator import build_generator
    from AdsDrift.model.drifting.mace_features import build_mace_features
    from AdsDrift.model.drifting import build_drifting_loss
    from AdsDrift.model.drifting import build_coordinate_gradient_balancer

    return AdsorptionDriftingObjective(
        build_generator(generator_config),
        build_mace_features(),
        build_drifting_loss(drifting_config),
        build_coordinate_gradient_balancer(coordinate_gradient_balancing_config),
    )


def main(argv: list[str] | None = None) -> None:
    """Dispatch training or inference, forwarding options to the existing runners."""
    import argparse
    import importlib
    import sys

    runners = {
        "train": "AdsDrift.model.utils.train",
        "train-distributed": "AdsDrift.model.utils.multi_gpu_training",
        "sample": "AdsDrift.model.utils.generate_structures",
    }
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="AdsDrift: conditional one-pass adsorption structure generation",
        usage="%(prog)s {train,train-distributed,sample} [options]",
    )
    parser.add_argument("command", choices=tuple(runners))
    # Parse only the command, so '<command> --help' reaches that runner's parser.
    command = parser.parse_args(arguments[:1]).command
    old_argv = sys.argv
    try:
        sys.argv = [f"{old_argv[0]} {command}", *arguments[1:]]
        importlib.import_module(runners[command]).main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
