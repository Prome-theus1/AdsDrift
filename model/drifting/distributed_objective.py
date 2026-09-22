"""Exact one-condition Drifting objective split over candidate structures."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.nn.functional import all_gather

from AdsDrift.model.generator import ROLE_ADSORBATE, ROLE_FIXED, ROLE_SURFACE
from AdsDrift.model.drifting.loss import FeatureDriftingLoss
from AdsDrift.model.drifting.mace_features import FrozenMACEInterfaceFeatures


class DistributedCandidateDriftingObjective(nn.Module):
    """Split one condition's R0 bank across ranks without changing its loss.

    Each rank maps a disjoint contiguous subset of the 100 R0 candidates through
    the same DDP-synchronized generator and its own frozen MACE replica. The
    differentiable all-gather restores all 100 feature vectors before every rank
    evaluates the identical conditional Drifting loss.
    """

    def __init__(
        self,
        generator: nn.Module,
        feature_encoder: FrozenMACEInterfaceFeatures,
        drifting_loss: FeatureDriftingLoss,
    ) -> None:
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("Distributed process group must be initialized")
        self.generator = generator
        self.feature_encoder = feature_encoder
        self.drifting_loss = drifting_loss
        self.world_size = dist.get_world_size()

    @staticmethod
    def _positive_features(batch: dict) -> dict[str, Tensor]:
        return {
            "scalar_ads": batch["positive_scalar_ads"],
            "message_l1_ads": batch["positive_message_l1_ads"],
            "scalar_movable": batch["positive_scalar_movable"],
            "message_l1_movable": batch["positive_message_l1_movable"],
        }

    def _global_generated_features(
        self, local_features: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        global_features = {}
        for name, values in local_features.items():
            pieces = all_gather(values.contiguous())
            global_features[name] = torch.cat(tuple(pieces), dim=0).unsqueeze(0)
        return global_features

    @staticmethod
    def _all_reduce_sum(value: Tensor) -> Tensor:
        value = value.clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def forward(self, batch: dict) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        if len(batch["r0_positions"]) != 1:
            raise ValueError("Distributed test_18 training requires one condition per batch")
        generated = self.generator(
            atomic_numbers=batch["atomic_numbers"],
            roles=batch["roles"],
            atom_mask=batch["atom_mask"],
            cell=batch["cell"],
            r0_positions=batch["r0_positions"],
            condition=batch["condition"],
        )
        _, local_count, atoms = generated["positions"].shape[:3]
        flat_positions = generated["positions"].reshape(local_count, atoms, 3)
        flat_numbers = batch["atomic_numbers"].repeat_interleave(local_count, dim=0)
        flat_cell = batch["cell"].repeat_interleave(local_count, dim=0)
        flat_mask = batch["atom_mask"].repeat_interleave(local_count, dim=0)
        flat_roles = batch["roles"].repeat_interleave(local_count, dim=0)
        local_features = self.feature_encoder(
            flat_numbers,
            flat_positions,
            flat_cell,
            flat_mask,
            flat_roles,
        )
        generated_features = self._global_generated_features(local_features)
        global_count = next(iter(generated_features.values())).shape[1]
        if global_count != local_count * self.world_size:
            raise RuntimeError("Differentiable candidate gather has the wrong size")
        loss, metrics = self.drifting_loss(
            generated_features,
            self._positive_features(batch),
            batch["positive_condition"],
            batch["positive_modes"],
        )
        metrics["coordinate_routing_enabled"] = loss.detach().new_zeros(())

        fixed = (batch["roles"] == ROLE_FIXED) & batch["atom_mask"]
        local_fixed_error = (
            generated["positions"] - batch["r0_positions"]
        ).abs().masked_select(fixed[:, None, :, None]).max().detach()
        dist.all_reduce(local_fixed_error, op=dist.ReduceOp.MAX)

        surface_mask = (
            (batch["roles"] == ROLE_SURFACE) & batch["atom_mask"]
        )[:, None, :, None]
        adsorbate_mask = (
            (batch["roles"] == ROLE_ADSORBATE) & batch["atom_mask"]
        )[:, None, :, None]
        surface_sum = self._all_reduce_sum(
            generated["surface_displacement"].detach().square().sum()
        )
        surface_count = self._all_reduce_sum(
            surface_mask.expand_as(generated["surface_displacement"]).sum()
        )
        internal_sum = self._all_reduce_sum(
            generated["ads_internal_displacement"].detach().square().sum()
        )
        internal_count = self._all_reduce_sum(
            adsorbate_mask.expand_as(generated["ads_internal_displacement"]).sum()
        )
        center_sum = self._all_reduce_sum(
            generated["ads_center_displacement"].detach().square().sum()
        )
        center_count = self._all_reduce_sum(
            torch.tensor(
                generated["ads_center_displacement"].numel(),
                device=center_sum.device,
                dtype=center_sum.dtype,
            )
        )
        metrics.update(
            fixed_coordinate_max_error_A=local_fixed_error,
            surface_displacement_rms_A=(
                surface_sum / surface_count.clamp_min(1)
            ).sqrt(),
            ads_center_displacement_rms_A=(
                center_sum / center_count.clamp_min(1)
            ).sqrt(),
            ads_internal_displacement_rms_A=(
                internal_sum / internal_count.clamp_min(1)
            ).sqrt(),
        )
        return loss, metrics, generated
