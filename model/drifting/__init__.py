"""Feature-space Drifting and first-order coordinate-gradient balancing."""
from .coordinate_gradient_balancing import (
    CoordinateGradientBalancer,
    CoordinateGradientBalancingConfig,
    build_coordinate_gradient_balancer,
    rigid_gradient_components,
    value_preserving_coordinate_surrogate,
)
from .loss import DriftingConfig, FeatureDriftingLoss, build_drifting_loss

__all__ = [
    "CoordinateGradientBalancer",
    "CoordinateGradientBalancingConfig",
    "DriftingConfig",
    "FeatureDriftingLoss",
    "build_coordinate_gradient_balancer",
    "build_drifting_loss",
    "rigid_gradient_components",
    "value_preserving_coordinate_surrogate",
]
