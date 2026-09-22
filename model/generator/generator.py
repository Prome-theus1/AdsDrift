"""Local EquiformerV3 generator for fixed-cell adsorption systems.

The model is a one-pass map from a bank of random initial structures to a bank
of generated structures.  It uses three parameter-distinct EquiformerV3
attention paths at every layer:

1. full slab -> full slab self-attention;
2. adsorbate -> adsorbate self-attention;
3. bidirectional slab <-> adsorbate cross-attention.

The two self-attention paths update node features first.  Cross-attention then
reads the normalized/conditioned updated features, followed by the FFN residual.
The full-graph geometric input embedding is unchanged from test_16.

All valid nodes, including fixed slab atoms, update their hidden features in
attention and FFN residuals.  Fixed slab atoms participate as queries, keys and
values, but remain unchanged in Cartesian space.  Only movable surface and
adsorbate atoms receive coordinate residuals.
All coordinate residuals are read from l=1 features, so the learned displacement
field is SE(3)-equivariant by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from AdsDrift.model.condition import CONDITION_BATCH_KEYS, FactorizedConditionEncoder

from .core.attention import (
    EdgeDegreeEmbedding,
    EquivariantGraphAttention,
    FeedForwardNetwork,
)
from .core.geometry import SO3Linear, SO3Rotation, init_edge_rot_mat
from .core.layers import (
    GaussianSmearing,
    PolynomialEnvelope,
    RadialFunction,
    get_normalization_layer,
)


ROLE_PADDING = 0
ROLE_FIXED = 1
ROLE_SURFACE = 2
ROLE_ADSORBATE = 3


def _masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weight = mask.to(values.dtype)
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    return (values * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def _zero_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Remove the duplicate translation degree of freedom from ads residuals."""
    return (values - _masked_mean(values, mask, dim=1).unsqueeze(1)) * mask[..., None]


@dataclass(frozen=True)
class EquiformerV3GeneratorConfig:
    """Hyperparameters for the local three-branch equivariant generator."""

    num_layers: int = 4
    num_channels: int = 128
    attn_hidden_channels: int = 64
    num_heads: int = 8
    attn_alpha_channels: int = 32
    attn_value_channels: int = 16
    ffn_hidden_channels: int = 256
    lmax: int = 3
    mmax: int = 2
    attn_grid_resolution: tuple[int, int] = (12, 8)
    ffn_grid_resolution: tuple[int, int] = (12, 12)
    num_radial_basis: int = 64
    edge_channels: int = 128
    slab_radius_A: float = 6.0
    adsorbate_radius_A: float = 4.5
    cross_radius_A: float = 6.0
    max_neighbors_per_relation: int = 24
    periodic_image_range: int = 1
    pbc_axes: tuple[bool, bool, bool] = (True, True, False)
    norm_type: str = "merge_layer_norm"
    attention_activation: str = "sep-merge_gates2_swiglu"
    ffn_activation: str = "sep-merge_gates2_swiglu"
    use_grid_mlp: bool = True
    use_add_merge: bool = True
    attention_weight_dropout: float = 0.0
    ffn_dropout: float = 0.0
    activation_checkpointing: bool = True
    surface_output_scale_A: float = 0.5
    ads_center_output_scale_A: float = 3.0
    ads_internal_output_scale_A: float = 1.0
    max_atomic_number: int = 118
    condition_hidden_channels: int = 128
    condition_message_layers: int = 2
    condition_num_radial_basis: int = 32
    condition_max_distance_A: float = 8.0
    condition_modulation_scale: float = 0.25

    @classmethod
    def from_dict(cls, values: dict) -> "EquiformerV3GeneratorConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(values) - known
        if unknown:
            raise ValueError(
                f"Unknown EquiformerV3 generator settings: {sorted(unknown)}"
            )
        converted = dict(values)
        for name in ("attn_grid_resolution", "ffn_grid_resolution", "pbc_axes"):
            if name in converted:
                converted[name] = tuple(converted[name])
        config = cls(**converted)
        config.validate()
        return config

    def validate(self) -> None:
        integer_positive = {
            "num_layers": self.num_layers,
            "num_channels": self.num_channels,
            "attn_hidden_channels": self.attn_hidden_channels,
            "num_heads": self.num_heads,
            "attn_alpha_channels": self.attn_alpha_channels,
            "attn_value_channels": self.attn_value_channels,
            "ffn_hidden_channels": self.ffn_hidden_channels,
            "num_radial_basis": self.num_radial_basis,
            "edge_channels": self.edge_channels,
            "max_neighbors_per_relation": self.max_neighbors_per_relation,
            "condition_hidden_channels": self.condition_hidden_channels,
            "condition_message_layers": self.condition_message_layers,
            "condition_num_radial_basis": self.condition_num_radial_basis,
        }
        if any(value < 1 for value in integer_positive.values()):
            raise ValueError(f"Positive integer settings required: {integer_positive}")
        if not (1 <= self.lmax <= 8 and 0 <= self.mmax <= self.lmax):
            raise ValueError("Need 1 <= lmax <= 8 and 0 <= mmax <= lmax")
        if self.num_heads * self.attn_value_channels != self.num_channels:
            raise ValueError(
                "num_heads * attn_value_channels must equal num_channels"
            )
        if len(self.attn_grid_resolution) != 2 or len(self.ffn_grid_resolution) != 2:
            raise ValueError("S2 grid resolutions need [latitude, longitude]")
        if len(self.pbc_axes) != 3 or not any(self.pbc_axes):
            raise ValueError("pbc_axes must contain three booleans and one periodic axis")
        if self.periodic_image_range < 0:
            raise ValueError("periodic_image_range must be nonnegative")
        if self.condition_max_distance_A <= 0:
            raise ValueError("condition_max_distance_A must be positive")
        if not (0 < self.condition_modulation_scale <= 1):
            raise ValueError("condition_modulation_scale must lie in (0,1]")
        positive_scales = (
            self.slab_radius_A,
            self.adsorbate_radius_A,
            self.cross_radius_A,
            self.surface_output_scale_A,
            self.ads_center_output_scale_A,
            self.ads_internal_output_scale_A,
        )
        if min(positive_scales) <= 0:
            raise ValueError("Cutoffs and coordinate output scales must be positive")


@dataclass
class _MasterNeighborTable:
    """Nearest periodic source images for every target node."""

    source_local: Tensor  # [Q, N, K]
    source_global: Tensor  # [Q, N, K]
    target_global: Tensor  # [Q, N, K]
    vector: Tensor  # [Q, N, K, 3], source image minus target
    distance: Tensor  # [Q, N, K]
    valid: Tensor  # [Q, N, K]


@dataclass
class _RelationGraph:
    edge_index: Tensor  # [2, E], source -> target
    edge_vector: Tensor  # [E, 3]
    edge_distance: Tensor  # [E]

    @property
    def edge_count(self) -> int:
        return int(self.edge_index.shape[1])


@dataclass
class _PreparedRelation:
    edge_index: Tensor
    distance_features: Tensor
    envelope: Tensor

    @property
    def edge_count(self) -> int:
        return int(self.edge_index.shape[1])


class _TriBranchEquiformerV3Block(nn.Module):
    """Sequential self-attention, cross-attention, and FFN residual updates."""

    def __init__(
        self,
        config: EquiformerV3GeneratorConfig,
        rotations: dict[str, SO3Rotation],
        edge_channels_list: list[int],
    ) -> None:
        super().__init__()
        self.norm_attention = get_normalization_layer(
            config.norm_type, config.lmax, config.num_channels
        )

        attention_kwargs = dict(
            num_in_channels=config.num_channels,
            num_hidden_channels=config.attn_hidden_channels,
            num_heads=config.num_heads,
            attn_alpha_channels=config.attn_alpha_channels,
            attn_value_channels=config.attn_value_channels,
            num_out_channels=config.num_channels,
            lmax=config.lmax,
            mmax=config.mmax,
            grid_resolution_list=list(config.attn_grid_resolution),
            max_num_elements=config.max_atomic_number + 1,
            edge_channels_list=edge_channels_list,
            use_atom_edge_embedding=True,
            activation=config.attention_activation,
            use_attn_renorm=True,
            use_add_merge=config.use_add_merge,
            use_rad_l_parametrization=True,
            softcap=None,
            eps=1.0e-16,
            alpha_drop=0.0,
            attn_mask_rate=0.0,
            attn_weights_drop=config.attention_weight_dropout,
            value_drop=0.0,
        )
        self.slab_self_attention = EquivariantGraphAttention(
            so3_rotation=rotations["slab"], **attention_kwargs
        )
        self.adsorbate_self_attention = EquivariantGraphAttention(
            so3_rotation=rotations["adsorbate"], **attention_kwargs
        )
        self.interface_cross_attention = EquivariantGraphAttention(
            so3_rotation=rotations["cross"], **attention_kwargs
        )
        self.norm_ffn = get_normalization_layer(
            config.norm_type, config.lmax, config.num_channels
        )
        self.ffn = FeedForwardNetwork(
            num_in_channels=config.num_channels,
            num_hidden_channels=config.ffn_hidden_channels,
            num_out_channels=config.num_channels,
            lmax=config.lmax,
            mmax=config.mmax,
            grid_resolution_list=list(config.ffn_grid_resolution),
            activation=config.ffn_activation,
            use_grid_mlp=config.use_grid_mlp,
            dropout=config.ffn_dropout,
        )

    @staticmethod
    def _attention_message(
        attention: EquivariantGraphAttention,
        values: Tensor,
        atomic_numbers: Tensor,
        graph: _PreparedRelation,
    ) -> Tensor:
        if graph.edge_count == 0:
            return torch.zeros_like(values)
        source = graph.edge_index[0]
        target = graph.edge_index[1]
        return attention(
            values,
            atomic_numbers[source],
            atomic_numbers[target],
            graph.distance_features,
            graph.edge_index,
            graph.envelope,
        )

    def forward(
        self,
        values: Tensor,
        atomic_numbers: Tensor,
        feature_update_mask: Tensor,
        slab_graph: _PreparedRelation,
        adsorbate_graph: _PreparedRelation,
        cross_graph: _PreparedRelation,
        condition_scale: Tensor,
        condition_shift: Tensor,
        condition_strength: float,
    ) -> Tensor:
        normalized = self.norm_attention(values)
        conditioned_scalar = normalized[:, 0, :] * (
            1.0 + condition_strength * torch.tanh(condition_scale)
        ) + condition_strength * condition_shift
        normalized = torch.cat(
            (conditioned_scalar[:, None, :], normalized[:, 1:, :]), dim=1
        )
        slab_message = self._attention_message(
            self.slab_self_attention, normalized, atomic_numbers, slab_graph
        )
        adsorbate_message = self._attention_message(
            self.adsorbate_self_attention,
            normalized,
            atomic_numbers,
            adsorbate_graph,
        )
        # Preserve the baseline 1/sqrt(2) contribution scale for both paths.
        # Slab and adsorbate self-attention have disjoint target sets.
        mask = feature_update_mask[:, None, None].to(values.dtype)
        values = values + mask * (slab_message + adsorbate_message) / math.sqrt(2.0)

        # Cross-attention consumes the updated self-attention state.  Reuse the
        # existing normalization and FiLM parameters to keep parameter counts
        # and random initialization identical to the parallel baseline.
        normalized_cross = self.norm_attention(values)
        conditioned_cross_scalar = normalized_cross[:, 0, :] * (
            1.0 + condition_strength * torch.tanh(condition_scale)
        ) + condition_strength * condition_shift
        normalized_cross = torch.cat(
            (conditioned_cross_scalar[:, None, :], normalized_cross[:, 1:, :]), dim=1
        )
        cross_message = self._attention_message(
            self.interface_cross_attention, normalized_cross, atomic_numbers, cross_graph
        )
        values = values + mask * cross_message / math.sqrt(2.0)
        normalized_ffn = self.norm_ffn(values)
        conditioned_ffn_scalar = normalized_ffn[:, 0, :] * (
            1.0 + condition_strength * torch.tanh(condition_scale)
        ) + condition_strength * condition_shift
        normalized_ffn = torch.cat(
            (conditioned_ffn_scalar[:, None, :], normalized_ffn[:, 1:, :]), dim=1
        )
        values = values + mask * self.ffn(normalized_ffn)
        return values


class EquiformerV3AdsorptionGenerator(nn.Module):
    """Three-branch local SE(3)-equivariant adsorption structure generator."""

    def __init__(self, config: EquiformerV3GeneratorConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.max_radius_A = max(
            config.slab_radius_A,
            config.adsorbate_radius_A,
            config.cross_radius_A,
        )
        # Each movable/adsorbate target has at most two source relation classes.
        self.master_max_neighbors = 2 * config.max_neighbors_per_relation

        shifts_per_axis = [
            range(-config.periodic_image_range, config.periodic_image_range + 1)
            if periodic
            else (0,)
            for periodic in config.pbc_axes
        ]
        integer_shifts = torch.tensor(
            list(itertools.product(*shifts_per_axis)), dtype=torch.get_default_dtype()
        )
        self.register_buffer("integer_cell_shifts", integer_shifts, persistent=False)
        zero_shift = torch.nonzero(
            (integer_shifts == 0).all(dim=1), as_tuple=False
        ).flatten()
        if len(zero_shift) != 1:
            raise RuntimeError("Periodic shift table must contain one zero image")
        self.zero_shift_index = int(zero_shift.item())

        self.element_embedding = nn.Embedding(
            config.max_atomic_number + 1,
            config.num_channels,
            padding_idx=0,
        )
        self.role_embedding = nn.Embedding(4, config.num_channels, padding_idx=0)
        self.condition_encoder = FactorizedConditionEncoder(
            num_channels=config.num_channels,
            num_layers=config.num_layers,
            hidden_channels=config.condition_hidden_channels,
            primitive_message_layers=config.condition_message_layers,
            num_radial_basis=config.condition_num_radial_basis,
            max_distance_A=config.condition_max_distance_A,
            max_atomic_number=config.max_atomic_number,
        )
        self.distance_expansion = GaussianSmearing(
            0.0,
            self.max_radius_A,
            config.num_radial_basis,
            2.0,
        )
        edge_channels_list = [config.num_radial_basis, config.edge_channels, config.edge_channels]

        self.rotations = nn.ModuleDict(
            {
                name: SO3Rotation(config.lmax, config.mmax)
                for name in ("input", "slab", "adsorbate", "cross")
            }
        )
        self.envelopes = nn.ModuleDict(
            {
                "input": PolynomialEnvelope(self.max_radius_A, exponent=5),
                "slab": PolynomialEnvelope(config.slab_radius_A, exponent=5),
                "adsorbate": PolynomialEnvelope(
                    config.adsorbate_radius_A, exponent=5
                ),
                "cross": PolynomialEnvelope(config.cross_radius_A, exponent=5),
            }
        )
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            num_channels=config.num_channels,
            lmax=config.lmax,
            mmax=config.mmax,
            so3_rotation=self.rotations["input"],
            max_num_elements=config.max_atomic_number + 1,
            edge_channels_list=edge_channels_list,
            use_atom_edge_embedding=True,
            rescale_factor=math.sqrt(float(config.max_neighbors_per_relation)),
        )
        branch_rotations = {
            "slab": self.rotations["slab"],
            "adsorbate": self.rotations["adsorbate"],
            "cross": self.rotations["cross"],
        }
        self.blocks = nn.ModuleList(
            [
                _TriBranchEquiformerV3Block(
                    config, branch_rotations, edge_channels_list
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_norm = get_normalization_layer(
            config.norm_type, config.lmax, config.num_channels
        )
        # Applying the same channel projection to all three l=1 components is
        # equivariant; no Cartesian component mixing is allowed here.
        self.surface_head = nn.Linear(config.num_channels, 1, bias=False)
        self.ads_center_head = nn.Linear(config.num_channels, 1, bias=False)
        self.ads_internal_head = nn.Linear(config.num_channels, 1, bias=False)

        self.apply(self._initialize_module)
        for head in (self.surface_head, self.ads_center_head, self.ads_internal_head):
            nn.init.zeros_(head.weight)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, SO3Linear)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RadialFunction):
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    bound = 1.0 / math.sqrt(child.in_features)
                    nn.init.uniform_(child.weight, -bound, bound)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate(
        self,
        atomic_numbers: Tensor,
        roles: Tensor,
        atom_mask: Tensor,
        cell: Tensor,
        r0_positions: Tensor,
        condition: dict[str, Tensor],
    ) -> tuple[int, int, int]:
        if atomic_numbers.ndim != 2 or roles.shape != atomic_numbers.shape:
            raise ValueError("atomic_numbers and roles must both have shape [B,N]")
        if atom_mask.shape != atomic_numbers.shape:
            raise ValueError("atom_mask must have shape [B,N]")
        batch, atoms = atomic_numbers.shape
        if cell.shape != (batch, 3, 3):
            raise ValueError("cell must have shape [B,3,3]")
        if r0_positions.ndim != 4 or r0_positions.shape[:1] != (batch,):
            raise ValueError("r0_positions must have shape [B,G,N,3]")
        generated = r0_positions.shape[1]
        if r0_positions.shape[2:] != (atoms, 3) or generated < 1:
            raise ValueError("r0_positions has inconsistent atom dimensions")
        valid_roles = roles[atom_mask]
        allowed = roles.new_tensor([ROLE_FIXED, ROLE_SURFACE, ROLE_ADSORBATE])
        if not torch.isin(valid_roles, allowed).all():
            raise ValueError("Valid atoms need fixed, surface, or adsorbate roles")
        if not ((roles == ROLE_SURFACE) & atom_mask).any(dim=1).all():
            raise ValueError("Every condition needs a movable surface atom")
        if not ((roles == ROLE_ADSORBATE) & atom_mask).any(dim=1).all():
            raise ValueError("Every condition needs an adsorbate atom")
        if not torch.isfinite(cell).all() or not torch.isfinite(r0_positions).all():
            raise ValueError("Non-finite cell or coordinates")
        if (torch.linalg.det(cell).abs() < 1.0e-8).any():
            raise ValueError("Cell matrices must be invertible")
        missing = sorted(set(CONDITION_BATCH_KEYS) - set(condition))
        if missing:
            raise ValueError(f"Missing factorized condition tensors: {missing}")
        wrong_batch = {
            name: tuple(condition[name].shape)
            for name in CONDITION_BATCH_KEYS
            if condition[name].ndim < 1 or condition[name].shape[0] != batch
        }
        if wrong_batch:
            raise ValueError(f"Condition tensors have the wrong batch dimension: {wrong_batch}")
        return batch, generated, atoms

    def _build_master_neighbors(
        self,
        positions: Tensor,
        cell: Tensor,
        atom_mask: Tensor,
    ) -> _MasterNeighborTable:
        """Find nearest periodic source images without per-atom Python loops."""
        structures, atoms = atom_mask.shape
        k = min(self.master_max_neighbors, atoms * len(self.integer_cell_shifts))
        inf = torch.tensor(float("inf"), device=positions.device, dtype=positions.dtype)
        best_distance_sq = positions.new_full((structures, atoms, 0), float("inf"))
        best_vector = positions.new_zeros((structures, atoms, 0, 3))
        best_source = torch.empty(
            (structures, atoms, 0), device=positions.device, dtype=torch.long
        )
        local_source = torch.arange(atoms, device=positions.device)
        local_source = local_source.view(1, 1, atoms).expand(structures, atoms, atoms)
        pair_valid = atom_mask[:, :, None] & atom_mask[:, None, :]
        cartesian_shifts = torch.einsum(
            "sd,qdc->qsc", self.integer_cell_shifts.to(cell.dtype), cell
        )

        for shift_index in range(len(self.integer_cell_shifts)):
            vector = (
                positions[:, None, :, :]
                - positions[:, :, None, :]
                + cartesian_shifts[:, shift_index, None, None, :]
            )
            distance_sq = vector.square().sum(dim=-1)
            current_valid = pair_valid & (distance_sq < self.max_radius_A**2)
            if shift_index == self.zero_shift_index:
                diagonal = torch.eye(atoms, device=positions.device, dtype=torch.bool)
                current_valid = current_valid & ~diagonal[None]
            distance_sq = torch.where(current_valid, distance_sq, inf)

            candidate_distance = torch.cat((best_distance_sq, distance_sq), dim=-1)
            candidate_vector = torch.cat((best_vector, vector), dim=2)
            candidate_source = torch.cat((best_source, local_source), dim=-1)
            keep = min(k, candidate_distance.shape[-1])
            best_distance_sq, selected = torch.topk(
                candidate_distance, keep, dim=-1, largest=False, sorted=True
            )
            best_vector = torch.gather(
                candidate_vector,
                2,
                selected[..., None].expand(-1, -1, -1, 3),
            )
            best_source = torch.gather(candidate_source, 2, selected)

        valid = torch.isfinite(best_distance_sq)
        structure_index = torch.arange(structures, device=positions.device)
        structure_index = structure_index[:, None, None]
        target_local = torch.arange(atoms, device=positions.device)[None, :, None]
        source_global = structure_index * atoms + best_source
        target_global = (structure_index * atoms + target_local).expand_as(best_source)
        return _MasterNeighborTable(
            source_local=best_source,
            source_global=source_global,
            target_global=target_global,
            vector=best_vector,
            distance=best_distance_sq.clamp_max(self.max_radius_A**2).sqrt(),
            valid=valid,
        )

    def _relation_from_master(
        self,
        master: _MasterNeighborTable,
        source_mask: Tensor,
        target_mask: Tensor,
        cutoff: float,
        exclude_same_node: bool = False,
    ) -> _RelationGraph:
        source_allowed = torch.gather(
            source_mask[:, None, :].expand(-1, source_mask.shape[1], -1),
            2,
            master.source_local,
        )
        relation = (
            master.valid
            & source_allowed
            & target_mask[:, :, None]
            & (master.distance < cutoff)
        )
        if exclude_same_node:
            relation = relation & (master.source_global != master.target_global)
        relation_rank = relation.long().cumsum(dim=-1)
        relation = relation & (
            relation_rank <= self.config.max_neighbors_per_relation
        )
        structure, target, slot = torch.nonzero(relation, as_tuple=True)
        edge_index = torch.stack(
            (
                master.source_global[structure, target, slot],
                master.target_global[structure, target, slot],
            ),
            dim=0,
        )
        return _RelationGraph(
            edge_index=edge_index,
            edge_vector=master.vector[structure, target, slot],
            edge_distance=master.distance[structure, target, slot],
        )

    @staticmethod
    def _merge_relations(*graphs: _RelationGraph) -> _RelationGraph:
        return _RelationGraph(
            edge_index=torch.cat([graph.edge_index for graph in graphs], dim=1),
            edge_vector=torch.cat([graph.edge_vector for graph in graphs], dim=0),
            edge_distance=torch.cat([graph.edge_distance for graph in graphs], dim=0),
        )

    def _prepare_relation(self, name: str, graph: _RelationGraph) -> _PreparedRelation:
        if graph.edge_count:
            self.rotations[name].set_wigner(init_edge_rot_mat(graph.edge_vector))
        return _PreparedRelation(
            edge_index=graph.edge_index,
            distance_features=self.distance_expansion(graph.edge_distance),
            envelope=self.envelopes[name](graph.edge_distance),
        )

    @staticmethod
    def _wrap_adsorbate_center(
        center: Tensor, relative: Tensor, cell: Tensor, ads_mask: Tensor
    ) -> Tensor:
        """Wrap only the molecular center in-plane; never split the molecule."""
        fractional = torch.bmm(center[:, None], torch.linalg.inv(cell)).squeeze(1)
        wrapped_fractional = torch.cat(
            [fractional[:, :2] - torch.floor(fractional[:, :2]), fractional[:, 2:]],
            dim=-1,
        )
        wrapped_center = torch.bmm(wrapped_fractional[:, None], cell).squeeze(1)
        return (wrapped_center[:, None] + relative) * ads_mask[..., None]

    def forward(
        self,
        atomic_numbers: Tensor,
        roles: Tensor,
        atom_mask: Tensor,
        cell: Tensor,
        r0_positions: Tensor,
        condition: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        batch, generated, atoms = self._validate(
            atomic_numbers, roles, atom_mask, cell, r0_positions, condition
        )
        initial_condition, layer_condition = self.condition_encoder(
            condition, atomic_numbers, roles, atom_mask, cell
        )
        structures = batch * generated
        flat_positions = r0_positions.reshape(structures, atoms, 3)
        flat_cell = cell.repeat_interleave(generated, dim=0)
        flat_numbers = atomic_numbers.repeat_interleave(generated, dim=0)
        flat_roles = roles.repeat_interleave(generated, dim=0)
        flat_mask = atom_mask.repeat_interleave(generated, dim=0).bool()
        slab_mask = flat_mask & (flat_roles != ROLE_ADSORBATE)
        ads_mask = flat_mask & (flat_roles == ROLE_ADSORBATE)
        surface_mask = flat_mask & (flat_roles == ROLE_SURFACE)
        # Feature updates include the fixed slab; coordinate heads below still
        # use surface_mask/ads_mask so fixed coordinates cannot move.
        feature_update_mask = flat_mask

        master = self._build_master_neighbors(flat_positions, flat_cell, flat_mask)
        input_graph = self._relation_from_master(
            master, flat_mask, flat_mask, self.max_radius_A
        )
        slab_graph = self._relation_from_master(
            master,
            slab_mask,
            slab_mask,
            self.config.slab_radius_A,
        )
        adsorbate_graph = self._relation_from_master(
            master,
            ads_mask,
            ads_mask,
            self.config.adsorbate_radius_A,
            exclude_same_node=True,
        )
        slab_to_adsorbate = self._relation_from_master(
            master,
            slab_mask,
            ads_mask,
            self.config.cross_radius_A,
        )
        adsorbate_to_slab = self._relation_from_master(
            master,
            ads_mask,
            slab_mask,
            self.config.cross_radius_A,
        )
        cross_graph = self._merge_relations(
            slab_to_adsorbate, adsorbate_to_slab
        )

        prepared_input = self._prepare_relation("input", input_graph)
        prepared_slab = self._prepare_relation("slab", slab_graph)
        prepared_adsorbate = self._prepare_relation("adsorbate", adsorbate_graph)
        prepared_cross = self._prepare_relation("cross", cross_graph)

        node_count = structures * atoms
        flat_numbers_1d = flat_numbers.reshape(node_count)
        flat_roles_1d = flat_roles.reshape(node_count)
        flat_valid_1d = flat_mask.reshape(node_count)
        flat_feature_update_1d = feature_update_mask.reshape(node_count)
        structure_index = torch.arange(structures, device=roles.device)
        structure_index = structure_index[:, None].expand(-1, atoms).reshape(-1)
        condition_index = torch.div(structure_index, generated, rounding_mode="floor")
        role_slot = (flat_roles_1d - 1).clamp(0, 2)
        values = flat_positions.new_zeros(
            (node_count, (self.config.lmax + 1) ** 2, self.config.num_channels)
        )
        scalar = self.element_embedding(flat_numbers_1d) + self.role_embedding(
            flat_roles_1d
        )
        scalar = scalar + initial_condition[condition_index, role_slot]
        values[:, 0, :] = scalar * flat_valid_1d[:, None]
        if prepared_input.edge_count:
            values = values + self.edge_degree_embedding(
                flat_numbers_1d,
                prepared_input.distance_features,
                prepared_input.edge_index,
                prepared_input.envelope,
            )
        values = values * flat_valid_1d[:, None, None]

        for layer_index, block in enumerate(self.blocks):
            node_modulation = layer_condition[
                condition_index, layer_index, role_slot
            ]
            condition_scale = node_modulation[:, 0, :] * flat_valid_1d[:, None]
            condition_shift = node_modulation[:, 1, :] * flat_valid_1d[:, None]
            if self.config.activation_checkpointing and self.training:
                def run_block(
                    current: Tensor,
                    scale: Tensor,
                    shift: Tensor,
                    module: nn.Module = block,
                ) -> Tensor:
                    return module(
                        current,
                        flat_numbers_1d,
                        flat_feature_update_1d,
                        prepared_slab,
                        prepared_adsorbate,
                        prepared_cross,
                        scale,
                        shift,
                        self.config.condition_modulation_scale,
                    )

                values = checkpoint(
                    run_block,
                    values,
                    condition_scale,
                    condition_shift,
                    use_reentrant=False,
                )
            else:
                values = block(
                    values,
                    flat_numbers_1d,
                    flat_feature_update_1d,
                    prepared_slab,
                    prepared_adsorbate,
                    prepared_cross,
                    condition_scale,
                    condition_shift,
                    self.config.condition_modulation_scale,
                )

        values = self.output_norm(values)
        l1 = values[:, 1:4, :]
        surface_raw = self.surface_head(l1).squeeze(-1)
        ads_center_raw = self.ads_center_head(l1).squeeze(-1)
        ads_internal_raw = self.ads_internal_head(l1).squeeze(-1)
        surface_raw = surface_raw.reshape(structures, atoms, 3)
        ads_center_raw = ads_center_raw.reshape(structures, atoms, 3)
        ads_internal_raw = ads_internal_raw.reshape(structures, atoms, 3)

        surface_delta = (
            surface_raw
            * self.config.surface_output_scale_A
            * surface_mask[..., None]
        )
        center_delta = (
            _masked_mean(ads_center_raw, ads_mask, dim=1)
            * self.config.ads_center_output_scale_A
        )
        internal_delta = _zero_mean(
            ads_internal_raw * self.config.ads_internal_output_scale_A,
            ads_mask,
        )

        r0_center = _masked_mean(flat_positions, ads_mask, dim=1)
        r0_relative = (flat_positions - r0_center[:, None]) * ads_mask[..., None]
        generated_ads = self._wrap_adsorbate_center(
            r0_center + center_delta,
            r0_relative + internal_delta,
            flat_cell,
            ads_mask,
        )
        output = flat_positions + surface_delta
        output = torch.where(ads_mask[..., None], generated_ads, output)
        output = output * flat_mask[..., None]

        shape = (batch, generated)
        return {
            "positions": output.reshape(*shape, atoms, 3),
            "surface_displacement": surface_delta.reshape(*shape, atoms, 3),
            "ads_center_displacement": center_delta.reshape(*shape, 3),
            "ads_internal_displacement": internal_delta.reshape(*shape, atoms, 3),
            "atom_mask": atom_mask,
            "roles": roles,
        }


def build_generator(config: dict) -> EquiformerV3AdsorptionGenerator:
    return EquiformerV3AdsorptionGenerator(
        EquiformerV3GeneratorConfig.from_dict(config)
    )
