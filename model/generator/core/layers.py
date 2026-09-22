"""Reusable radial, normalization, activation, dropout, and graph layers.

GraphSoftmax is implemented with native PyTorch scatter operations, avoiding a
runtime dependency on torch-geometric while preserving target-wise attention
normalization.
"""


# ============================= radial_function.py =============================

import torch


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class RadialFunction(torch.nn.Module):
    """
        1.  Contruct a radial function (linear layers + layer normalization + SiLU) given a list of channels
        2.  If `use_rad_l_parametrization` == True and `use_expand` == True, all the m components
            within a type-L vector will share the same weight.
            We expand the outputs so that they can be directly multiplied with SO(3) features.
        3.  If `use_rad_l_parametrization` == False and `use_expand` == True, the +=m within the same
            type-L vector will share the same weight while different m will have different weights.
            Thus, this will have more parameters than 3.
            Different from 2., We expand the outputs so that they can be directly multiplied with SO(2) features.
        4.  If `use_expand` == False, we simply return the outputs of radial functions.
    """
    def __init__(self, channels_list, lmax=None, mmax=None, use_rad_l_parametrization=True, use_expand=True):
        super().__init__()
        modules = []
        input_channels = channels_list[0]
        for i in range(len(channels_list)):
            if i == 0:
                continue

            modules.append(torch.nn.Linear(input_channels, channels_list[i], bias=True))
            input_channels = channels_list[i]

            if i == len(channels_list) - 1:
                break

            modules.append(torch.nn.LayerNorm(channels_list[i]))
            modules.append(torch.nn.SiLU())

        self.net = torch.nn.Sequential(*modules)

        self.lmax = lmax
        self.mmax = mmax
        self.use_rad_l_parametrization = use_rad_l_parametrization
        self.use_expand = use_expand

        if self.use_expand:
            if not self.use_rad_l_parametrization:
                expand_index = []
                offset = 0
                for m in range(self.mmax + 1):
                    index = torch.arange((self.lmax + 1 - m))
                    index = index + offset
                    expand_index.append(index)
                    if m > 0:
                        expand_index.append(index)    # +- m
                    offset = offset + len(index)
                expand_index = torch.cat(expand_index, dim=0)
                expand_index = expand_index.long()
                self.register_buffer('expand_index', expand_index)
                self.num_m_components = offset
                assert channels_list[-1] % self.num_m_components == 0
            else:
                assert self.lmax == self.mmax
                expand_index = torch.zeros([((self.lmax + 1) ** 2)]).long()
                start_idx = 0
                for l in range(self.lmax + 1):
                    length = 2 * l + 1
                    expand_index[start_idx : (start_idx + length)] = l
                    start_idx = start_idx + length
                self.register_buffer('expand_index', expand_index)
                assert channels_list[-1] % (self.lmax + 1) == 0


    def forward(self, inputs):
        outputs = self.net(inputs)
        if self.use_expand:
            if not self.use_rad_l_parametrization:
                # Convert to the format that can be directly multiplied with SO(2) features
                outputs = outputs.view(outputs.shape[0], self.num_m_components, -1)
            else:
                # Convert to the format that can be directly multiplied with SO(3) features
                outputs = outputs.view(outputs.shape[0], (self.lmax + 1), -1)
            outputs = torch.index_select(outputs, dim=1, index=self.expand_index)
        return outputs


# ================================ envelope.py =================================

import torch


class PolynomialEnvelope(torch.nn.Module):
    """
        1.  Polynomial envelope function that ensures a smooth cutoff.
        2.  Reference: https://github.com/facebookresearch/fairchem/blob/518d0ea12110548bd5ffaf9a43060b8eae152e13/src/fairchem/core/models/esen/nn/radial.py#L22
    """
    def __init__(self, cutoff: float = 6.0, exponent: int = 5) -> None:
        super().__init__()
        assert exponent > 0
        self.cutoff = float(cutoff)
        self.exponent = exponent
        self.p: float = float(exponent)
        self.a: float = -(self.p + 1) * (self.p + 2) / 2
        self.b: float = self.p * (self.p + 2)
        self.c: float = -self.p * (self.p + 1) / 2


    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        d_scaled = distance / self.cutoff
        env_val = (
            1
            + self.a * d_scaled**self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        outputs = torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))
        outputs = outputs.view(-1, 1)
        return outputs


    def extra_repr(self):
        return 'cutoff={}, exponent={}'.format(self.cutoff, self.exponent)


# =============================== layer_norm.py ================================

import torch
from functools import partial


_NORM_TYPE_LIST = [
    'equivariant_layer_norm',
    'sep_layer_norm',
    'merge_layer_norm',
    'merge_layer_norm_attn_rms_norm',   # Use `EquivariantMergeLayerNorm` for the pre-norm layer
                                        # and `RMSNorm` for attention re-normalization
    'merge_rms_norm'
]


def get_normalization_layer(norm_type, lmax, num_channels, eps=1e-5, affine=True, normalization='component'):
    assert norm_type in _NORM_TYPE_LIST
    if norm_type == 'equivariant_layer_norm':
        norm_class = EquivariantLayerNorm
    elif norm_type == 'sep_layer_norm':
        norm_class = EquivariantSeparableLayerNorm
    elif norm_type in ['merge_layer_norm', 'merge_layer_norm_attn_rms_norm']:
        norm_class = EquivariantMergeLayerNorm
    elif norm_type == 'merge_rms_norm':
        norm_class = partial(EquivariantMergeLayerNorm, centering=False)
    else:
        raise ValueError
    return norm_class(lmax, num_channels, eps, affine, normalization)


class EquivariantLayerNorm(torch.nn.Module):
    def __init__(self, lmax, num_channels, eps=1e-5, affine=True, normalization='component'):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.affine_weight = torch.nn.Parameter(torch.ones((self.lmax + 1), self.num_channels))
            self.affine_bias   = torch.nn.Parameter(torch.zeros(self.num_channels))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization


    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps})"


    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, inputs):
        """
            1.   `inputs` shape: (num_nodes, (self.lmax + 1) ** 2, self.num_channels)
        """
        outputs = []

        for l in range(self.lmax + 1):
            start_idx = l ** 2
            length = 2 * l + 1

            feature = inputs.narrow(1, start_idx, length)

            # For scalars, first compute and subtract the mean
            if l == 0:
                feature_mean = torch.mean(feature, dim=2, keepdim=True)
                feature = feature - feature_mean

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == 'norm':
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True)      # [N, 1, C]
            elif self.normalization == 'component':
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)     # [N, 1, C]

            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)    # [N, 1, 1]
            feature_norm = (feature_norm + self.eps).pow(-0.5)

            if self.affine:
                weight = self.affine_weight.narrow(0, l, 1)     # [1, C]
                weight = weight.view(1, 1, -1)                  # [1, 1, C]
                feature_norm = feature_norm * weight            # [N, 1, C]

            feature = feature * feature_norm

            if self.affine and l == 0:
                bias = self.affine_bias
                bias = bias.view(1, 1, -1)
                feature = feature + bias

            outputs.append(feature)

        outputs = torch.cat(outputs, dim=1)

        return outputs


class EquivariantSeparableLayerNorm(torch.nn.Module):
    """
        1.  Use `expand_index` to skip for loop during affine transformation.
    """
    def __init__(self, lmax, num_channels, eps=1e-5, affine=True, normalization='component', std_balance_degrees=True):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.std_balance_degrees = std_balance_degrees

        # for L = 0
        self.norm_l0 = torch.nn.LayerNorm(self.num_channels, eps=self.eps, elementwise_affine=self.affine)

        # for L > 0
        if self.affine:
            self.affine_weight = torch.nn.Parameter(torch.ones(self.lmax, self.num_channels))
            expand_index = torch.zeros([((self.lmax + 1) ** 2 - 1)]).long()     # L > 0
            for l in range(1, self.lmax + 1):
                start_idx = l ** 2 - 1
                length = 2 * l + 1
                expand_index[start_idx : (start_idx + length)] = (l - 1)
            self.register_buffer('expand_index', expand_index)
        else:
            self.register_parameter('affine_weight', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2 - 1, 1)
            for l in range(1, self.lmax + 1):
                start_idx = l ** 2 - 1
                length = 2 * l + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = (1.0 / length)
            balance_degree_weight = balance_degree_weight / self.lmax
            balance_degree_weight = balance_degree_weight.permute((1, 0))
            self.register_buffer('balance_degree_weight', balance_degree_weight)
        else:
            self.balance_degree_weight = None


    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, std_balance_degrees={self.std_balance_degrees})"


    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, inputs):
        """
            1.  `inputs` shape: (num_nodes, (self.lmax + 1) ** 2, self.num_channels)
        """
        outputs = []

        # for L = 0
        scalars = inputs.narrow(1, 0, 1)
        scalars = self.norm_l0(scalars)
        outputs.append(scalars)

        # for L > 0
        if self.lmax > 0:
            num_m_components = (self.lmax + 1) ** 2
            feature = inputs.narrow(1, 1, num_m_components - 1)

            feature_norm = feature.pow(2)
            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)        # [N, (L_max + 1)**2 - 1, 1]

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == 'norm':
                feature_norm = feature_norm.sum(dim=1, keepdim=True)            # [N, 1, 1]
            elif self.normalization == 'component':
                if self.std_balance_degrees:
                    #feature_norm = feature.pow(2)                               # [N, (L_max + 1)**2 - 1, C], without L = 0
                    #feature_norm = torch.einsum('nic, ia -> nac', feature_norm, self.balance_degree_weight) # [N, 1, C]
                    feature_norm = torch.einsum('ai, nic -> nac', self.balance_degree_weight, feature_norm) # [N, 1, C]
                    #feature_norm = torch.matmul(self.balance_degree_weight, feature_norm) # [N, 1, 1]
                else:
                    feature_norm = feature_norm.mean(dim=1, keepdim=True)       # [N, 1, 1]

            feature_norm = (feature_norm + self.eps).pow(-0.5)

            if self.affine:
                weight = self.affine_weight.view(1, self.lmax, self.num_channels)
                weight = torch.index_select(weight, dim=1, index=self.expand_index)
                feature_norm = feature_norm * weight
            feature = feature * feature_norm

            outputs.append(feature)

        outputs = torch.cat(outputs, dim=1)
        return outputs


class EquivariantMergeLayerNorm(torch.nn.Module):
    """
        1.  Use `expand_index` to skip for loop during affine transformation.
        2.  Different from `EquivariantSeparableLayerNorm`, we normalize over all degrees L >= 0.
        3.  If `centering == False`, this becomes RMSNorm for all degrees.
    """
    def __init__(self, lmax, num_channels, eps=1e-5, affine=True, normalization='component', std_balance_degrees=True, centering=True):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.std_balance_degrees = std_balance_degrees
        self.centering = centering

        if self.affine:
            self.affine_weight = torch.nn.Parameter(torch.ones((self.lmax + 1), self.num_channels))
            expand_index = torch.zeros([((self.lmax + 1) ** 2)]).long()     # L >= 0
            for l in range(self.lmax + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                expand_index[start_idx : (start_idx + length)] = l
            self.register_buffer('expand_index', expand_index)

            if self.centering:
                self.affine_bias = torch.nn.Parameter(torch.zeros(self.num_channels))
            else:
                self.register_parameter('affine_bias', None)
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2, 1)
            for l in range(self.lmax + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = (1.0 / length)
            balance_degree_weight = balance_degree_weight / (self.lmax + 1)
            balance_degree_weight = balance_degree_weight.permute((1, 0))
            self.register_buffer('balance_degree_weight', balance_degree_weight)
        else:
            self.balance_degree_weight = None


    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, std_balance_degrees={self.std_balance_degrees}, centering={self.centering})"


    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, inputs):
        """
            1.  `inputs` shape: (num_nodes, (self.lmax + 1) ** 2, self.num_channels)
        """
        # for L = 0
        if self.centering:
            scalars = inputs.narrow(1, 0, 1)
            scalars_mean = scalars.mean(dim=2, keepdim=True) # [N, 1, 1]
            scalars = scalars - scalars_mean
            inputs = torch.cat((scalars, inputs.narrow(1, 1, inputs.shape[1] - 1)), dim=1)

        # for L >= 0
        feature_norm = inputs.pow(2)
        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)        # [N, (L_max + 1)**2, 1]
        if self.normalization == 'norm':
            feature_norm = feature_norm.sum(dim=1, keepdim=True)            # [N, 1, 1]
        elif self.normalization == 'component':
            if self.std_balance_degrees:
                feature_norm = torch.einsum('ai, nic -> nac', self.balance_degree_weight, feature_norm) # [N, 1, 1]
            else:
                feature_norm = feature_norm.mean(dim=1, keepdim=True)       # [N, 1, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)
        if self.affine:
            weight = self.affine_weight.view(1, (self.lmax + 1), self.num_channels)
            weight = torch.index_select(weight, dim=1, index=self.expand_index)
            feature_norm = feature_norm * weight
        outputs = inputs * feature_norm

        if self.affine and self.centering:
            outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + self.affine_bias.view(1, 1, self.num_channels)

        return outputs


class RMSNorm(torch.nn.Module):
    """
        1. Reference: https://github.com/meta-llama/llama/blob/1e8375848d3a3ebaccab83fd670b880864cf9409/llama/model.py#L34
    """
    def __init__(self, num_channels: int, eps: float = 1e-5):
        """
            Initialize the RMSNorm normalization layer.

            Args:
                dim (int): The dimension of the input tensor.
                eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

            Attributes:
                eps (float): A small value added to the denominator for numerical stability.
                weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.ones(self.num_channels))


    def _norm(self, x):
        """
            Apply the RMSNorm normalization to the input tensor.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


    def forward(self, x):
        """
            Forward pass through the RMSNorm layer.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


    def __repr__(self):
        return f"{self.__class__.__name__}(num_channels={self.num_channels}, eps={self.eps})"


# ================================== drop.py ===================================

import torch


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class GraphDropPath(torch.nn.Module):
    '''
        Consider batch for graph data when dropping paths.
    '''
    def __init__(self, drop_prob=None):
        super(GraphDropPath, self).__init__()
        self.drop_prob = drop_prob


    def forward(self, x, batch):
        batch_size = batch.max() + 1
        shape = (batch_size, ) + (1, ) * (x.ndim - 1)  # work with different dim tensors
        ones = torch.ones(shape, dtype=x.dtype, device=x.device)
        drop = drop_path(ones, self.drop_prob, self.training)
        out = x * drop[batch]
        return out


    def extra_repr(self):
        return 'drop_prob={}'.format(self.drop_prob)


class EquivariantDropout(torch.nn.Module):
    """
        1.  When dropping one type-L vector, we set all the m components to zeros.
    """
    def __init__(self, lmax, mmax, drop_prob, use_m_primary=False):
        super(EquivariantDropout, self).__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.drop_prob = drop_prob
        self.use_m_primary = use_m_primary

        self.drop = torch.nn.Dropout(drop_prob, True)

        expand_index = []
        if not self.use_m_primary:
            for l in range(self.lmax + 1):
                mmax = min(l, self.mmax)
                l_index_tensor = torch.ones(((2 * mmax + 1), ), dtype=torch.long) * l
                expand_index.append(l_index_tensor)
        elif self.use_m_primary:
            for m in range(self.mmax + 1):
                l_index = torch.arange((self.lmax + 1 - m))
                expand_index.append(l_index)
                if m > 0:
                    expand_index.append(l_index)    # +- m
        expand_index = torch.cat(expand_index, dim=0)
        expand_index = expand_index.long()
        self.register_buffer('expand_index', expand_index)


    def extra_repr(self):
        return 'lmax={}, mmax={}, drop_prob={}, use_m_primary={}'.format(
            self.lmax, self.mmax, self.drop_prob, self.use_m_primary
        )


    def forward(self, x):
        # x shape: (num_tokens, num_m_coefficients, num_channels)
        if not self.training or self.drop_prob == 0.0:
            return x

        assert len(x.shape) == 3
        shape = (x.shape[0], (self.lmax + 1), x.shape[2])
        mask = torch.ones(shape, dtype=x.dtype, device=x.device)
        mask = self.drop(mask)
        mask = torch.index_select(mask, dim=1, index=self.expand_index)
        out = x * mask
        return out


# =============================== activation.py ================================

import torch
import copy
from .geometry import SO3Grid


def check_activation_name(act_name):
    assert act_name in [
        'gate',
        's2',
        'sep_s2',
        's2_swiglu',
        's2_swiglu_mem',
        'sep_s2_swiglu',
        'sep-merge_s2_swiglu',
        'sep_s2_swiglu_mem',
        'sep-merge_s2_swiglu_mem',
        'sep_s2_square',
        'sep-merge_gates2_swiglu',
        'sep-merge_gates2_swiglu_mem'
    ]
    return


def get_activation(act_name, lmax, mmax, grid_resolution_list=None, use_m_primary=False):
    """
        use_m_primary (bool):   Default: False
                                Whether to change the layout of m components.
                                If `False`, the layout of m is (0), (-1, 0, +1), (-2, -1, 0, +1, +2), ...
                                If `True`, the layout of m is (0, 0, ...), (1, 1, ...), ...
                                The second one is used in SO(2) linear operations to avoid redundant
                                matrix multiplications.
    """
    check_activation_name(act_name)
    if act_name == 'gate':
        act_class = GateActivation
    elif act_name == 's2':
        act_class = S2Activation
    elif act_name == 'sep_s2':
        act_class = SeparableS2Activation
    elif act_name == 's2_swiglu':
        act_class = S2Activation_SwiGLU
    elif act_name == 's2_swiglu_mem':
        act_class = S2Activation_SwiGLU_MemoryEfficient
    elif act_name == 'sep_s2_swiglu':
        act_class = SeparableS2Activation_SwiGLU
    elif act_name == 'sep-merge_s2_swiglu':
        act_class = SeparableS2Activation_SwiGLU_Merge
    elif act_name == 'sep_s2_swiglu_mem':
        act_class = SeparableS2Activation_SwiGLU_MemoryEfficient
    elif act_name == 'sep-merge_s2_swiglu_mem':
        act_class = SeparableS2Activation_SwiGLU_Merge_MemoryEfficient
    elif act_name == 'sep_s2_square':
        act_class = SeparableS2Activation_Square
    elif act_name == 'sep-merge_gates2_swiglu':
        act_class = SeparableGateS2Activation_SwiGLU_Merge
    elif act_name == 'sep-merge_gates2_swiglu_mem':
        act_class = SeparableGateS2Activation_SwiGLU_Merge_MemoryEfficient
    args = {
        'lmax': lmax,
        'mmax': mmax,
        'use_m_primary': use_m_primary
    }
    if act_name != 'gate':
        args['grid_resolution_list'] = grid_resolution_list
    return act_class(**args)


def has_scalars(act_name):
    if act_name not in ['s2', 's2_swiglu', 's2_swiglu_mem']:
        return True
    return False


def add_dropout(act, drop):
    """
        1.  Add extra dropout to original activation functions
    """
    attribute_name_list = ['act', 'gate_act', 'scalar_act']
    for attr_name in attribute_name_list:
        if attr_name == 'gate_act' and isinstance(act, SeparableGateS2Activation_SwiGLU_Merge):
            continue # For `SeparableGateS2Activation_SwiGLU_Merge`, dropout is from `grid_drop`
        if hasattr(act, attr_name):
            temp = copy.deepcopy(getattr(act, attr_name))
            update_act_list = [
                temp,
                torch.nn.Dropout(drop)
            ]
            delattr(act, attr_name)
            setattr(act, attr_name, torch.nn.Sequential(*update_act_list))
    if hasattr(act, 'grid_drop'):
        delattr(act, 'grid_drop')
        setattr(act, 'grid_drop', torch.nn.Dropout(drop))
    return


def prepare_activation_forward_param(act_name, inputs, scalars):
    output_dict = {
        'inputs': inputs
    }
    if has_scalars(act_name):
        output_dict['scalars'] = scalars
    return output_dict


class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope


    def forward(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2


    def extra_repr(self):
        return 'negative_slope={}'.format(self.alpha)


class GateActivation(torch.nn.Module):
    def __init__(self, lmax, mmax, use_m_primary=False):
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax
        self.use_m_primary = use_m_primary

        # compute `expand_index` based on `lmax` and `mmax`
        num_components = 0
        for l in range(1, self.lmax + 1):
            num_m_components = min((2 * l + 1), (2 * self.mmax + 1))
            num_components = num_components + num_m_components
        if not self.use_m_primary:
            expand_index = torch.zeros([num_components]).long()
            start_idx = 0
            for l in range(1, self.lmax + 1):
                length = min((2 * l + 1), (2 * self.mmax + 1))
                expand_index[start_idx : (start_idx + length)] = (l - 1)
                start_idx = start_idx + length
        elif self.use_m_primary:
            expand_index = []
            for m in range(self.mmax + 1):
                if m == 0:
                    l_index = torch.arange(self.lmax)       # We do not have L = 0
                else:
                    l_index = torch.arange((m - 1), self.lmax)
                expand_index.append(l_index)
                if m > 0:
                    expand_index.append(l_index)            # +- m
            expand_index = torch.cat(expand_index, dim=0)
            expand_index = expand_index.long()
        self.register_buffer('expand_index', expand_index)

        self.scalar_act = torch.nn.SiLU()
        self.gate_act   = torch.nn.Sigmoid()


    def forward(self, inputs, scalars):
        '''
            `inputs`: shape  [N, (lmax + 1) ** 2, num_channels]
            `scalars`: shape [N, lmax * num_channels]
        '''
        gate_scalars = self.gate_act(scalars)
        gate_scalars = gate_scalars.reshape(gate_scalars.shape[0], self.lmax, -1)
        gate_scalars = torch.index_select(gate_scalars, dim=1, index=self.expand_index)

        # L = 0
        input_scalars = inputs.narrow(1, 0, 1)
        input_scalars = self.scalar_act(input_scalars)
        # L > 0
        input_vectors = inputs.narrow(1, 1, inputs.shape[1] - 1)
        input_vectors = input_vectors * gate_scalars

        output_tensors = torch.cat((input_scalars, input_vectors), dim=1)

        return output_tensors


    def extra_repr(self):
        return 'lmax={}, mmax={}, use_m_primary={}'.format(self.lmax, self.mmax, self.use_m_primary)


class S2Activation(torch.nn.Module):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.so3_grid = SO3Grid(self.lmax, self.mmax, resolution_list=grid_resolution_list, use_m_primary=use_m_primary)
        self.act = torch.nn.SiLU()


    def forward(self, inputs):
        x_grid = self.so3_grid.to_grid(inputs)
        x_grid = self.act(x_grid)
        outputs = self.so3_grid.from_grid(x_grid)
        return outputs


class SeparableS2Activation(S2Activation):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary)


    def forward(self, inputs, scalars):
        output_scalars = self.act(scalars)
        output_scalars = output_scalars.reshape(output_scalars.shape[0], 1, output_scalars.shape[1])
        output_vectors = super().forward(inputs)
        outputs = torch.cat(
            (output_scalars, output_vectors.narrow(1, 1, output_vectors.shape[1] - 1)),
            dim=1
        )
        return outputs


def swiglu_torch(gate, up_states):
    gate = torch.nn.functional.silu(gate)
    outputs = gate * up_states
    return outputs


class SwiGLU(torch.nn.Module):
    '''
        1.  The module only contains the activation.
        2.  The number of output channels is the half of that of input channels.
    '''
    def __init__(self, backend='torch'):
        super(SwiGLU, self).__init__()
        assert backend in ['torch']
        self.backend = backend
        self.func = swiglu_torch


    def forward(self, inputs):
        x_1, x_2 = torch.chunk(inputs, chunks=2, dim=-1)
        outputs = self.func(x_1, x_2)
        return outputs


    def extra_repr(self):
        return 'backend={}'.format(self.backend)


class LinearSwiGLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, backend='torch'):
        super(LinearSwiGLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, 2 * out_channels, bias=bias)
        self.act = SwiGLU(backend)


    def forward(self, inputs):
        outputs = self.linear(inputs)
        outputs = self.act(outputs)
        return outputs


class S2Activation_SwiGLU(S2Activation):
    '''
        1.  Assume we only have one resolution.
        2.  Use SwiGLU after projecting to grids.
    '''
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary)
        del self.act
        self.act = SwiGLU(backend)


class S2Activation_SwiGLU_MemoryEfficient(S2Activation_SwiGLU):
    '''
        1.  Assume we only have one resolution.
        2.  Use SwiGLU after projecting to grids.
        3.  We use gradient checkpointing for this activation function.
    '''
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary, backend)


    def kernel(self, inputs):
        x_grid = self.so3_grid.to_grid(inputs)
        x_grid = self.act(x_grid)
        outputs = self.so3_grid.from_grid(inputs)
        return outputs


    def forward(self, inputs):
        outputs = torch.utils.checkpoint.checkpoint(
            self.kernel,
            inputs,
            use_reentrant=False
        )
        return outputs


class SeparableS2Activation_SwiGLU(S2Activation_SwiGLU):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary, backend)


    def forward(self, inputs, scalars):
        output_scalars = self.act(scalars)
        output_scalars = output_scalars.reshape(output_scalars.shape[0], 1, output_scalars.shape[1])
        output_vectors = super().forward(inputs)
        outputs = torch.cat(
            (output_scalars, output_vectors.narrow(1, 1, output_vectors.shape[1] - 1)),
            dim=1
        )
        return outputs


class SeparableS2Activation_SwiGLU_Merge(S2Activation_SwiGLU):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary, backend)


    def forward(self, inputs, scalars):
        output_scalars = self.act(scalars)
        output_scalars = output_scalars.reshape(output_scalars.shape[0], 1, output_scalars.shape[1])
        output_vectors = super().forward(inputs)
        outputs = output_vectors #.clone()
        outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + output_scalars
        return outputs


class SeparableS2Activation_SwiGLU_MemoryEfficient(S2Activation_SwiGLU_MemoryEfficient):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary, backend)


    def forward(self, inputs, scalars):
        output_scalars = self.act(scalars)
        output_scalars = output_scalars.reshape(output_scalars.shape[0], 1, output_scalars.shape[1])
        output_vectors = super().forward(inputs)
        outputs = torch.cat(
            (output_scalars, output_vectors.narrow(1, 1, output_vectors.shape[1] - 1)),
            dim=1
        )
        return outputs


class SeparableS2Activation_SwiGLU_Merge_MemoryEfficient(S2Activation_SwiGLU_MemoryEfficient):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary, backend)


    def kernel(self, inputs, scalars):
        x_grid = self.so3_grid.to_grid(inputs)
        x_grid = self.act(x_grid)
        output_vectors = self.so3_grid.from_grid(x_grid)
        output_scalars = self.act(scalars)
        output_scalars = output_scalars.reshape(output_scalars.shape[0], 1, output_scalars.shape[-1])
        outputs = output_vectors #.clone()
        outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + output_scalars
        return outputs


    def forward(self, inputs, scalars):
        outputs = torch.utils.checkpoint.checkpoint(
            self.kernel,
            inputs,
            scalars,
            use_reentrant=False
        )
        return outputs


class Square(torch.nn.Module):
    '''
        The number of output channels is the half of that of input channels.
    '''
    def __init__(self):
        super(Square, self).__init__()


    def forward(self, inputs):
        x_1, x_2 = torch.chunk(inputs, chunks=2, dim=-1)
        outputs = x_1 * x_2
        return outputs


class LinearSquare(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(LinearSquare, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, 2 * out_channels, bias=bias)
        self.act = Square()


    def forward(self, inputs):
        outputs = self.linear(inputs)
        outputs = self.act(outputs)
        return outputs


class SeparableS2Activation_Square(torch.nn.Module):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.so3_grid = SO3Grid(self.lmax, self.mmax, resolution_list=grid_resolution_list, use_m_primary=use_m_primary)
        self.act  = Square()


    def forward(self, inputs, scalars):
        x_grid = self.so3_grid.to_grid(inputs)
        x_grid = self.act(x_grid)
        output_vectors = self.so3_grid.from_grid(x_grid)
        output_scalars = self.act(scalars)
        output_scalars = output_scalars.reshape(output_scalars.shape[0], 1, output_scalars.shape[-1])
        outputs = torch.cat(
            (output_scalars, output_vectors.narrow(1, 1, output_vectors.shape[1] - 1)),
            dim=1
        )
        return outputs


class SeparableGateS2Activation_SwiGLU_Merge(GateActivation):
    """
        1.  'Separable' means that we have two paths (type-0 vectors and type-L vectors (L >= 0)).
        2.  We divide type-0 vectors into two parts for 3. and 4.
        3.  We apply SwiGLU to the first part of type-0 path ('SwiGLU').
        4.  We apply Sigmoid to the second part of type-0 path ('Gate'), which is to be used in 5.
        5.  For the type-L path, we project to S2 grid signals ('S2') and divide into two parts of
            equal sizes.
            We use the nonlinear weights in 4. to gate the first part of the S2 grid signals
            ('GateS2Activation').
            Then, we perform elementwise multiplication, which is equivalent to the self tensor product for
            many-body interactions.
            After elementwise multiplication, we optionally apply dropout, which enables training in a
            non-equivariant manner.
            Finally, we project S2 grid signals back to equivariant features.
            Note that this is similar to SwiGLU (but without SiLU activation) in 3.
        6.  We merge the two paths mentioned in 1. by adding the type-0 parts ('Merge').
    """
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, use_m_primary)
        self.so3_grid = SO3Grid(self.lmax, self.mmax, resolution_list=grid_resolution_list, use_m_primary=use_m_primary)
        del self.scalar_act
        self.scalar_act = SwiGLU(backend)
        del self.expand_index
        self.grid_drop = torch.nn.Identity()


    def forward(self, inputs, scalars):
        """
            `inputs`: shape  [N, (lmax + 1) ** 2, 2 * num_channels]
            `scalars`: shape [N, 2 * num_channels + num_channels] or
                             [N, 1, 2 * num_channels + num_channels]
        """
        scalars = scalars.view(scalars.shape[0], 1, scalars.shape[-1])
        # 3. SwiGLU to type-0 path
        output_scalars = scalars.narrow(2, 0, inputs.shape[2])
        gate_scalars = scalars.narrow(2, output_scalars.shape[2], (scalars.shape[2] - output_scalars.shape[2]))
        output_scalars = self.scalar_act(output_scalars)    # [N, 1, num_channels]
        # 4. Sigmoid for gating
        gate_scalars = self.gate_act(gate_scalars)          # [N, 1, num_channels]
        # 5. Project to S2 grid signals, perform nonlinear gating, perform elementwise multiplication,
        #    optionally perform dropout, and project back
        x_grid = self.so3_grid.to_grid(inputs)
        x_grid_1, x_grid_2 = torch.chunk(x_grid, chunks=2, dim=-1)
        #x_grid_1 = x_grid_1 * gate_scalars
        x_grid = x_grid_1 * x_grid_2
        x_grid = self.grid_drop(x_grid)
        output_vectors = self.so3_grid.from_grid(x_grid)
        output_vectors = output_vectors * gate_scalars
        # 6. Merge
        outputs = output_vectors
        outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + output_scalars
        return outputs


class SeparableGateS2Activation_SwiGLU_Merge_MemoryEfficient(SeparableGateS2Activation_SwiGLU_Merge):
    """
        1.  Add gradient checkpointing to `SeparableGateS2Activation_SwiGLU_Merge`
    """
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary, backend)


    def kernel(self, inputs, scalars):
        """
            `inputs`: shape  [N, (lmax + 1) ** 2, 2 * num_channels]
            `scalars`: shape [N, 2 * num_channels + num_channels] or
                             [N, 1, 2 * num_channels + num_channels]
        """
        scalars = scalars.view(scalars.shape[0], 1, scalars.shape[-1])
        # 3. SwiGLU to type-0 path
        output_scalars = scalars.narrow(2, 0, inputs.shape[2])
        gate_scalars = scalars.narrow(2, output_scalars.shape[2], (scalars.shape[2] - output_scalars.shape[2]))
        output_scalars = self.scalar_act(output_scalars)    # [N, 1, num_channels]
        # 4. Sigmoid for gating
        gate_scalars = self.gate_act(gate_scalars)          # [N, 1, num_channels]
        # 5. Project to S2 grid signals, perform nonlinear gating, perform elementwise multiplication,
        #    optionally perform dropout, and project back
        x_grid = self.so3_grid.to_grid(inputs)
        x_grid_1, x_grid_2 = torch.chunk(x_grid, chunks=2, dim=-1)
        #x_grid_1 = x_grid_1 * gate_scalars
        x_grid = x_grid_1 * x_grid_2
        x_grid = self.grid_drop(x_grid)
        output_vectors = self.so3_grid.from_grid(x_grid)
        output_vectors = output_vectors * gate_scalars
        # 6. Merge
        outputs = output_vectors
        outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + output_scalars
        return outputs


    def forward(self, inputs, scalars):
        outputs = torch.utils.checkpoint.checkpoint(
            self.kernel,
            inputs,
            scalars,
            use_reentrant=False
        )
        return outputs

import torch


class SoftCap(torch.nn.Module):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap

    def forward(self, inputs):
        return torch.tanh(inputs / self.cap) * self.cap

    def __repr__(self):
        return f"{self.__class__.__name__}(cap={self.cap})"


class GraphSoftmax(torch.nn.Module):
    """Target-wise stable softmax with optional envelope and edge dropout."""

    def __init__(self, eps=1e-16, exp_dropout=0.0, softcap=None):
        super().__init__()
        self.eps = eps
        self.exp_dropout = exp_dropout
        self.dropout = (
            torch.nn.Dropout(exp_dropout)
            if self.exp_dropout > 0.0
            else torch.nn.Identity()
        )
        self.softcap = (
            SoftCap(cap=softcap) if softcap is not None else torch.nn.Identity()
        )

    @staticmethod
    def _expanded_index(index, src):
        return index.view(-1, *([1] * (src.ndim - 1))).expand_as(src)

    def forward(
        self,
        src,
        index=None,
        ptr=None,
        num_nodes=None,
        dim=0,
        exp_rescale=None,
    ):
        if dim != 0:
            raise NotImplementedError("AdsDrift GraphSoftmax currently uses dim=0")
        if ptr is not None:
            count = ptr[1:] - ptr[:-1]
            index = torch.arange(len(count), device=ptr.device).repeat_interleave(count)
            num_nodes = len(count)
        if index is None:
            raise NotImplementedError("GraphSoftmax needs an index or CSR pointer")

        src = self.softcap(src)
        groups = (
            int(num_nodes)
            if num_nodes is not None
            else (int(index.max().item()) + 1 if index.numel() else 0)
        )
        output_shape = (groups, *src.shape[1:])
        expanded_index = self._expanded_index(index, src)
        src_max = src.new_full(output_shape, float("-inf"))
        src_max.scatter_reduce_(
            0, expanded_index, src.detach(), reduce="amax", include_self=True
        )
        out = (src - src_max.index_select(0, index)).exp()
        if exp_rescale is not None:
            out = out * exp_rescale
        out = self.dropout(out)
        out_sum = src.new_zeros(output_shape)
        out_sum.scatter_add_(0, expanded_index, out)
        return out / (out_sum.index_select(0, index) + self.eps)

    def extra_repr(self):
        return f"eps={self.eps}"


# ================================== utils.py ==================================

import torch


def reduce_edge(inputs, edge_index, output_shape):
    outputs = torch.zeros(
        *output_shape,
        device=inputs.device,
        dtype=inputs.dtype,
    )
    outputs.index_add_(0, edge_index, inputs)
    return outputs
