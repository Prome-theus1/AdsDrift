"""Geometry, Wigner-D, SO(2), and SO(3) operators used by AdsDrift.

These operators are consolidated from the vendored EquiformerV3 implementation.
AdsDrift-specific graph construction and model composition live in the parent
``generator.py`` module.
"""


# ============================== edge_rot_mat.py ===============================

import torch


"""
    For gradient methods, we do not backpropogate rotation if y component of
    the unit vector of relative position is very close to `_ROTATION_MASK_THRESHOLD`.
"""
_ROTATION_MASK_THRESHOLD = 0.999999


def init_edge_rot_mat(edge_distance_vec, use_rotation_mask=False):
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

    # Make sure the atoms are far enough apart
    #assert torch.min(edge_vec_0_distance) < 0.0001
    if torch.min(edge_vec_0_distance) < 0.0001:
        print(
            "Error edge_vec_0_distance: {}".format(
                torch.min(edge_vec_0_distance)
            )
        )

    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

    if use_rotation_mask:
        """
            For gradient methods, we do not backpropogate rotation if y component of
            the unit vector of relative position is very close to `_ROTATION_MASK_THRESHOLD`.
        """
        yprod = norm_x @ norm_x.new_tensor([0.0, 1.0, 0.0])
        norm_x[yprod >  _ROTATION_MASK_THRESHOLD] = norm_x.new_tensor([0.0,  1.0, 0.0])
        norm_x[yprod < -_ROTATION_MASK_THRESHOLD] = norm_x.new_tensor([0.0, -1.0, 0.0])

    edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
    edge_vec_2 = edge_vec_2 / (
        torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1)
    )
    # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
    # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]
    vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(
        -1, 1
    )
    vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(
        -1, 1
    )

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(
        torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2
    )
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(
        torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2
    )

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
    # Check the vectors aren't aligned
    assert torch.max(vec_dot) < 0.99

    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / (
        torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True))
    )
    norm_z = norm_z / (
        torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1)
    )
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / (
        torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True))
    )

    # Construct the 3D rotation matrix
    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

    if use_rotation_mask:
        return edge_rot_mat
    else:
        return edge_rot_mat.detach()


# ================================= wigner.py ==================================

import os
import torch


# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L10
# _Jd is a list of tensors of shape (2l+1, 2l+1)
_Jd = torch.load(
    os.path.join(os.path.dirname(__file__), "Jd.pt"), weights_only=True
)


# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L37
#
# In 0.5.0, e3nn shifted to torch.matrix_exp which is significantly slower:
# https://github.com/e3nn/e3nn/blob/0.5.0/e3nn/o3/_wigner.py#L92
def wigner_D(l, alpha, beta, gamma):
    if not l < len(_Jd):
        raise NotImplementedError(
            f"wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more"
        )

    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[l].to(dtype=alpha.dtype, device=alpha.device)
    Xa = _z_rot_mat(alpha, l)
    Xb = _z_rot_mat(beta, l)
    Xc = _z_rot_mat(gamma, l)
    return Xa @ J @ Xb @ J @ Xc


def _z_rot_mat(angle, l):
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
    inds = torch.arange(0, 2 * l + 1, 1, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    return M


# ================================= so2_ops.py =================================

import torch
import math


class SO2MLinear(torch.nn.Module):
    """
        Perform an SO(2) linear operation to features corresponding to +- m

        Args:
            m (int):                    Order of the spherical harmonic coefficients
            num_in_channels (int):      Number of input channels
            num_out_channels (int):     Number of output channels
            lmax (int):                 Maximum degrees (l)
            mmax (int):                 Maximum order (m)
    """
    def __init__(
        self,
        m,
        num_in_channels,
        num_out_channels,
        lmax,
        mmax
    ):
        super(SO2MLinear, self).__init__()

        self.m = m
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.lmax = lmax
        self.mmax = mmax

        num_m_components = self.lmax - self.m + 1
        assert num_m_components > 0

        self.in_features = num_m_components * self.num_in_channels
        self.out_features = num_m_components * self.num_out_channels

        self.fc = torch.nn.Linear(
            self.in_features,
            (2 * self.out_features),
            bias=False
        )
        self.fc.weight.data.mul_(1 / math.sqrt(2))


    def forward(self, x_m, concat_outputs=True):
        x_m = self.fc(x_m)
        x_r = x_m.narrow(2, 0, self.out_features)
        x_i = x_m.narrow(2, self.out_features, self.out_features)
        x_m_r = x_r.narrow(1, 0, 1) - x_i.narrow(1, 1, 1) #x_r[:, 0] - x_i[:, 1]
        x_m_i = x_r.narrow(1, 1, 1) + x_i.narrow(1, 0, 1) #x_r[:, 1] + x_i[:, 0]
        #x_r, x_i = torch.chunk(x_m, chunks=2, dim=2)
        #x_r_0, x_r_1 = torch.chunk(x_r, chunks=2, dim=1)
        #x_i_0, x_i_1 = torch.chunk(x_i, chunks=2, dim=1)
        #x_m_r = x_r_0 - x_i_1
        #x_m_i = x_r_1 + x_i_0
        x_out = (x_m_r, x_m_i)
        if concat_outputs:
            x_out = torch.cat(x_out, dim=1)
        return x_out


class SO2Linear(torch.nn.Module):
    """
        Perform SO(2) linear operations to all m (orders) components

        Args:
            num_in_channels (int):      Number of input channels
            num_out_channels (int):     Number of output channels
            lmax (int):                 Maximum degrees (l)
            mmax (int):                 Maximum order (m)
            extra_m0_out_channels (int):    If not None, return `outputs` (torch.Tensor) and `extra_m0_features` (torch.Tensor).
    """
    def __init__(
        self,
        num_in_channels,
        num_out_channels,
        lmax,
        mmax,
        extra_m0_out_channels=None
    ):
        super(SO2Linear, self).__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.lmax = lmax
        self.mmax = mmax
        self.extra_m0_out_channels = extra_m0_out_channels

        # for m = 0
        num_in_channels_m0 = (self.lmax + 1) * self.num_in_channels
        num_out_channels_m0 = (self.lmax + 1) * self.num_out_channels
        if self.extra_m0_out_channels is not None:
            self.num_channels_m0_list = [self.extra_m0_out_channels, num_out_channels_m0]
            num_out_channels_m0 = num_out_channels_m0 + self.extra_m0_out_channels
        self.fc_m0 = torch.nn.Linear(num_in_channels_m0, num_out_channels_m0)

        # SO(2) linear for non-zero m
        self.so2_m_linear = torch.nn.ModuleList()
        for m in range(1, self.mmax + 1):
            self.so2_m_linear.append(
                SO2MLinear(
                    m,
                    self.num_in_channels,
                    self.num_out_channels,
                    self.lmax,
                    self.mmax,
                )
            )


    def forward(self, x):
        """
            1.  `x` shape: [num_edges, num_m_components, num_channels]
            2.  We assume the layout of m components is (0, 0, ...), (1, 1, ...), ...
        """
        num_edges = x.shape[0]
        outputs = []

        # Compute m=0 coefficients separately since they only have real values (no imaginary)
        x_m0 = x.narrow(1, 0, (self.lmax + 1))
        x_m0 = x_m0.reshape(num_edges, -1)
        x_m0 = self.fc_m0(x_m0)

        x_m0_extra = None
        # extract extra m0 features
        if self.extra_m0_out_channels is not None:
            x_m0_extra, x_m0 = torch.split(x_m0, self.num_channels_m0_list, dim=1)

        x_m0 = x_m0.view(num_edges, -1, self.num_out_channels)
        outputs.append(x_m0)

        # Compute the values for the m > 0 coefficients
        offset = self.lmax + 1
        for m in range(1, self.mmax + 1):
            x_m = x.narrow(1, offset, 2 * (self.lmax + 1 - m))
            offset = offset + 2 * (self.lmax + 1 - m)
            x_m = x_m.reshape(num_edges, 2, -1)
            """
            x_m = self.so2_m_linear[m - 1](x_m, concat_outputs=True)
            x_m = x_m.view(num_edges, -1, self.num_out_channels)
            out.append(x_m)
            """
            # Replace the original one with the followings to prevent one `torch.cat()` for each m > 0
            x_m = self.so2_m_linear[m - 1](x_m, concat_outputs=False)
            x_m_pos, x_m_neg = x_m[0], x_m[1]
            x_m_pos = x_m_pos.view(num_edges, -1, self.num_out_channels)
            x_m_neg = x_m_neg.view(num_edges, -1, self.num_out_channels)
            outputs.append(x_m_pos)
            outputs.append(x_m_neg)

        outputs = torch.cat(outputs, dim=1)

        if self.extra_m0_out_channels is not None:
            return outputs, x_m0_extra
        else:
            return outputs


# =================================== so3.py ===================================

import math
import torch
import copy

# e3nn 0.4.x stores slice objects in its trusted packaged constants.  PyTorch
# 2.6+ defaults torch.load to weights_only=True, so allow that single builtin
# before importing e3nn rather than disabling safe loading globally.
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])

from e3nn import o3
from e3nn.o3 import FromS2Grid, ToS2Grid



class CoefficientMappingModule(torch.nn.Module):
    """
    Helper module for coefficients used to reshape l <--> m and to get coefficients of specific degree or order

    Args:
        lmax (int):             Maximum degree of the spherical harmonics
        mmax (int):             Maximum order of the spherical harmonics
        use_rotate_inv_rescale (bool):
                                Whether to pre-compute inverse rotation rescale matrices
    """
    def __init__(
        self,
        lmax,
        mmax,
        use_rotate_inv_rescale=False
    ):
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax
        self.use_rotate_inv_rescale = use_rotate_inv_rescale

        # Compute the degree (l) and order (m) for each entry of the embedding
        l_harmonic = []
        m_harmonic = []
        m_complex  = []

        for l in range(0, self.lmax + 1):
            mmax = min(self.mmax, l)
            m = torch.arange(-mmax, mmax + 1).long()
            m_complex.append(m)
            m_harmonic.append(torch.abs(m).long())
            l_harmonic.append(torch.fill(m, l))
        m_complex = torch.cat(m_complex, dim=0)
        m_harmonic = torch.cat(m_harmonic, dim=0)
        l_harmonic = torch.cat(l_harmonic, dim=0)

        num_m_coefficients = len(l_harmonic)
        # `self.to_m` moves m components from different L to contiguous index
        to_m = torch.zeros([num_m_coefficients, num_m_coefficients])

        offset = 0
        for m in range(self.mmax + 1):
            idx_r, idx_i = self.complex_idx(m, -1, m_complex, l_harmonic)

            for idx_out, idx_in in enumerate(idx_r):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_r)

            for idx_out, idx_in in enumerate(idx_i):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_i)

        to_m = to_m.detach()

        # save tensors and they will be moved to GPU
        self.register_buffer('l_harmonic', l_harmonic)
        self.register_buffer('m_harmonic', m_harmonic)
        self.register_buffer('m_complex',  m_complex)
        self.register_buffer('to_m',       to_m)

        # for `torch.compile()` compatibility
        self.pre_compute_coefficient_idx()
        if self.use_rotate_inv_rescale:
            self.pre_compute_rotate_inv_rescale()


    # Return mask containing coefficients of order m (real and imaginary parts)
    def complex_idx(self, m, lmax, m_complex, l_harmonic):
        '''
            Add `m_complex` and `l_harmonic` to the input arguments
            since we cannot use `self.m_complex`.
        '''
        if lmax == -1:
            lmax = self.lmax

        indices = torch.arange(len(l_harmonic))
        # Real part
        mask_r = torch.bitwise_and(
            l_harmonic.le(lmax), m_complex.eq(m)
        )
        mask_idx_r = torch.masked_select(indices, mask_r)

        mask_idx_i = torch.tensor([]).long()
        # Imaginary part
        if m != 0:
            mask_i = torch.bitwise_and(
                l_harmonic.le(lmax), m_complex.eq(-m)
            )
            mask_idx_i = torch.masked_select(indices, mask_i)

        return mask_idx_r, mask_idx_i


    def pre_compute_coefficient_idx(self):
        '''
            Pre-compute the results of `coefficient_idx()` and access them with `prepare_coefficient_idx()`
        '''
        for l in range(self.lmax + 1):
            for m in range(self.lmax + 1):
                mask = torch.bitwise_and(
                    self.l_harmonic.le(l), self.m_harmonic.le(m)
                )
                indices = torch.arange(len(mask))
                mask_indices = torch.masked_select(indices, mask)
                self.register_buffer('coefficient_idx_l{}_m{}'.format(l, m), mask_indices)
        return


    def prepare_coefficient_idx(self):
        '''
            Construct a list of buffers
        '''
        coefficient_idx_list = []
        for l in range(self.lmax + 1):
            l_list = []
            for m in range(self.lmax + 1):
                l_list.append(getattr(self, 'coefficient_idx_l{}_m{}'.format(l, m), None))
            coefficient_idx_list.append(l_list)
        return coefficient_idx_list


    # Return mask containing coefficients less than or equal to degree (l) and order (m)
    def coefficient_idx(self, lmax, mmax):
        if lmax > self.lmax or mmax > self.lmax:
            mask = torch.bitwise_and(
                self.l_harmonic.le(lmax), self.m_harmonic.le(mmax)
            )
            indices = torch.arange(len(mask), device=mask.device)
            mask_indices = torch.masked_select(indices, mask)
            return mask_indices
        else:
            temp = self.prepare_coefficient_idx()
            return temp[lmax][mmax]


    def pre_compute_rotate_inv_rescale(self):
        for l in range(self.lmax + 1):
            for m in range(self.lmax + 1):
                mask_indices = self.coefficient_idx(l, m)
                rotate_inv_rescale = torch.ones((1, int((l + 1)**2), int((l + 1)**2)))
                for l_sub in range(l + 1):
                    if l_sub <= m:
                        continue
                    start_idx = l_sub ** 2
                    length = 2 * l_sub + 1
                    rescale_factor = math.sqrt(length / (2 * m + 1))
                    rotate_inv_rescale[:, start_idx : (start_idx + length), start_idx : (start_idx + length)] = rescale_factor
                rotate_inv_rescale = rotate_inv_rescale[:, :, mask_indices]
                self.register_buffer('rotate_inv_rescale_l{}_m{}'.format(l, m), rotate_inv_rescale)
        return


    def prepare_rotate_inv_rescale(self):
        rotate_inv_rescale_list = []
        for l in range(self.lmax + 1):
            l_list = []
            for m in range(self.lmax + 1):
                l_list.append(getattr(self, 'rotate_inv_rescale_l{}_m{}'.format(l, m), None))
            rotate_inv_rescale_list.append(l_list)
        return rotate_inv_rescale_list


    # Return the re-scaling for rotating back to original frame
    # this is required since we only use a subset of m components for SO(2) convolution
    def get_rotate_inv_rescale(self, lmax, mmax):
        temp = self.prepare_rotate_inv_rescale()
        return temp[lmax][mmax]


    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, mmax={self.mmax})"


class SO3Embedding():
    """
    1.  Helper functions for performing operations on irreps embedding
    2.  Deprecated since we can infer the lmax and mmax from the shape of tensors.

    Args:
        lmax (int):             Maximum degree of the spherical harmonics
        num_channels (int):     Number of channels
        device:                 Device of the output
        dtype:                  type of the output tensors
    """
    def __init__(
        self,
        lmax,
        num_channels,
        device,
        dtype,
    ):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.device = device
        self.dtype = dtype
        self.num_m_coefficients = (self.lmax + 1) ** 2
        self.set_lmax_mmax(self.lmax, self.lmax)


    # Clone an embedding of irreps
    def clone(self):
        clone = SO3Embedding(
            self.lmax,
            self.num_channels,
            self.device,
            self.dtype,
        )
        clone.set_embedding(self.embedding.clone())
        return clone


    # Initialize an embedding of irreps
    def set_embedding(self, embedding):
        self.embedding = embedding


    def set_lmax_mmax(self, lmax, mmax):
        self.lmax = lmax
        self.mmax = mmax


    # Expand the node embeddings to the number of edges
    def _expand_edge(self, edge_index):
        embedding = self.embedding[edge_index]
        self.set_embedding(embedding)


    # Initialize an embedding of irreps of a neighborhood
    def expand_edge(self, edge_index):
        x_expand = SO3Embedding(
            self.lmax,
            self.num_channels,
            self.device,
            self.dtype,
        )
        x_expand.set_embedding(self.embedding[edge_index])
        return x_expand


    # Compute the sum of the embeddings of the neighborhood
    def _reduce_edge(self, edge_index, num_nodes):
        new_embedding = torch.zeros(
            num_nodes,
            self.num_m_coefficients,
            self.num_channels,
            device=self.embedding.device,
            dtype=self.embedding.dtype,
        )
        new_embedding.index_add_(0, edge_index, self.embedding)
        self.set_embedding(new_embedding)


    # Reshape the embedding l -> m
    def _m_primary(self, mapping):
        self.embedding = torch.einsum("nac, ba -> nbc", self.embedding, mapping.to_m)


    # Reshape the embedding m -> l
    def _l_primary(self, mapping):
        self.embedding = torch.einsum("nac, ab -> nbc", self.embedding, mapping.to_m)


    # Rotate the embedding
    def _rotate(self, so3_rotation):
        embedding_rotate = so3_rotation.rotate(self.embedding)
        self.embedding = embedding_rotate
        self.set_lmax_mmax(so3_rotation.lmax, so3_rotation.mmax)


    # Rotate the embedding by the inverse of the rotation matrix
    def _rotate_inv(self, so3_rotation):
        embedding_rotate = so3_rotation.rotate_inv(self.embedding)
        self.embedding = embedding_rotate
        # Assume mmax = lmax when rotating back
        self.set_lmax_mmax(so3_rotation.lmax, so3_rotation.lmax)


class SO3Rotation(torch.nn.Module):
    """
        1.  Helper functions for Wigner-D rotations
        2.  We merge the rotation with the original `._m_primary()` so after rotation, the layout of orders
            would be changed from (0, (-1, 0, +1), (-2, -1, 0, +1, +2) ...) to ((0, ...), (1, ...)).
            This can skip one matrix multiplication.
        3.  Similar to 2., we also merge the inverse rotation with `._l_primary()`.
        4.  To stabilize gradient methods, in `_rotation_to_wigner_matrix()`, we set `use_rotation_mask` == True
            so that we do not backpropogate rotation if y component of the unit vector of relative position is
            very close to `_ROTATION_MASK_THRESHOLD`. This implementation is based on eSEN.

        Args:
            lmax (int):     Maximum degree of irreps features
            mmax (int):     Maximum order of irreps features after rotation
    """
    def __init__(
        self,
        lmax,
        mmax,
        use_rotation_mask=False
    ):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.use_rotation_mask = use_rotation_mask

        # The output of Wigner-D matrix only has a subset of m components
        mapping = CoefficientMappingModule(
            lmax=self.lmax,
            mmax=self.lmax,
            use_rotate_inv_rescale=True
        )
        wigner_index_mask = mapping.coefficient_idx(self.lmax, self.mmax)
        wigner_inv_rescale = mapping.get_rotate_inv_rescale(self.lmax, self.mmax)

        # Merge converting m and l layout
        mapping = CoefficientMappingModule(
            lmax=self.lmax,
            mmax=self.mmax,
            use_rotate_inv_rescale=False
        )
        to_m = mapping.to_m
        wigner_inv_rescale = torch.einsum('nia, ba -> nib', wigner_inv_rescale, to_m)
        wigner_index_to_m_array = torch.zeros(
            to_m.shape[0],
            ((self.lmax + 1) ** 2)
        )
        wigner_index_to_m_array[:, wigner_index_mask] = to_m

        self.register_buffer('wigner_index_to_m_array', wigner_index_to_m_array)
        self.register_buffer('wigner_inv_rescale', wigner_inv_rescale)


    def set_wigner(self, rot_mat3x3):
        wigner = self._rotation_to_wigner_matrix(rot_mat3x3, 0, self.lmax)
        #wigner = torch.matmul(self.wigner_index_to_m_array, wigner)
        wigner = torch.einsum('mi, nij -> nmj', self.wigner_index_to_m_array, wigner)
        if torch.is_autocast_enabled():
            wigner = wigner.to(torch.float16)
        wigner_inv = torch.transpose(wigner, 1, 2).contiguous()
        wigner_inv = wigner_inv * self.wigner_inv_rescale
        if torch.is_autocast_enabled():
            wigner_inv = wigner_inv.to(torch.float16)
        self.wigner = wigner            #.detach()
        self.wigner_inv = wigner_inv    #.detach()


    # Rotate the embedding
    def rotate(self, inputs):
        #outputs = torch.einsum('bji, bic -> bjc', self.wigner, inputs)
        outputs = torch.bmm(self.wigner, inputs)
        return outputs


    # Rotate the embedding by the inverse of the rotation matrix
    def rotate_inv(self, inputs):
        #outputs = torch.einsum('bij, bjc -> bic', self.wigner_inv, inputs)
        outputs = torch.bmm(self.wigner_inv, inputs)
        return outputs


    # Compute Wigner matrices from rotation matrix
    def _rotation_to_wigner_matrix(self, edge_rot_mat, start_lmax, end_lmax):
        #x = edge_rot_mat @ edge_rot_mat.new_tensor([0.0, 1.0, 0.0])
        #x = torch.einsum(
        #    'bij, j -> bi',
        #    edge_rot_mat,
        #    edge_rot_mat.new_tensor([0.0, 1.0, 0.0])
        #)
        x = edge_rot_mat[:, :, 1]

        alpha, beta = o3.xyz_to_angles(x)
        R = o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2)

        #R = R @ edge_rot_mat
        #R = torch.einsum('bik, bkj -> bij', R, edge_rot_mat)
        R = torch.bmm(R, edge_rot_mat)

        gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])

        if self.use_rotation_mask:
            yprod = (x @ x.new_tensor([0, 1, 0])).detach()
            backprop_mask = (yprod > -_ROTATION_MASK_THRESHOLD) & (yprod < _ROTATION_MASK_THRESHOLD)
            alpha_detach = alpha[(~backprop_mask)].clone().detach()
            gamma_detach = gamma[(~backprop_mask)].clone().detach()
            beta_detach = beta.clone().detach()
            beta_detach[yprod >  _ROTATION_MASK_THRESHOLD] = 0.0
            beta_detach[yprod < -_ROTATION_MASK_THRESHOLD] = math.pi
            beta_detach = beta_detach[(~backprop_mask)]

        size = int((end_lmax + 1) ** 2) - int((start_lmax) ** 2)
        wigner = torch.zeros(len(alpha), size, size, device=edge_rot_mat.device)
        start = 0
        for lmax in range(start_lmax, end_lmax + 1):
            if self.use_rotation_mask:
                block = wigner_D(
                    lmax,
                    alpha[backprop_mask],
                    beta[backprop_mask],
                    gamma[backprop_mask]
                )
                block_detach = wigner_D(
                    lmax,
                    alpha_detach,
                    beta_detach,
                    gamma_detach
                )
                end = start + block.size()[1]
                wigner[   backprop_mask, start:end, start:end] = block
                wigner[(~backprop_mask), start:end, start:end] = block_detach
            elif not self.use_rotation_mask:
                block = wigner_D(lmax, alpha, beta, gamma)
                end = start + block.size()[1]
                wigner[:, start:end, start:end] = block
            start = end
        if self.use_rotation_mask:
            return wigner
        else:
            return wigner.detach()


    def extra_repr(self):
        return 'lmax={}, mmax={}'.format(self.lmax, self.mmax)


class SO3Grid(torch.nn.Module):
    """
    Helper functions for grid representation of the irreps

    Args:
        lmax (int):   Maximum degree of the spherical harmonics
        mmax (int):   Maximum order of the spherical harmonics
        normalization (str):    Default: 'component'
                                How grid samples are normalized.
        resolution_list (list:int):
                                Default: None
                                List of grid resolutions corresponding to `lat_resolution` and `long_resolution`.
                                Set to `None` to use default resolutions.
        use_m_primary (bool):   Default: False
                                Whether to change the layout of m components.
                                If `False`, the layout of m is (0), (-1, 0, +1), (-2, -1, 0, +1, +2), ...
                                If `True`, the layout of m is (0, 0, ...), (1, 1, ...), ...
                                The second one is used in SO(2) linear operations to avoid redundant
                                matrix multiplications.
    """
    def __init__(
        self,
        lmax,
        mmax,
        normalization='component',
        resolution_list=None,
        use_m_primary=False
    ):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.use_m_primary = use_m_primary
        self.lat_resolution = 2 * (self.lmax + 1)
        if lmax == mmax:
            self.long_resolution = 2 * (self.mmax + 1) + 1
        else:
            self.long_resolution = 2 * (self.mmax) + 1
        if resolution_list is not None:
            assert isinstance(resolution_list, list)
            resolution_list = copy.deepcopy(resolution_list)
            self.lat_resolution = resolution_list[0]
            self.long_resolution = resolution_list[1]

        mapping = CoefficientMappingModule(
            lmax=self.lmax,
            mmax=self.lmax,
            use_rotate_inv_rescale=False
        )

        to_grid = ToS2Grid(
            self.lmax,
            (self.lat_resolution, self.long_resolution),
            normalization=normalization, #normalization="integral",
            device='cpu',
        )
        to_grid_mat = torch.einsum("mbi, am -> bai", to_grid.shb, to_grid.sha).detach()
        # rescale based on mmax
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l ** 2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                to_grid_mat[:, :, start_idx : (start_idx + length)] = to_grid_mat[:, :, start_idx : (start_idx + length)] * rescale_factor
        to_grid_mat = to_grid_mat[:, :, mapping.coefficient_idx(self.lmax, self.mmax)]

        from_grid = FromS2Grid(
            (self.lat_resolution, self.long_resolution),
            self.lmax,
            normalization=normalization, #normalization="integral",
            device='cpu',
        )
        from_grid_mat = torch.einsum("am, mbi -> bai", from_grid.sha, from_grid.shb).detach()
        # rescale based on mmax
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l ** 2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                from_grid_mat[:, :, start_idx : (start_idx + length)] = from_grid_mat[:, :, start_idx : (start_idx + length)] * rescale_factor
        from_grid_mat = from_grid_mat[:, :, mapping.coefficient_idx(self.lmax, self.mmax)]

        # flatten and permute
        to_grid_mat   = to_grid_mat.flatten(0, 1)
        from_grid_mat = from_grid_mat.flatten(0, 1)
        from_grid_mat = from_grid_mat.permute(1, 0)

        # change the layout of m components
        if self.use_m_primary:
            temp = CoefficientMappingModule(self.lmax, self.mmax, False)
            to_grid_mat = torch.einsum('ai, ji -> aj', to_grid_mat, temp.to_m)
            from_grid_mat = torch.einsum('ia, ji -> ja', from_grid_mat, temp.to_m)
            #from_grid_mat = torch.einsum('ai, ji -> aj', from_grid_mat, temp.to_m)
            #to_grid_mat = torch.einsum('bai, ji -> baj', to_grid_mat, temp.to_m)
            #from_grid_mat = torch.einsum('bai, ji -> baj', from_grid_mat, temp.to_m)

        # save tensors and they will be moved to GPU
        self.register_buffer('to_grid_mat',   to_grid_mat)
        self.register_buffer('from_grid_mat', from_grid_mat)


    # Compute matrices to transform irreps to grid
    def get_to_grid_mat(self):
        return self.to_grid_mat


    # Compute matrices to transform grid to irreps
    def get_from_grid_mat(self):
        return self.from_grid_mat


    # Compute grid from irreps representation
    def to_grid(self, embedding):
        #grid = torch.matmul(self.to_grid_mat, embedding)
        grid = torch.einsum('aj, njc -> nac', self.to_grid_mat, embedding)
        #grid = torch.einsum('baj, njc -> nbac', self.to_grid_mat, embedding)
        return grid


    # Compute irreps from grid representation
    def from_grid(self, grid):
        #embedding = torch.matmul(self.from_grid_mat, grid)
        embedding = torch.einsum('ja, nac -> njc', self.from_grid_mat, grid)
        #embedding = torch.einsum('aj, nac -> njc', self.from_grid_mat, grid)
        #embedding = torch.einsum('baj, nbac -> njc', self.from_grid_mat, grid)
        return embedding


    def extra_repr(self):
        return 'lmax={}, mmax={}, lat_resolution={}, long_resolution={}, use_m_primary={}'.format(self.lmax, self.mmax, self.lat_resolution, self.long_resolution, self.use_m_primary)


class SO3Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, lmax, bias=True):
        '''
            1.  Use `torch.einsum` to prevent slicing and concatenation
            2.  Need to specify some behaviors in `no_weight_decay` and weight initialization.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax

        self.weight = torch.nn.Parameter(torch.randn((self.lmax + 1), out_features, in_features))
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(1, 1, out_features)) if bias else None

        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for l in range(lmax + 1):
            start_idx = l ** 2
            length = 2 * l + 1
            expand_index[start_idx : (start_idx + length)] = l
        self.register_buffer('expand_index', expand_index)


    def forward(self, inputs):
        weight = torch.index_select(self.weight, dim=0, index=self.expand_index)        # [(L_max + 1) ** 2, C_out, C_in]
        outputs = torch.einsum('bmi, moi -> bmo', inputs, weight)                       # [N, (L_max + 1) ** 2, C_out]
        outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + self.bias
        return outputs


    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lmax={self.lmax}, bias={(self.bias is not None)})"
