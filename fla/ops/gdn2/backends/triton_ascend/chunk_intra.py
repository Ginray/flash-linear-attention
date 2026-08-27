# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GDN-2 chunk intra kernels for triton-ascend on Ascend NPU.

Architecture mirrors the KDA Ascend port: the fused CUDA kernels are split so
that (a) scalar row-substitution loops run on a small dedicated grid
(diag_solve), (b) the block-merge kernel only does Cube dots (inter_solve),
and the backward is split at the debug_barrier into dq_db / dkt_future /
dk_dg with a 1D core-grid task loop. GDN-2 replaces KDA's scalar beta with
channel-wise gates b [.., K] (erase) applied to the key side, so every Akk
construction loads a [BC, BK] b-tile instead of post-multiplying a [BC] row
vector, and the backward db partials are [BC, BK] tiles rather than row sums.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.runtime import driver

from fla.ops.gdn2.backends.triton_ascend.wy_fast import (
    _gk_npu_arg,
    recompute_w_u_fwd_gdn2_npu as _recompute_w_u_fwd_npu,
)
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    max_grid_axis_chunks,
)

_BC = 16
_FWD_BC = 32
# GDN-2 carries one extra [BC, BK] gate tile per operand set vs KDA.
_SUB_CHUNK_MEM_MULT = 7.0
_INTER_MEM_MULT = 17.0
_SAFETY_MARGIN = 0.80
_FALLBACK_BK = 16
_MAX_INTER_BK = 64
# limit programs per launch to stay within Ascend AICore task time.
_GDN2_LAUNCH_BLOCK_BUDGET = 4096

# Backward split at debug_barrier: dq_db half vs dk/dg half have different
# peak live sets; GDN-2 adds the b-tile and the [BC, BK] db partial on top of
# KDA's budget (9.0/4.5), and dkt_future adds the per-channel b_j tile.
# dq_db is further split into dq_past (dynamic j-loop over past sub-chunks,
# first-row decay reference) and a straight-line diagonal kernel, mirroring
# the dkt_future + dk_dg pairing: BiSheng emits scalar-bound code when dots
# sit inside a dynamic loop, so the loop lives in its own kernel whose dots
# never mix with straight-line dots. Both halves tile for fp32 compute.
_BWD_INTRA_BC = 32
_BWD_INTRA_DQ_MEM_MULT = 9.0
_BWD_INTRA_DK_MEM_MULT = 5.5
_BWD_INTRA_FUT_MEM_MULT = 7.5
_MAX_BK_DQ = 128
_MAX_BK_DK = 256


def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


def _get_sub_chunk_bk(K: int, BC: int = _BC) -> int:
    return compute_row_tile_block_size(
        BC,
        K,
        _SUB_CHUNK_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BK,
        min_block=16,
        max_block=triton.next_power_of_2(K),
    )


def _get_inter_bk(K: int, BC: int = _BC) -> int:
    return compute_row_tile_block_size(
        BC,
        K,
        _INTER_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BK,
        min_block=16,
        max_block=min(_MAX_INTER_BK, triton.next_power_of_2(K)),
    )


def _launch_sub_chunk_kernel(
    kernel,
    *,
    nt: int,
    nc: int,
    bh_total: int,
    kernel_kwargs: dict,
) -> None:
    budget = _GDN2_LAUNCH_BLOCK_BUDGET
    chunk_indices = kernel_kwargs.get('chunk_indices')
    cu_seqlens = kernel_kwargs.get('cu_seqlens')
    nt_step = nt if nt * nc * bh_total <= budget else max(1, budget // max(nc * bh_total, 1))
    for nt_off in range(0, nt, nt_step):
        nt_len = min(nt_step, nt - nt_off)
        if cu_seqlens is not None and chunk_indices is not None:
            kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs['NT_OFFSET'] = 0
        else:
            kernel_kwargs['NT_OFFSET'] = nt_off
        nc_budget = max(1, budget // max(nt_len * bh_total, 1))
        max_nc = min(
            nc_budget,
            max_grid_axis_chunks(nc, nt_len * bh_total, max_grid=ASCEND_MAX_GRID_DIM),
        )
        for nc_off in range(0, nc, max_nc):
            nc_len = min(max_nc, nc - nc_off)
            kernel_kwargs['NC_OFFSET'] = nc_off
            bh_budget = max(1, budget // max(nt_len * nc_len, 1))
            max_bh = min(
                bh_budget,
                max_grid_axis_chunks(bh_total, nt_len * nc_len, max_grid=ASCEND_MAX_GRID_DIM),
            )
            for bh_off in range(0, bh_total, max_bh):
                bh_len = min(max_bh, bh_total - bh_off)
                kernel_kwargs['BH_OFFSET'] = bh_off
                kernel[(nt_len, nc_len, bh_len)](**kernel_kwargs)


def _launch_inter_kernel(
    kernel,
    *,
    nt: int,
    bh_total: int,
    kernel_kwargs: dict,
) -> None:
    budget = _GDN2_LAUNCH_BLOCK_BUDGET
    chunk_indices = kernel_kwargs.get('chunk_indices')
    cu_seqlens = kernel_kwargs.get('cu_seqlens')
    nt_step = nt if nt * bh_total <= budget else max(1, min(nt, budget // max(bh_total, 1)))
    for nt_off in range(0, nt, nt_step):
        nt_len = min(nt_step, nt - nt_off)
        if cu_seqlens is not None and chunk_indices is not None:
            kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs['NT_OFFSET'] = 0
        else:
            kernel_kwargs['NT_OFFSET'] = nt_off
        bh_budget = max(1, budget // max(nt_len, 1))
        max_bh = min(
            bh_budget,
            max_grid_axis_chunks(bh_total, nt_len, max_grid=ASCEND_MAX_GRID_DIM),
        )
        for bh_off in range(0, bh_total, max_bh):
            bh_len = min(max_bh, bh_total - bh_off)
            kernel_kwargs['BH_OFFSET'] = bh_off
            kernel[(nt_len, bh_len)](**kernel_kwargs)


@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'NC_OFFSET', 'BH_OFFSET'])
def chunk_gdn2_fwd_kernel_diag_solve_npu(
    Akkd,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET,
    NC_OFFSET,
    BH_OFFSET,
):
    """Per-subchunk lower-triangular forward substitution into Akkd.

    Run before inter_solve so the fused inter kernel only merges off-diagonal
    blocks, keeping scalar BC loops off the large (NT, BH) grid.
    """
    i_t = tl.program_id(0) + NT_OFFSET
    i_i = tl.program_id(1) + NC_OFFSET
    i_bh = tl.program_id(2) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T
        eos = bos + T

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    Akkd = Akkd + (bos * H + i_h).to(tl.int64) * BC
    o_i = tl.arange(0, BC)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    # Pure scalar forward substitution, no tl.dot in this kernel: mixing the
    # scalar column loop with dots (block inversion via [[A^-1,0],[-D^-1 C
    # A^-1, D^-1]] was tried, with both dynamic and fixed trip counts) trips
    # BiSheng codegen and roughly doubles the kernel time (61us -> 110+us).
    p_Akk = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_ti, 0), (BC, BC), (1, 0))
    b_Akk = tl.load(p_Akk, boundary_check=(0, 1)).to(tl.float32)
    b_Ai = -tl.where(m_A, b_Akk, 0)
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tl.load(Akkd + (i_ti + i).to(tl.int64) * H * BC + o_i)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    tl.store(p_Akk, b_Ai.to(Akkd.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'NC_OFFSET', 'BH_OFFSET'])
def chunk_gdn2_fwd_kernel_intra_sub_chunk_npu(
    q,
    k,
    g,
    b,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET,
    NC_OFFSET,
    BH_OFFSET,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_i = tl.program_id(1) + NC_OFFSET
    i_bh = tl.program_id(2) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T
        eos = bos + T

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_c = i_ti + tl.arange(0, BC)
    m_c = o_c < T

    q = q + (bos * H + i_h).to(tl.int64) * K
    k = k + (bos * H + i_h).to(tl.int64) * K
    g = g + (bos * H + i_h).to(tl.int64) * K
    b = b + (bos * H + i_h).to(tl.int64) * K
    Aqk = Aqk + (bos * H + i_h).to(tl.int64) * BT
    Akk = Akk + (bos * H + i_h).to(tl.int64) * BC

    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_b = tl.make_block_ptr(b, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_b = tl.load(p_b, boundary_check=(0, 1))

    p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)).to(tl.int64) * H * K + tl.arange(0, BK)
    b_gn = tl.load(p_gn, mask=tl.arange(0, BK) < K, other=0.0)
    b_gn = b_gn[None, :]

    b_gm = (b_g - b_gn).to(tl.float32)
    b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.)
    b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.)

    b_kgt = tl.trans(b_k * b_gk)

    # GDN-2: channel-wise erase gate b rides the key side of Akk.
    b_bk = b_b * b_k
    b_Aqk = tl.dot(b_q * b_gq, b_kgt, allow_tf32=False) * scale
    b_Akk = tl.dot(b_bk * b_gq, b_kgt, allow_tf32=False)

    o_i = tl.arange(0, BC)
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)

    p_Aqk = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
    p_Akk = tl.make_block_ptr(Akk, (T, BC), (H * BC, 1), (i_ti, 0), (BC, BC), (1, 0))
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'BH_OFFSET'])
def chunk_gdn2_fwd_kernel_inter_solve_fused_npu(
    q,
    k,
    g,
    b,
    Aqk,
    Akkd,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET,
    BH_OFFSET,
):
    # Diagonal Akkd blocks are inverted by diag_solve before this kernel.
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T
        eos = bos + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    q += (bos * H + i_h).to(tl.int64) * K
    k += (bos * H + i_h).to(tl.int64) * K
    g += (bos * H + i_h).to(tl.int64) * K
    b += (bos * H + i_h).to(tl.int64) * K
    Aqk += (bos * H + i_h).to(tl.int64) * BT
    Akk += (bos * H + i_h).to(tl.int64) * BT
    Akkd += (bos * H + i_h).to(tl.int64) * BC

    o_i = tl.arange(0, BC)
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k0 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        p_g0 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1)).to(tl.float32)
        b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)

        # Ascend cannot compile dynamic `if i_tc* < T` around dots (scf.if shape mismatch);
        # block_ptr uses boundary_check, and bare g/b loads mask out-of-range rows.
        p_q1 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
        p_k1 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
        p_g1 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
        p_b1 = tl.make_block_ptr(b, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
        b_q1 = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32)
        b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
        b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
        b_b1 = tl.load(p_b1, boundary_check=(0, 1)).to(tl.float32)
        b_gn1 = tl.load(g + i_tc1.to(tl.int64) * H * K + o_k, mask=m_k & (i_tc1 < T), other=0).to(tl.float32)
        b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
        b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))
        b_bk1 = b_b1 * b_k1
        b_Aqk10 += tl.dot(b_q1 * b_gqn, b_kgt, allow_tf32=False)
        b_Akk10 += tl.dot(b_bk1 * b_gqn, b_kgt, allow_tf32=False)

        if NC >= 3:
            p_q2 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
            p_k2 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
            p_g2 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
            p_b2 = tl.make_block_ptr(b, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
            b_q2 = tl.load(p_q2, boundary_check=(0, 1)).to(tl.float32)
            b_k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
            b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
            b_b2 = tl.load(p_b2, boundary_check=(0, 1)).to(tl.float32)
            b_gn2 = tl.load(g + i_tc2.to(tl.int64) * H * K + o_k, mask=m_k & (i_tc2 < T), other=0).to(tl.float32)
            b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
            b_qg2 = b_q2 * b_gqn2
            b_bkg2 = (b_b2 * b_k2) * b_gqn2
            b_qg2_c = b_qg2 + 0.0
            b_bkg2_c = b_bkg2 + 0.0
            b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0))
            b_Aqk20 += tl.dot(b_qg2, b_kgt, allow_tf32=False)
            b_Akk20 += tl.dot(b_bkg2, b_kgt, allow_tf32=False)
            b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1))
            b_Aqk21 += tl.dot(b_qg2_c, b_kgt, allow_tf32=False)
            b_Akk21 += tl.dot(b_bkg2_c, b_kgt, allow_tf32=False)

            if NC >= 4:
                p_q3 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                p_k3 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                p_g3 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                p_b3 = tl.make_block_ptr(b, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                b_q3 = tl.load(p_q3, boundary_check=(0, 1)).to(tl.float32)
                b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
                b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
                b_b3 = tl.load(p_b3, boundary_check=(0, 1)).to(tl.float32)
                b_gn3 = tl.load(g + i_tc3.to(tl.int64) * H * K + o_k, mask=m_k & (i_tc3 < T), other=0).to(tl.float32)
                b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                b_qg3 = b_q3 * b_gqn3
                b_bkg3 = (b_b3 * b_k3) * b_gqn3
                b_qg3_c1 = b_qg3 + 0.0
                b_bkg3_c1 = b_bkg3 + 0.0
                b_qg3_c2 = b_qg3 + 0.0
                b_bkg3_c2 = b_bkg3 + 0.0
                b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0))
                b_Aqk30 += tl.dot(b_qg3, b_kgt, allow_tf32=False)
                b_Akk30 += tl.dot(b_bkg3, b_kgt, allow_tf32=False)
                b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1))
                b_Aqk31 += tl.dot(b_qg3_c1, b_kgt, allow_tf32=False)
                b_Akk31 += tl.dot(b_bkg3_c1, b_kgt, allow_tf32=False)
                b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2))
                b_Aqk32 += tl.dot(b_qg3_c2, b_kgt, allow_tf32=False)
                b_Akk32 += tl.dot(b_bkg3_c2, b_kgt, allow_tf32=False)

    p_Aqk10 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    tl.store(p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    if NC >= 3:
        p_Aqk20 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
        p_Aqk21 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
        tl.store(p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    if NC >= 4:
        p_Aqk30 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
        p_Aqk31 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
        p_Aqk32 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
        tl.store(p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

    p_Akk00 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc1, 0), (BC, BC), (1, 0))
    b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
    if NC >= 3:
        p_Akk22 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc2, 0), (BC, BC), (1, 0))
        b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
    if NC >= 4:
        p_Akk33 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc3, 0), (BC, BC), (1, 0))
        b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)

    b_Ai11_c = b_Ai11 + 0.0
    if NC >= 3:
        b_Ai22_c = b_Ai22 + 0.0
        b_Ai22_c2 = b_Ai22 + 0.0
        b_Ai22_c3 = b_Ai22 + 0.0
    if NC >= 4:
        b_Ai33_c = b_Ai33 + 0.0
        b_Ai33_c2 = b_Ai33 + 0.0
        b_Ai33_c3 = b_Ai33 + 0.0
        b_Akk31_c = b_Akk31 + 0.0
        b_Akk32_c = b_Akk32 + 0.0

    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, allow_tf32=False),
        b_Ai00,
        allow_tf32=False,
    )

    if NC >= 3:
        b_Ai21 = -tl.dot(
            tl.dot(b_Ai22, b_Akk21, allow_tf32=False),
            b_Ai11_c,
            allow_tf32=False,
        )
        b_Ai20 = -tl.dot(
            b_Ai22_c2,
            tl.dot(b_Akk20, b_Ai00, allow_tf32=False) +
            tl.dot(b_Akk21, b_Ai10, allow_tf32=False),
            allow_tf32=False,
        )
    if NC >= 4:
        b_Ai32 = -tl.dot(
            tl.dot(b_Ai33, b_Akk32, allow_tf32=False),
            b_Ai22_c3,
            allow_tf32=False,
        )
        b_Ai31 = -tl.dot(
            b_Ai33_c2,
            tl.dot(b_Akk31, b_Ai11_c, allow_tf32=False) +
            tl.dot(b_Akk32, b_Ai21, allow_tf32=False),
            allow_tf32=False,
        )
        b_Ai30 = -tl.dot(
            b_Ai33_c3,
            tl.dot(b_Akk30, b_Ai00, allow_tf32=False) +
            tl.dot(b_Akk31_c, b_Ai10, allow_tf32=False) +
            tl.dot(b_Akk32_c, b_Ai20, allow_tf32=False),
            allow_tf32=False,
        )

    p_Akk00 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk10 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))

    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk11, b_Ai11_c.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    if NC >= 3:
        p_Akk20 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
        p_Akk21 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
        p_Akk22 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0))
        tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk22, b_Ai22_c.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    if NC >= 4:
        p_Akk30 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
        p_Akk31 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
        p_Akk32 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
        p_Akk33 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0))
        tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk33, b_Ai33_c.to(Akk.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def chunk_gdn2_fwd_intra_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    w_gate: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    safe_gate: bool = False,
    disable_recompute: bool = False,
):
    B, T, H, K = k.shape
    BT = chunk_size
    if BT not in (32, 64):
        raise ValueError(f"GDN2 intra chunk kernel only supports chunk_size 32 or 64, got {BT}.")
    # NC=2 sub-chunks: halving the sub-chunk count quarters the inter-solve
    # dot/solve chain (12 dots + 9 solve dots at NC=4 become 4 + 1) while the
    # diag-solve scalar column loop only doubles.
    BC = min(_FWD_BC, BT // 2)
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    is_varlen = cu_seqlens is not None

    # Aqk/Akk must stay zero-initialized: their per-chunk strict upper
    # triangles are never written (sub_chunk stores diagonal blocks, inter
    # stores lower off-diagonal blocks) and downstream kernels read them
    # without a tril mask (chunk_o masks with tl.where, but the shared KDA
    # dAv kernel consumes A unmasked). Akkd is fully overwritten by
    # sub_chunk (all rows < T across all sequences, all BC columns), so it
    # can be uninitialized like the CUDA reference.
    Aqk = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    Akk = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    Akkd = torch.empty(B, T, H, BC, device=k.device, dtype=torch.float32)

    # Always use the blocked sub-chunk kernel: it computes the whole [BC, BC]
    # score block with straight-line dots, while the token-parallel variant
    # stores one narrow [BH] column per inner step and is MTE3-bound (~2.4x
    # slower on B2T1024H4K64). safe_gate only selects the kernel variant, and
    # both produce numerically equivalent Aqk/Akkd here.
    sub_bk = _get_sub_chunk_bk(K, BC)
    _launch_sub_chunk_kernel(
        chunk_gdn2_fwd_kernel_intra_sub_chunk_npu,
        nt=NT,
        nc=NC,
        bh_total=B * H,
        kernel_kwargs=dict(
            q=q,
            k=k,
            g=gk,
            b=b,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            K=K,
            BT=BT,
            BC=BC,
            BK=sub_bk,
            IS_VARLEN=is_varlen,
            NT_OFFSET=0,
            NC_OFFSET=0,
            BH_OFFSET=0,
        ),
    )

    # Invert diagonal Akkd blocks first; inter then only merges off-diagonals.
    _launch_sub_chunk_kernel(
        chunk_gdn2_fwd_kernel_diag_solve_npu,
        nt=NT,
        nc=NC,
        bh_total=B * H,
        kernel_kwargs=dict(
            Akkd=Akkd,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            BT=BT,
            BC=BC,
            IS_VARLEN=is_varlen,
            NT_OFFSET=0,
            NC_OFFSET=0,
            BH_OFFSET=0,
        ),
    )

    inter_bk = _get_inter_bk(K, BC)
    _launch_inter_kernel(
        chunk_gdn2_fwd_kernel_inter_solve_fused_npu,
        nt=NT,
        bh_total=B * H,
        kernel_kwargs=dict(
            q=q,
            k=k,
            g=gk,
            b=b,
            Aqk=Aqk,
            Akkd=Akkd,
            Akk=Akk,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            K=K,
            BT=BT,
            BC=BC,
            NC=NC,
            BK=inter_bk,
            IS_VARLEN=is_varlen,
            NT_OFFSET=0,
            BH_OFFSET=0,
        ),
    )
    w, u, qg, kg = _recompute_w_u_fwd_npu(
        k=k,
        v=v,
        b=b,
        w_gate=w_gate,
        A=Akk,
        q=q if disable_recompute else None,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, qg, kg, Aqk, Akk


def _get_bwd_intra_bk(K: int, BC: int = _BWD_INTRA_BC, *, element_size: int = 2) -> int:
    """dq_past/dq_db_diag shared tile: both hold the j-loop live set plus
    the db partial, and must agree so the db2 [NK, .., BK] layout matches."""
    max_bk = _MAX_BK_DQ if element_size <= 2 else min(_MAX_BK_DQ, 64)
    return compute_row_tile_block_size(
        BC,
        K,
        _BWD_INTRA_DQ_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BK,
        min_block=16,
        max_block=min(max_bk, triton.next_power_of_2(K)),
    )


def _get_bwd_intra_bk_dk(K: int, BC: int = _BWD_INTRA_BC, *, element_size: int = 2) -> int:
    """Wider K tile for dk_dg (no past-subchunk live set). dkt_future cannot use this."""
    max_bk = _MAX_BK_DK if element_size <= 2 else min(_MAX_BK_DK, 64)
    return compute_row_tile_block_size(
        BC,
        K,
        _BWD_INTRA_DK_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BK,
        min_block=16,
        max_block=min(max_bk, triton.next_power_of_2(K)),
    )


def _launch_bwd_intra_core_grid(kernel, *, task_num: int, kernel_kwargs: dict) -> None:
    num_core = get_npu_properties()["num_aicore"]
    kernel[(num_core,)](task_num=task_num, num_core=num_core, **kernel_kwargs)


@triton.jit
def _bwd_intra_g_base(g, bos, i_b, i_h, T_seq, K, H: tl.constexpr, IS_VARLEN: tl.constexpr, G_T_CONTIG: tl.constexpr):
    # GDN-2 has no value-head split (HV == H), so the transposed layout is [B, H, T, K].
    if G_T_CONTIG:
        if IS_VARLEN:
            return g + (bos * K).to(tl.int64) + i_h.to(tl.int64) * T_seq * K
        return g + tl.cast(i_b, tl.int64) * H * T_seq * K + i_h.to(tl.int64) * T_seq * K
    return g + (bos * H + i_h).to(tl.int64) * K


@triton.jit
def _bwd_intra_g_row_stride(G_T_CONTIG: tl.constexpr, K: tl.constexpr, H: tl.constexpr):
    if G_T_CONTIG:
        return K
    return H * K


@triton.jit
def _bwd_intra_g_block_ptr(g_base, T, row, col, BC, BK, g_row_stride, K: tl.constexpr):
    return tl.make_block_ptr(g_base, (T, K), (g_row_stride, 1), (row, col), (BC, BK), (1, 0))


@triton.jit(do_not_specialize=['B', 'T', 'NT', 'BH_TOTAL', 'task_num', 'num_core'])
def chunk_gdn2_bwd_kernel_intra_dq_past_npu(
    q, k, g, b, dAqk, dAkk, dq, dq2, dk2, dg2, db,
    cu_seqlens, chunk_indices, B, T, NT, BH_TOTAL, task_num, num_core,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
    BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
    FULL_TILE: tl.constexpr,
):
    # Past-subchunk half of the old dq_db kernel: rows i accumulate
    # dA_j @ k_j over strictly-past sub-chunks j < i_i, using the same
    # mid-row decay reference as the diag kernel. It runs after diag and
    # accumulates onto its dq2/dk2/dg2/db2 results. Keeping the dynamic
    # loop here in isolation avoids BiSheng's mixed loop/straight-line dot
    # codegen pathology.
    core_id = tl.program_id(0)
    g_row_stride = _bwd_intra_g_row_stride(G_T_CONTIG, K, H)
    bc: tl.constexpr = () if FULL_TILE else (0, 1)
    for task_id in tl.range(core_id, task_num, num_core):
        i_bh = task_id % BH_TOTAL
        rem = task_id // BH_TOTAL
        i_t = rem % NT
        i_kc = rem // NT
        i_b, i_h = i_bh // H, i_bh % H
        # Grid encodes i_i one-based so sub-chunk 0 (no past) never launches.
        i_k, i_i = i_kc // NC, i_kc % NC + 1
        T_seq = T
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        else:
            bos = tl.cast(i_b, tl.int64) * T
            eos = bos + T
        T_cur = (eos - bos).to(tl.int32)
        i_ti = i_t * BT + i_i * BC
        if i_ti < T_cur:
            all = tl.cast(B, tl.int64) * T
            q_ptr = q + (bos * H + i_h).to(tl.int64) * K
            k_ptr = k + (bos * H + i_h).to(tl.int64) * K
            b_ptr = b + (bos * H + i_h).to(tl.int64) * K
            g_base = _bwd_intra_g_base(g, bos, i_b, i_h, T_seq, K, H, IS_VARLEN, G_T_CONTIG)
            dAqk_ptr = dAqk + (bos * H + i_h).to(tl.int64) * BT
            dAkk_ptr = dAkk + (bos * H + i_h).to(tl.int64) * BT
            dq_ptr = dq + (bos * H + i_h).to(tl.int64) * K
            dq2_ptr = dq2 + (bos * H + i_h).to(tl.int64) * K
            dk2_ptr = dk2 + (bos * H + i_h).to(tl.int64) * K
            dg2_ptr = dg2 + (bos * H + i_h).to(tl.int64) * K
            db_ptr = db + ((i_k * all + bos) * H + i_h).to(tl.int64) * BK
            p_g = _bwd_intra_g_block_ptr(g_base, T_cur, i_ti, i_k * BK, BC, BK, g_row_stride, K)
            b_g = tl.load(p_g, boundary_check=bc).to(tl.float32)
            # Same mid-row reference as the diag kernel: both exp2(g_mid - g_j)
            # and exp2(g_i - g_mid) are bounded by half a sub-chunk of decay.
            i_mid = i_ti + tl.minimum(BC // 2, T_cur - i_ti - 1)
            p_gm = g_base + i_mid.to(tl.int64) * g_row_stride + i_k * BK + tl.arange(0, BK)
            b_gm = tl.load(p_gm, mask=(i_k * BK + tl.arange(0, BK)) < K, other=0).to(tl.float32)[None, :]
            b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
            b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)
            for i_j in range(0, i_i):
                p_k = tl.make_block_ptr(k_ptr, (T_cur, K), (H * K, 1),
                                        (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
                p_gk = _bwd_intra_g_block_ptr(g_base, T_cur, i_t * BT + i_j * BC,
                                              i_k * BK, BC, BK, g_row_stride, K)
                p_dAqk = tl.make_block_ptr(dAqk_ptr, (T_cur, BT), (H * BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
                p_dAkk = tl.make_block_ptr(dAkk_ptr, (T_cur, BT), (H * BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
                b_k = tl.load(p_k, boundary_check=bc)
                b_gk = tl.load(p_gk, boundary_check=bc)
                b_dAqk = tl.load(p_dAqk, boundary_check=bc)
                b_dAkk = tl.load(p_dAkk, boundary_check=bc)
                b_kg = b_k * exp2(b_gm - b_gk.to(tl.float32))
                b_dq2 += tl.dot(b_dAqk.to(tl.float32), b_kg.to(tl.float32), allow_tf32=False)
                b_dk2 += tl.dot(b_dAkk.to(tl.float32), b_kg.to(tl.float32), allow_tf32=False)
            # Out-of-range rows load g as 0; clamp the row factor to 0 there so
            # exp2(0 - g_mid) can never produce inf * 0 = NaN (stores skip
            # those rows anyway).
            o_i = tl.arange(0, BC)
            m_t_valid = (i_ti + o_i[:, None]) < T_cur
            b_gqm = tl.where(m_t_valid, exp2(b_g - b_gm), 0.)
            b_dq2 *= b_gqm
            b_dk2 *= b_gqm
            p_q = tl.make_block_ptr(q_ptr, (T_cur, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=bc)
            p_k = tl.make_block_ptr(k_ptr, (T_cur, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=bc)
            p_b = tl.make_block_ptr(b_ptr, (T_cur, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            b_b = tl.load(p_b, boundary_check=bc).to(tl.float32)
            p_dq2 = tl.make_block_ptr(dq2_ptr, (T_cur, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            p_dk2 = tl.make_block_ptr(dk2_ptr, (T_cur, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            p_dg2 = tl.make_block_ptr(dg2_ptr, (T_cur, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            p_db = tl.make_block_ptr(db_ptr, (T_cur, BK), (H * BK, 1), (i_ti, 0), (BC, BK), (1, 0))
            # Accumulate onto the fused diag kernel's results (same stream,
            # fused launched first): dq2 += past, dk2 += past*b, dg2 += q*past
            # + (past*b)*k (the k-term is split out of the old dk_dg update,
            # which ran after this kernel), db += k*past.
            b_past_kb = b_dk2 * b_b
            b_db = b_dk2 * b_k + tl.load(p_db, boundary_check=bc)
            b_dk2 = b_past_kb + tl.load(p_dk2, boundary_check=bc)
            b_dg2 = b_q * b_dq2 + b_past_kb * b_k + tl.load(p_dg2, boundary_check=bc)
            b_dq2 = b_dq2 + tl.load(p_dq2, boundary_check=bc)
            tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), boundary_check=bc)
            tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), boundary_check=bc)
            tl.store(p_dg2, b_dg2.to(p_dg2.dtype.element_ty), boundary_check=bc)
            tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=bc)


@triton.jit(do_not_specialize=['B', 'T', 'NT', 'BH_TOTAL', 'task_num', 'num_core'])
def chunk_gdn2_bwd_kernel_intra_diag_fused_npu(
    q, k, g, b, dAqk, dAkk, dq, dq2, dk, dk2, dg, dg2, db, dkt_part,
    cu_seqlens, chunk_indices, B, T, NT, BH_TOTAL, task_num, num_core,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
    BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
    FULL_TILE: tl.constexpr,
):
    # Fused diagonal half of the bwd intra chain: the old dq_db_diag kernel
    # and the diagonal part of the old dk_dg kernel share every load
    # (q/k/g/b/dAqk/dAkk), the same mid-row decay reference, and the same
    # task grid, so one pass keeps the diag dot results in registers and
    # removes the dk2/dg2 round-trip through GM. It runs after dkt_future
    # (reads dkt_part) and before dq_past, which adds the past-block terms:
    #   dk2 = (diag + dk_in + dkt) + past*b
    #   dg2 = q*dq2_diag + (diag - dkt)*k + dg_in + q*past + (past*b)*k
    # Straight-line (no dynamic loop): lower-triangular dA tiles run on Cube
    # via masked tl.dot; a mid-row reference (clamped in range) splits the
    # decay into two non-positive-exponent factors, so no overflow path
    # exists inside the triangle.
    core_id = tl.program_id(0)
    g_row_stride = _bwd_intra_g_row_stride(G_T_CONTIG, K, H)
    o_i = tl.arange(0, BC)
    # Every tile is in range (non-varlen, T % BT == 0, BT % BC == 0,
    # K % BK == 0): drop the per-element boundary predicates, which codegen
    # as scalar address checks on the vector core.
    bc: tl.constexpr = () if FULL_TILE else (0, 1)
    for task_id in tl.range(core_id, task_num, num_core):
        i_bh = task_id % BH_TOTAL
        rem = task_id // BH_TOTAL
        i_t = rem % NT
        i_kc = rem // NT
        i_b, i_h = i_bh // H, i_bh % H
        i_k, i_i = i_kc // NC, i_kc % NC
        T_seq = T
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        else:
            bos = tl.cast(i_b, tl.int64) * T
            eos = bos + T
        T_cur = (eos - bos).to(tl.int32)
        i_ti = i_t * BT + i_i * BC
        if i_ti < T_cur:
            all = tl.cast(B, tl.int64) * T
            q_ptr = q + (bos * H + i_h).to(tl.int64) * K
            k_ptr = k + (bos * H + i_h).to(tl.int64) * K
            b_ptr = b + (bos * H + i_h).to(tl.int64) * K
            g_base = _bwd_intra_g_base(g, bos, i_b, i_h, T_seq, K, H, IS_VARLEN, G_T_CONTIG)
            dAqk_ptr = dAqk + (bos * H + i_h).to(tl.int64) * BT
            dAkk_ptr = dAkk + (bos * H + i_h).to(tl.int64) * BT
            dq_ptr = dq + (bos * H + i_h).to(tl.int64) * K
            dq2_ptr = dq2 + (bos * H + i_h).to(tl.int64) * K
            dk_ptr = dk + (bos * H + i_h).to(tl.int64) * K
            dk2_ptr = dk2 + (bos * H + i_h).to(tl.int64) * K
            dg_ptr = dg + (bos * H + i_h).to(tl.int64) * K
            dg2_ptr = dg2 + (bos * H + i_h).to(tl.int64) * K
            dkt_part_ptr = dkt_part + (bos * H + i_h).to(tl.int64) * K
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            db_ptr = db + ((i_k * all + bos) * H + i_h).to(tl.int64) * BK
            # Loads shared by both halves.
            p_g = _bwd_intra_g_block_ptr(g_base, T_cur, i_ti, i_k * BK, BC, BK, g_row_stride, K)
            b_g = tl.load(p_g, boundary_check=bc).to(tl.float32)
            p_b = tl.make_block_ptr(b_ptr, (T_cur, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            b_b = tl.load(p_b, boundary_check=bc).to(tl.float32)
            p_k = tl.make_block_ptr(k_ptr, (T_cur, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=bc)
            p_q = tl.make_block_ptr(q_ptr, (T_cur, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=bc)
            # Mid-row reference: both exp2(g_mid - g_j) and exp2(g_i - g_mid)
            # are bounded by half a sub-chunk of decay inside the triangle;
            # clamp to 0 on out-of-range rows so masked stores never see NaN.
            i_mid = i_ti + tl.minimum(BC // 2, T_cur - i_ti - 1)
            p_gm = g_base + i_mid.to(tl.int64) * g_row_stride + o_k
            b_gm = tl.load(p_gm, mask=m_k, other=0).to(tl.float32)[None, :]
            m_t_valid = (i_ti + o_i[:, None]) < T_cur
            b_e_neg = tl.where(m_t_valid, exp2(b_gm - b_g), 0.)
            b_e_pos = tl.where(m_t_valid, exp2(b_g - b_gm), 0.)
            p_dAqk = tl.make_block_ptr(dAqk_ptr, (T_cur, BT), (H * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
            p_dAkk = tl.make_block_ptr(dAkk_ptr, (T_cur, BT), (H * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
            b_dAqk = tl.load(p_dAqk, boundary_check=bc).to(tl.float32)
            b_dAkk = tl.load(p_dAkk, boundary_check=bc).to(tl.float32)
            m_i_diag = (o_i[:, None] >= o_i[None, :]) & (
                (i_ti + o_i[:, None]) < T_cur) & ((i_ti + o_i[None, :]) < T_cur)
            b_dAqk = tl.where(m_i_diag, b_dAqk, 0.)
            b_dAkk = tl.where(m_i_diag, b_dAkk, 0.)
            # Transpose before any dot: on Ascend tl.dot clobbers its left
            # operand, so a post-dot trans would read corrupted data.
            b_dAqk_t = tl.trans(b_dAqk)
            b_dAkk_t = tl.trans(b_dAkk)
            # Diagonal-block dq/dk via lower-triangular dots.
            b_kg = b_k * b_e_neg
            b_dq2 = tl.dot(b_dAqk, b_kg.to(tl.float32), allow_tf32=False) * b_e_pos
            b_dk2 = tl.dot(b_dAkk, b_kg.to(tl.float32), allow_tf32=False) * b_e_pos
            # GDN-2: db is channel-wise [BC, BK], not a row sum; dk2 scales by the b-tile.
            b_db = b_dk2 * b_k
            b_dk2 *= b_b
            b_dg2 = b_q.to(tl.float32) * b_dq2
            # Diagonal contribution to dkt: transposed dA tiles on Cube; the
            # write gate k carries the channel-wise erase gate b.
            b_dkt = tl.dot(b_dAqk_t, b_q * b_e_pos, allow_tf32=False) * b_e_neg
            b_dkt += tl.dot(b_dAkk_t, b_k * b_b * b_e_pos, allow_tf32=False) * b_e_neg
            p_dkt_part = tl.make_block_ptr(dkt_part_ptr, (T_cur, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            b_dkt += tl.load(p_dkt_part, boundary_check=bc).to(tl.float32)
            p_dq = tl.make_block_ptr(dq_ptr, (T_cur, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk_ptr, (T_cur, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            p_dg = tl.make_block_ptr(dg_ptr, (T_cur, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            b_dq2 += tl.load(p_dq, boundary_check=bc)
            b_dg2 += (b_dk2 - b_dkt) * b_k + tl.load(p_dg, boundary_check=bc)
            b_dk2 += tl.load(p_dk, boundary_check=bc) + b_dkt
            p_dq2 = tl.make_block_ptr(dq2_ptr, (T_cur, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            p_dk2 = tl.make_block_ptr(dk2_ptr, (T_cur, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            p_dg2 = tl.make_block_ptr(dg2_ptr, (T_cur, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            p_db = tl.make_block_ptr(db_ptr, (T_cur, BK), (H * BK, 1), (i_ti, 0), (BC, BK), (1, 0))
            tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), boundary_check=bc)
            tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), boundary_check=bc)
            tl.store(p_dg2, b_dg2.to(p_dg2.dtype.element_ty), boundary_check=bc)
            tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=bc)


@triton.jit(do_not_specialize=['T', 'NT', 'BH_TOTAL', 'task_num', 'num_core'])
def chunk_gdn2_bwd_kernel_intra_dkt_future_npu(
    q, k, g, b, dAqk, dAkk, dkt_part,
    cu_seqlens, chunk_indices, T, NT, BH_TOTAL, task_num, num_core,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
    BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr, NC_FUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
    FULL_TILE: tl.constexpr,
):
    core_id = tl.program_id(0)
    g_row_stride = _bwd_intra_g_row_stride(G_T_CONTIG, K, H)
    bc: tl.constexpr = () if FULL_TILE else (0, 1)
    o_i = tl.arange(0, BC)
    for task_id in tl.range(core_id, task_num, num_core):
        i_bh = task_id % BH_TOTAL
        rem = task_id // BH_TOTAL
        i_t = rem % NT
        i_kc = rem // NT
        i_b, i_h = i_bh // H, i_bh % H
        i_k, i_i = i_kc // NC_FUT, i_kc % NC_FUT
        T_seq = T
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        else:
            bos = tl.cast(i_b, tl.int64) * T
            eos = bos + T
        T_cur = (eos - bos).to(tl.int32)
        i_ti = i_t * BT + i_i * BC
        if i_ti < T_cur:
            q_ptr = q + (bos * H + i_h).to(tl.int64) * K
            k_ptr = k + (bos * H + i_h).to(tl.int64) * K
            b_ptr = b + (bos * H + i_h).to(tl.int64) * K
            g_base = _bwd_intra_g_base(g, bos, i_b, i_h, T_seq, K, H, IS_VARLEN, G_T_CONTIG)
            dAqk_ptr = dAqk + (bos * H + i_h).to(tl.int64) * BT
            dAkk_ptr = dAkk + (bos * H + i_h).to(tl.int64) * BT
            dkt_part_ptr = dkt_part + (bos * H + i_h).to(tl.int64) * K
            nc_eff = min(NC, tl.cdiv(T_cur - i_t * BT, BC))
            o_k = i_k * BK + tl.arange(0, BK)
            p_g = _bwd_intra_g_block_ptr(g_base, T_cur, i_ti, i_k * BK, BC, BK, g_row_stride, K)
            b_g = tl.load(p_g, boundary_check=bc).to(tl.float32)
            b_dkt = tl.zeros([BC, BK], dtype=tl.float32)
            if i_i < nc_eff - 1:
                p_gn = g_base + (min(i_ti + BC, T_cur) - 1).to(tl.int64) * g_row_stride + o_k
                b_gn = tl.load(p_gn, mask=o_k < K, other=0).to(tl.float32)[None, :]
                for i_j in range(i_i + 1, nc_eff):
                    p_q = tl.make_block_ptr(q_ptr, (T_cur, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
                    p_kj = tl.make_block_ptr(k_ptr, (T_cur, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
                    p_gk = _bwd_intra_g_block_ptr(g_base, T_cur, i_t * BT + i_j * BC, i_k * BK, BC, BK, g_row_stride, K)
                    p_bj = tl.make_block_ptr(b_ptr, (T_cur, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
                    p_dAqk = tl.make_block_ptr(dAqk_ptr, (T_cur, BT), (H * BT, 1),
                                               (i_t * BT + i_j * BC, i_i * BC), (BC, BC), (1, 0))
                    p_dAkk = tl.make_block_ptr(dAkk_ptr, (T_cur, BT), (H * BT, 1),
                                               (i_t * BT + i_j * BC, i_i * BC), (BC, BC), (1, 0))
                    b_bj = tl.load(p_bj, boundary_check=bc).to(tl.float32)
                    b_qj = tl.load(p_q, boundary_check=bc)
                    b_kj = tl.load(p_kj, boundary_check=bc)
                    b_gk = tl.load(p_gk, boundary_check=bc).to(tl.float32)
                    b_dAqk = tl.trans(tl.load(p_dAqk, boundary_check=bc).to(tl.float32))
                    b_dAkk = tl.trans(tl.load(p_dAkk, boundary_check=bc).to(tl.float32))
                    o_j = i_t * BT + i_j * BC + o_i
                    m_j = o_j < T_cur
                    b_gkn = exp2(b_gk - b_gn)
                    b_qg = b_qj * tl.where(m_j[:, None], b_gkn, 0)
                    # GDN-2: channel-wise b_j replaces KDA's scalar beta row.
                    b_kbg = b_kj * b_bj * tl.where(m_j[:, None], b_gkn, 0)
                    b_dkt += tl.dot(b_dAqk, b_qg.to(tl.float32), allow_tf32=False)
                    b_dkt += tl.dot(b_dAkk, b_kbg.to(tl.float32), allow_tf32=False)
                b_dkt *= exp2(b_gn - b_g)
            p_dkt_part = tl.make_block_ptr(dkt_part_ptr, (T_cur, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            tl.store(p_dkt_part, b_dkt.to(p_dkt_part.dtype.element_ty), boundary_check=bc)


@input_guard
def chunk_gdn2_bwd_intra_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
):
    B, T, H, K = k.shape
    BT = chunk_size
    BC = min(_BWD_INTRA_BC, BT)
    # All intra-bwd halves write fp32 dk2/dg2/db2; tile for fp32 I/O even if dq is bf16.
    elem = max(dq.element_size(), 4)
    BK = _get_bwd_intra_bk(K, BC, element_size=elem)
    BK_dk = _get_bwd_intra_bk_dk(K, BC, element_size=elem)

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)
    NK_dk = triton.cdiv(K, BK_dk)
    is_varlen = cu_seqlens is not None
    # GDN-2 keeps a single head count, so the transposed gate is [B, H, T, K].
    g_arg, g_t_contig = _gk_npu_arg(g, H)

    dq2 = torch.empty_like(dq)
    # Stream-ordered: dq_db writes past+diag here; dk_dg loads then stores in place.
    # fp32 until the final store so dkt is fused without an intermediate bf16 round.
    dk2 = torch.empty_like(dk, dtype=torch.float)
    dg2 = torch.empty_like(dg, dtype=torch.float)
    # Last subchunk has no future blocks; keep zeros and skip those tasks.
    dkt_part = torch.zeros_like(dk, dtype=torch.float)
    # GDN-2 db is channel-wise [.., K]; partials are [NK, B, T, H, BK] tiles.
    db2 = q.new_empty(NK, B, T, H, BK, dtype=torch.float32)

    bh_total = B * H
    # dq_past skips sub-chunk 0 (no past blocks) — start from i_i=1.
    task_num_past = NK * (NC - 1) * NT * bh_total if NC > 1 else 0
    task_num_diag = NK * NC * NT * bh_total
    nc_fut = max(NC - 1, 1)
    task_num_fut = NK_dk * nc_fut * NT * bh_total
    common_launch = dict(
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        NT=NT,
        BH_TOTAL=bh_total,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
        IS_VARLEN=is_varlen,
        G_T_CONTIG=g_t_contig,
    )
    # Fused diag kernel tiles are provably in range when everything divides
    # evenly — skips the per-element boundary predicates. The dynamic-loop
    # kernels (dq_past/dkt_future) regress with boundary checks removed
    # (BiSheng codegen), so they keep FULL_TILE=False.
    full_tile = (not is_varlen) and T % BT == 0 and BT % BC == 0 and K % BK == 0

    # dkt_future first (the fused diag kernel reads dkt_part), then the
    # fused diagonal kernel (straight-line: diag dq/dk/dg/db + the diagonal
    # dkt contribution in one pass), then dq_past (dynamic loop, adds the
    # past-block terms on top) — same stream, so ordering is guaranteed and
    # no buffer initialization is needed.
    if NC > 1:
        _launch_bwd_intra_core_grid(
            chunk_gdn2_bwd_kernel_intra_dkt_future_npu,
            task_num=task_num_fut,
            kernel_kwargs=dict(
                q=q,
                k=k,
                g=g_arg,
                b=b,
                dAqk=dAqk,
                dAkk=dAkk,
                dkt_part=dkt_part,
                NC_FUT=nc_fut,
                # dynamic loop: removing boundary checks regresses codegen
                FULL_TILE=False,
                **{**common_launch, 'BK': BK_dk},
            ),
        )
    _launch_bwd_intra_core_grid(
        chunk_gdn2_bwd_kernel_intra_diag_fused_npu,
        task_num=task_num_diag,
        kernel_kwargs=dict(
            q=q,
            k=k,
            g=g_arg,
            b=b,
            dAqk=dAqk,
            dAkk=dAkk,
            dq=dq,
            dq2=dq2,
            dk=dk,
            dk2=dk2,
            dg=dg,
            dg2=dg2,
            db=db2,
            dkt_part=dkt_part,
            B=B,
            FULL_TILE=full_tile,
            **common_launch,
        ),
    )
    # dq_past only covers sub-chunks with at least one past block: decode
    # i_i from a 1-based grid so task_num stays divisible across cores.
    if task_num_past > 0:
        _launch_bwd_intra_core_grid(
            chunk_gdn2_bwd_kernel_intra_dq_past_npu,
            task_num=task_num_past,
            kernel_kwargs=dict(
                q=q,
                k=k,
                g=g_arg,
                b=b,
                dAqk=dAqk,
                dAkk=dAkk,
                dq=dq,
                dq2=dq2,
                dk2=dk2,
                dg2=dg2,
                db=db2,
                B=B,
                # dynamic loop: removing boundary checks regresses codegen
                FULL_TILE=False,
                **{**common_launch, 'NC': NC - 1},
            ),
        )
    dq = dq2
    dk = dk2
    db2_combined = db2.permute(1, 2, 3, 0, 4).contiguous().reshape(B, T, H, NK * BK)[..., :K]
    db = db.add_(db2_combined)
    dg = dg2

    return dq, dk, db, dg
