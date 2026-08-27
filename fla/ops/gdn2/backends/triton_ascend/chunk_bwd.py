# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GDN-2 chunk backward WY kernels for triton-ascend on Ascend NPU.

Mirrors ``chunk_gdn2_bwd_kernel_wy_dqkg_fused`` from
``fla/ops/gdn2/chunk_bwd.py``, split into four NPU-friendly kernels that
exchange partials through global memory (``dA_acc``):

  v_part:      dv2 = (A^T dv) * w_gate; dw = (A^T dv) * v; dA_acc = dv (v w_gate)^T
  k_part:      dq, dk(partial), dg(partial) over K-slabs
  dw_part:     db, dk +=, dg +=, dA_acc += -(dv h)(b e^g k)^T
  dA_finalize: dA = -mask(A (mask dA_acc) A)

GDN-2 twists vs KDA: gates b (key axis) / w_gate (value axis) are
channel-wise, so dA bakes them into both GEMM operands, db is a [BT, BK]
tile written directly (no cross-slab reduction), and there is no beta to
row-scale in the finalize.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.runtime import driver

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import compute_row_tile_block_size

_BC = 16
_BWD_MEM_MULT = 10.0
_SAFETY_MARGIN = 0.80
_FALLBACK_TILE = 16
_MAX_TILE = 128


def _get_bk(K: int) -> int:
    return compute_row_tile_block_size(
        _BC,
        K,
        _BWD_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_TILE,
        min_block=16,
        max_block=min(_MAX_TILE, triton.next_power_of_2(K)),
    )


def _get_bv(V: int) -> int:
    return compute_row_tile_block_size(
        _BC,
        V,
        _BWD_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_TILE,
        min_block=16,
        max_block=min(_MAX_TILE, triton.next_power_of_2(V)),
    )


def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH'])
def chunk_gdn2_bwd_kernel_wy_v_part_npu(
    v,
    w_gate,
    A,
    dv,
    dv2,
    dw,
    dA_acc,
    cu_seqlens,
    chunk_indices,
    T,
    BH,
    task_num,
    num_core,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    core_id = tl.program_id(0)

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // BH
        i_bh = task_id % BH
        i_b, i_h = i_bh // H, i_bh % H

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = (eos - bos).to(tl.int32)
        else:
            bos, eos = tl.cast(i_b, tl.int64) * T, tl.cast(i_b, tl.int64) * T + T

        v_ptr = v + (bos * H + i_h) * V
        wg_ptr = w_gate + (bos * H + i_h) * V
        A_ptr = A + (bos * H + i_h) * BT
        dv_ptr = dv + (bos * H + i_h) * V
        dv2_ptr = dv2 + (bos * H + i_h) * V
        dw_ptr = dw + (bos * H + i_h) * V
        dA_ptr = dA_acc + (bos * H + i_h) * BT

        # b_A[r, t] = A_mem[t, r], matching the CUDA kernel's transposed read.
        p_A = tl.make_block_ptr(A_ptr, (BT, T), (1, H * BT), (0, i_t * BT), (BT, BT), (0, 1))
        b_A = tl.load(p_A, boundary_check=(0, 1))

        b_dA = tl.zeros([BT, BT], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            p_dv = tl.make_block_ptr(dv_ptr, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_v = tl.make_block_ptr(v_ptr, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_wg = tl.make_block_ptr(wg_ptr, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_wg = tl.load(p_wg, boundary_check=(0, 1))
            # dA gets (w_gate * v) on the value side - the GDN-2 channel-wise twist.
            b_dA += tl.dot(b_dv, tl.trans(b_v * b_wg), allow_tf32=False)
            # Ascend tl.dot clobbers lhs; copy A before every V-slab use.
            b_A_c = b_A + 0.0
            b_dvb = tl.dot(b_A_c, b_dv, allow_tf32=False)

            p_dv2 = tl.make_block_ptr(dv2_ptr, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dw = tl.make_block_ptr(dw_ptr, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_dv2, (b_dvb * b_wg).to(p_dv2.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_dw, (b_dvb * b_v).to(p_dw.dtype.element_ty), boundary_check=(0, 1))

        p_dA = tl.make_block_ptr(dA_ptr, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH'])
def chunk_gdn2_bwd_kernel_wy_k_part_npu(
    q,
    k,
    v_new,
    g,
    h,
    do,
    dh,
    dq,
    dk,
    dg,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    scale,
    T,
    BH,
    task_num,
    num_core,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    K_OFFSET: tl.constexpr,
):
    i_k = K_OFFSET
    core_id = tl.program_id(0)

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // BH
        i_bh = task_id % BH
        i_b, i_h = i_bh // H, i_bh % H

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = (eos - bos).to(tl.int32)
            i_tg = tl.load(chunk_offsets + i_n).to(tl.int64) + i_t.to(tl.int64)
        else:
            i_tg = (i_b * tl.cdiv(T, BT) + i_t).to(tl.int64)
            bos, eos = tl.cast(i_b, tl.int64) * T, tl.cast(i_b, tl.int64) * T + T

        q_ptr = q + (bos * H + i_h) * K
        k_ptr = k + (bos * H + i_h) * K
        v_new_ptr = v_new + (bos * H + i_h) * V
        g_ptr = g + (bos * H + i_h) * K
        h_ptr = h + (i_tg * H + i_h) * K * V
        do_ptr = do + (bos * H + i_h) * V
        dh_ptr = dh + (i_tg * H + i_h) * K * V
        dq_ptr = dq + (bos * H + i_h) * K
        dk_ptr = dk + (bos * H + i_h) * K
        dg_ptr = dg + (bos * H + i_h) * K

        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_gn = g_ptr + (min(T, i_t * BT + BT) - 1).to(tl.int64) * H * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)

        o_i = tl.arange(0, BC)
        n_sub = BT // BC
        b_dgk = tl.zeros([BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(h_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                p_dh = tl.make_block_ptr(dh_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                p_dh = tl.make_block_ptr(dh_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))
            b_dgk += tl.sum(b_h * b_dh, axis=0)

        b_dgk *= exp2(b_gn)

        b_kdk_sum = tl.zeros([BK], dtype=tl.float32)
        for s in range(n_sub):
            i_tc_s = i_t * BT + s * BC
            m_s = (i_tc_s + o_i) < T

            p_k = tl.make_block_ptr(k_ptr, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            p_g = tl.make_block_ptr(g_ptr, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

            b_dk = tl.zeros([BC, BK], dtype=tl.float32)
            for i_v in range(tl.cdiv(V, BV)):
                p_v_new = tl.make_block_ptr(v_new_ptr, (T, V), (H * V, 1), (i_tc_s, i_v * BV), (BC, BV), (1, 0))
                if STATE_V_FIRST:
                    p_dh = tl.make_block_ptr(dh_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                else:
                    p_dh = tl.make_block_ptr(dh_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                b_v_new = tl.load(p_v_new, boundary_check=(0, 1))
                b_dh = tl.load(p_dh, boundary_check=(0, 1))
                b_dk += tl.dot(b_v_new, b_dh.to(b_v_new.dtype), allow_tf32=False)

            b_dk = b_dk * tl.where(m_s[:, None], exp2(b_gn[None, :] - b_g), 0)
            b_kdk_sum += tl.sum(b_k * b_dk, axis=0)
            p_dk = tl.make_block_ptr(dk_ptr, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

        b_dgk_total = b_dgk + b_kdk_sum

        for s in range(n_sub):
            i_tc_s = i_t * BT + s * BC
            m_last_s = (i_tc_s + o_i) == min(T, i_t * BT + BT) - 1

            p_k = tl.make_block_ptr(k_ptr, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            p_g = tl.make_block_ptr(g_ptr, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            p_q = tl.make_block_ptr(q_ptr, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk_ptr, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_dk = tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)

            b_dq = tl.zeros([BC, BK], dtype=tl.float32)
            for i_v in range(tl.cdiv(V, BV)):
                p_do = tl.make_block_ptr(do_ptr, (T, V), (H * V, 1), (i_tc_s, i_v * BV), (BC, BV), (1, 0))
                if STATE_V_FIRST:
                    p_h = tl.make_block_ptr(h_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                else:
                    p_h = tl.make_block_ptr(h_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                b_do = tl.load(p_do, boundary_check=(0, 1))
                b_h = tl.load(p_h, boundary_check=(0, 1))
                b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)

            b_dq = b_dq * exp2(b_g) * scale
            # dw_part later adds b_kg * b_dkgb * b_b to dg.
            b_dg = b_q * b_dq - b_k * b_dk + m_last_s[:, None] * b_dgk_total

            p_dq = tl.make_block_ptr(dq_ptr, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            p_dg = tl.make_block_ptr(dg_ptr, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH'])
def chunk_gdn2_bwd_kernel_wy_dw_part_npu(
    k,
    g,
    b,
    A,
    h,
    dv,
    dA_acc,
    db,
    dg,
    dk,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    T,
    BH,
    task_num,
    num_core,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    K_OFFSET: tl.constexpr,
):
    i_k = K_OFFSET
    core_id = tl.program_id(0)

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // BH
        i_bh = task_id % BH
        i_b, i_h = i_bh // H, i_bh % H

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = (eos - bos).to(tl.int32)
            i_tg = tl.load(chunk_offsets + i_n).to(tl.int64) + i_t.to(tl.int64)
        else:
            i_tg = (i_b * tl.cdiv(T, BT) + i_t).to(tl.int64)
            bos, eos = tl.cast(i_b, tl.int64) * T, tl.cast(i_b, tl.int64) * T + T

        k_ptr = k + (bos * H + i_h) * K
        g_ptr = g + (bos * H + i_h) * K
        b_ptr = b + (bos * H + i_h) * K
        A_ptr = A + (bos * H + i_h) * BT
        h_ptr = h + (i_tg * H + i_h) * K * V
        dv_ptr = dv + (bos * H + i_h) * V
        dA_ptr = dA_acc + (bos * H + i_h) * BT
        db_ptr = db + (bos * H + i_h) * K
        dg_ptr = dg + (bos * H + i_h) * K
        dk_ptr = dk + (bos * H + i_h) * K

        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            p_dv = tl.make_block_ptr(dv_ptr, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(h_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dw += tl.dot(b_dv, b_h.to(b_dv.dtype), allow_tf32=False)

        p_k = tl.make_block_ptr(k_ptr, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g_ptr, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_b = tl.make_block_ptr(b_ptr, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_A = tl.make_block_ptr(A_ptr, (BT, T), (1, H * BT), (0, i_t * BT), (BT, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_b = tl.load(p_b, boundary_check=(0, 1))
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_gk_exp = exp2(b_g)
        b_kg = b_k * b_gk_exp
        b_gb = b_gk_exp * b_b
        # Match CUDA: downcast dw to A.dtype before the dA / dkgb GEMMs.
        b_dw = -b_dw.to(b_A.dtype)
        # dA gets (b * exp(gk) * k) on the key side - the GDN-2 channel-wise twist.
        b_kg_b = (b_kg * b_b).to(b_A.dtype)

        p_dA_acc = tl.make_block_ptr(dA_ptr, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        b_dA = tl.load(p_dA_acc, boundary_check=(0, 1)).to(tl.float32)
        # Ascend tl.dot clobbers lhs; copy dw before the dA GEMM so the dkgb
        # GEMM below still sees the original value.
        b_dw_c = b_dw + 0.0
        b_dA += tl.dot(b_dw_c, tl.trans(b_kg_b), allow_tf32=False)
        tl.store(p_dA_acc, b_dA.to(p_dA_acc.dtype.element_ty), boundary_check=(0, 1))

        b_dkgb = tl.dot(b_A, b_dw, allow_tf32=False)

        # db is channel-wise [BT, BK]; each K-slab writes a disjoint tile.
        p_db = tl.make_block_ptr(db_ptr, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_db_partial = b_dkgb * b_kg
        tl.store(p_db, b_db_partial.to(p_db.dtype.element_ty), boundary_check=(0, 1))

        p_dk = tl.make_block_ptr(dk_ptr, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_dk = tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)
        b_dk = b_dk + b_dkgb * b_gb
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

        p_dg = tl.make_block_ptr(dg_ptr, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_dg = tl.load(p_dg, boundary_check=(0, 1)).to(tl.float32)
        b_dg = b_dg + b_kg * b_dkgb * b_b
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH', 'NT_OFFSET'])
def chunk_gdn2_bwd_kernel_wy_dA_finalize_npu(
    A,
    dA_acc,
    dA,
    cu_seqlens,
    chunk_indices,
    T,
    BH,
    task_num,
    num_core,
    NT_OFFSET,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    TAIL_MODE: tl.constexpr,
):
    """dA = mask(-A @ ((mask * dA_acc) @ A)).

    TAIL_MODE 0 = aligned bulk (no boundary_check). TAIL_MODE 1 = tail/varlen.
    First tl.dot clobbers masked dA (dead). Second uses b_A as lhs (dead after store).
    """
    core_id = tl.program_id(0)

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = NT_OFFSET + task_id // BH
        i_bh = task_id % BH
        i_b, i_h = i_bh // H, i_bh % H

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = (eos - bos).to(tl.int32)
        else:
            bos, eos = tl.cast(i_b, tl.int64) * T, tl.cast(i_b, tl.int64) * T + T

        A_ptr = A + (bos * H + i_h) * BT
        dA_acc_ptr = dA_acc + (bos * H + i_h) * BT
        dA_ptr = dA + (bos * H + i_h) * BT

        p_A = tl.make_block_ptr(A_ptr, (BT, T), (1, H * BT), (0, i_t * BT), (BT, BT), (0, 1))
        p_dA_acc = tl.make_block_ptr(dA_acc_ptr, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        p_dA = tl.make_block_ptr(dA_ptr, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))

        o_t = i_t * BT + tl.arange(0, BT)
        if TAIL_MODE == 0:
            b_A = tl.load(p_A)
            b_dA = tl.load(p_dA_acc).to(tl.float32)
            m_A = o_t[:, None] > o_t[None, :]
        else:
            b_A = tl.load(p_A, boundary_check=(0, 1))
            b_dA = tl.load(p_dA_acc, boundary_check=(0, 1)).to(tl.float32)
            m_t = o_t < T
            m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t[None, :])

        b_dA = tl.where(m_A, b_dA, 0)
        # mid: (mask * dA_acc) @ A. lhs clobbers b_dA; A is rhs then lhs.
        b_mid = tl.dot(b_dA.to(b_A.dtype), b_A, allow_tf32=False)
        b_fin = tl.dot(b_A, b_mid.to(b_A.dtype), allow_tf32=False)
        b_fin = tl.where(m_A, -b_fin, 0)

        if TAIL_MODE == 0:
            tl.store(p_dA, b_fin.to(p_dA.dtype.element_ty))
        else:
            tl.store(p_dA, b_fin.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


def _launch_wy_dA_finalize(
    kernel,
    *,
    nt: int,
    bh_total: int,
    T: int,
    BT: int,
    is_varlen: bool,
    num_core: int,
    kernel_kwargs: dict,
) -> None:
    """Host-split aligned bulk vs tail so TAIL_MODE is constexpr per launch."""
    kwargs = dict(kernel_kwargs)
    kwargs['num_core'] = num_core
    if is_varlen:
        kwargs['TAIL_MODE'] = 1
        kwargs['NT_OFFSET'] = 0
        kwargs['task_num'] = nt * bh_total
        kernel[(num_core,)](**kwargs)
        return
    n_bulk = nt if T % BT == 0 else max(nt - 1, 0)
    if n_bulk > 0:
        kwargs['TAIL_MODE'] = 0
        kwargs['NT_OFFSET'] = 0
        kwargs['task_num'] = n_bulk * bh_total
        kernel[(num_core,)](**kwargs)
    if T % BT != 0 and nt > 0:
        kwargs['TAIL_MODE'] = 1
        kwargs['NT_OFFSET'] = n_bulk
        kwargs['task_num'] = bh_total
        kernel[(num_core,)](**kwargs)


@input_guard
def chunk_gdn2_bwd_wy_dqkg_fused_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w_gate: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    state_v_first: bool = False,
):
    """Fused WY backward producing dq, dk, dv, db (K-dim), dw (V-dim), dg, dA."""
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    if BT % _BC != 0:
        raise ValueError(f'GDN2 Ascend bwd requires chunk_size % {_BC} == 0, got {BT}')

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.empty_like(k, dtype=torch.float32)
    dv2 = torch.empty_like(v)
    dg = torch.empty_like(g, dtype=torch.float32)
    db = torch.empty_like(b, dtype=torch.float32)
    dw = torch.empty_like(w_gate, dtype=torch.float32)
    dA = torch.empty_like(A, dtype=torch.float32)
    dA_acc = torch.zeros(B, T, H, BT, dtype=torch.float32, device=A.device)

    BK = _get_bk(K)
    BV = _get_bv(V)
    NK = triton.cdiv(K, BK)
    is_varlen = cu_seqlens is not None
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT) if is_varlen else g.new_zeros(1, dtype=torch.int64)

    bh_total = B * H
    num_core = get_npu_properties()['num_vectorcore']
    task_num = NT * bh_total

    chunk_gdn2_bwd_kernel_wy_v_part_npu[(num_core,)](
        v=v,
        w_gate=w_gate,
        A=A,
        dv=dv,
        dv2=dv2,
        dw=dw,
        dA_acc=dA_acc,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        BH=bh_total,
        task_num=task_num,
        num_core=num_core,
        H=H,
        V=V,
        BT=BT,
        BV=BV,
        IS_VARLEN=is_varlen,
    )

    k_part_kwargs = dict(
        q=q,
        k=k,
        v_new=v_new,
        g=g,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        BH=bh_total,
        task_num=task_num,
        num_core=num_core,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=32 if BT >= 32 else _BC,
        BK=BK,
        BV=BV,
        STATE_V_FIRST=state_v_first,
        IS_VARLEN=is_varlen,
    )
    for k_off in range(NK):
        k_part_kwargs['K_OFFSET'] = k_off
        chunk_gdn2_bwd_kernel_wy_k_part_npu[(num_core,)](**k_part_kwargs)

    dw_kwargs = dict(
        k=k,
        g=g,
        b=b,
        A=A,
        h=h,
        dv=dv,
        dA_acc=dA_acc,
        db=db,
        dg=dg,
        dk=dk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        T=T,
        BH=bh_total,
        task_num=task_num,
        num_core=num_core,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        STATE_V_FIRST=state_v_first,
        IS_VARLEN=is_varlen,
    )
    for k_off in range(NK):
        dw_kwargs['K_OFFSET'] = k_off
        chunk_gdn2_bwd_kernel_wy_dw_part_npu[(num_core,)](**dw_kwargs)

    _launch_wy_dA_finalize(
        chunk_gdn2_bwd_kernel_wy_dA_finalize_npu,
        nt=NT,
        bh_total=bh_total,
        T=T,
        BT=BT,
        is_varlen=is_varlen,
        num_core=num_core,
        kernel_kwargs=dict(
            A=A,
            dA_acc=dA_acc,
            dA=dA,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            BH=bh_total,
            H=H,
            BT=BT,
            IS_VARLEN=is_varlen,
        ),
    )

    dv = dv2
    return dq, dk, dv, db, dw, dg, dA
