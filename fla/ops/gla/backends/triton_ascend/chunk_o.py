# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""chunk_gla_fwd_o_gk adapted for triton-ascend on Ascend NPU.

Consumes the precomputed intra-chunk score matrix ``A`` (GDN-2 / GLA style)
instead of recomputing q @ k^T, so the kernel is a pure output composition:
inter-chunk ``dot(q * exp2(g), h)`` plus intra-chunk ``dot(tril(A), v)``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import input_guard


def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


def _g_npu_arg(g: torch.Tensor, HV: int) -> tuple[torch.Tensor, bool]:
    """Transpose g [B, T, HV, K] to [B, HV, T, K] for stride-1 row loads along T."""
    if HV == 1:
        return g, False
    return g.transpose(1, 2).contiguous(), True


@triton.jit
def _g_block_ptr(g_base, T, K, i_t, i_k, BT, BK, G_T_CONTIG: tl.constexpr, HV: tl.constexpr):
    if G_T_CONTIG:
        return tl.make_block_ptr(g_base, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    return tl.make_block_ptr(g_base, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))


@triton.jit(do_not_specialize=["T", "B", "total_chunks", "task_num", "num_core"])
def chunk_gla_fwd_kernel_o_npu(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B,
    total_chunks,
    task_num,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
):
    T_max = T
    BH = B * HV
    bh_chunks = BH * total_chunks
    core_id = tl.program_id(0)

    for task_id in tl.range(core_id, task_num, num_core):
        # Flatten (i_v, i_bh, global chunk) into task_id, i_v-major.
        i_v = task_id // bh_chunks
        rem = task_id % bh_chunks
        i_bh = rem // total_chunks
        i_t_o = rem % total_chunks
        i_hv = i_bh % HV
        i_h = i_hv // (HV // H)

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t_o * 2).to(tl.int32), tl.load(
                chunk_indices + i_t_o * 2 + 1,
            ).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = (eos - bos).to(tl.int32)
            i_tg = i_t_o
        else:
            NT = tl.cdiv(T, BT)
            i_n = i_t_o // NT
            i_t = i_t_o % NT
            bos = tl.cast(i_n, tl.int64) * T
            i_tg = i_t_o

        q_ptr = q + (bos * H + i_h) * K
        v_ptr = v + (bos * HV + i_hv) * V
        o_ptr = o + (bos * HV + i_hv) * V
        A_ptr = A + (bos * HV + i_hv) * BT
        h_base = h + tl.cast(i_tg * HV + i_hv, tl.int64) * K * V

        if G_T_CONTIG:
            if IS_VARLEN:
                g_base = g + tl.cast(i_hv, tl.int64) * T_max * K + bos * K
            else:
                g_base = g + tl.cast(i_n * HV + i_hv, tl.int64) * T_max * K
        else:
            g_base = g + (bos * HV + i_hv) * K

        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(q_ptr, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            p_g = _g_block_ptr(g_base, T, K, i_t, i_k, BT, BK, G_T_CONTIG, HV)
            b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
            b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(h_base, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                b_h = tl.load(p_h, boundary_check=(0, 1))
                b_o += tl.dot(b_qg, tl.trans(b_h), allow_tf32=False)
            else:
                p_h = tl.make_block_ptr(h_base, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
                b_h = tl.load(p_h, boundary_check=(0, 1))
                b_o += tl.dot(b_qg, b_h, allow_tf32=False)
        b_o *= scale

        # A already carries the attention scale (baked in by the intra kernel).
        p_A = tl.make_block_ptr(A_ptr, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        b_A = tl.load(p_A, boundary_check=(0, 1))
        m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
        b_A = tl.where(m_s, b_A, 0.)

        p_v = tl.make_block_ptr(v_ptr, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_o += tl.dot(b_A.to(b_v.dtype), b_v, allow_tf32=False)

        p_o = tl.make_block_ptr(o_ptr, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def chunk_gla_fwd_o_gk_npu(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    if cu_seqlens is None:
        B_eff, chunk_indices_eff = B, None
        total_chunks = B * triton.cdiv(T, BT)
    else:
        B_eff, chunk_indices_eff = 1, chunk_indices
        total_chunks = len(chunk_indices)

    # Please ensure zeros, since vllm will use padding v.
    o = torch.zeros_like(v)

    g_t, g_t_contig = _g_npu_arg(g, HV)

    BK = min(128, triton.next_power_of_2(K))
    BV = 128
    NV = triton.cdiv(V, BV)
    num_core = get_npu_properties()["num_aicore"]
    task_num = NV * B_eff * HV * total_chunks

    chunk_gla_fwd_kernel_o_npu[(num_core,)](
        q=q,
        v=v,
        g=g_t,
        h=h,
        o=o,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices_eff,
        scale=scale,
        T=T,
        B=B_eff,
        total_chunks=total_chunks,
        task_num=task_num,
        num_core=num_core,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        STATE_V_FIRST=state_v_first,
        IS_VARLEN=cu_seqlens is not None,
        G_T_CONTIG=g_t_contig,
    )
    return o
