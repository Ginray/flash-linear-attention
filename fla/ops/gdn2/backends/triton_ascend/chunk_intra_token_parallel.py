# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GDN2 token-parallel intra-chunk kernel for Triton-Ascend."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import ASCEND_MAX_GRID_DIM, max_grid_axis_chunks

_BH = 4


def _launch_token_parallel_kernel(kernel, *, tg_total: int, h_total: int, kernel_kwargs: dict) -> None:
    hg_total = triton.cdiv(h_total, _BH)
    max_tg = max_grid_axis_chunks(tg_total, hg_total, max_grid=ASCEND_MAX_GRID_DIM)
    for tg_off in range(0, tg_total, max_tg):
        tg_len = min(max_tg, tg_total - tg_off)
        max_hg = max_grid_axis_chunks(hg_total, tg_len, max_grid=ASCEND_MAX_GRID_DIM)
        for hg_off in range(0, hg_total, max_hg):
            hg_len = min(max_hg, hg_total - hg_off)
            kernel[(tg_len, hg_len)](TG_OFFSET=tg_off, HG_OFFSET=hg_off, **kernel_kwargs)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T', 'N'])
def chunk_gdn2_fwd_kernel_intra_token_parallel_npu(
    q,
    k,
    g,
    b,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    N,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    TG_OFFSET: tl.constexpr,
    HG_OFFSET: tl.constexpr,
):
    i_tg = tl.cast(tl.program_id(0) + TG_OFFSET, tl.int64)
    i_hg = tl.program_id(1) + HG_OFFSET

    if IS_VARLEN:
        left, right = 0, N
        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                if i_tg < tl.load(cu_seqlens + mid + 1).to(tl.int64):
                    right = mid
                else:
                    left = mid + 1
        i_n = left
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_cur = (eos - bos).to(tl.int32)
        i_t = (i_tg - bos).to(tl.int32)
    else:
        bos = (i_tg // T) * T
        T_cur = T
        i_t = (i_tg % T).to(tl.int32)

    if i_t < T_cur:
        i_c = i_t // BT
        i_s = (i_t % BT) // BC
        i_ts = i_c * BT + i_s * BC

        q += bos * H * K
        k += bos * H * K
        g += bos * H * K
        b += bos * H * K
        Aqk += bos * H * BT
        Akk += bos * H * BC

        o_h = i_hg * BH + tl.arange(0, BH)
        o_k = tl.arange(0, BK)
        m_h = o_h < H
        m_k = o_k < K
        m_hk = m_h[:, None] & m_k[None, :]
        p_hk = o_h[:, None] * K + o_k[None, :]

        b_q = tl.load(q + tl.cast(i_t, tl.int64) * H * K + p_hk, mask=m_hk, other=0).to(tl.float32)
        b_k = tl.load(k + tl.cast(i_t, tl.int64) * H * K + p_hk, mask=m_hk, other=0).to(tl.float32)
        b_g = tl.load(g + tl.cast(i_t, tl.int64) * H * K + p_hk, mask=m_hk, other=0).to(tl.float32)
        b_b = tl.load(b + tl.cast(i_t, tl.int64) * H * K + p_hk, mask=m_hk, other=0).to(tl.float32)
        b_bk = b_b * b_k

        for j in range(i_ts, min(i_t + 1, min(T_cur, i_ts + BC))):
            b_kj = tl.load(k + tl.cast(j, tl.int64) * H * K + p_hk, mask=m_hk, other=0).to(tl.float32)
            b_gj = tl.load(g + tl.cast(j, tl.int64) * H * K + p_hk, mask=m_hk, other=0).to(tl.float32)
            b_kgj = tl.where(m_k[None, :], b_kj * exp2(b_g - b_gj), 0)
            b_Aqk = tl.sum(b_q * b_kgj, axis=1) * scale
            b_Akk = tl.sum(b_bk * b_kgj, axis=1) * tl.where(j < i_t, 1.0, 0.0)

            tl.store(
                Aqk + tl.cast(i_t, tl.int64) * H * BT + o_h * BT + j % BT,
                b_Aqk.to(Aqk.dtype.element_ty),
                mask=m_h,
            )
            tl.store(
                Akk + tl.cast(i_t, tl.int64) * H * BC + o_h * BC + j - i_ts,
                b_Akk.to(Akk.dtype.element_ty),
                mask=m_h,
            )


@input_guard
def chunk_gdn2_fwd_intra_token_parallel_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    b: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K = q.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BT = chunk_size
    BC = sub_chunk_size
    BK = triton.next_power_of_2(K)

    _launch_token_parallel_kernel(
        chunk_gdn2_fwd_kernel_intra_token_parallel_npu,
        tg_total=B * T,
        h_total=H,
        kernel_kwargs=dict(
            q=q,
            k=k,
            g=gk,
            b=b,
            Aqk=Aqk,
            Akk=Akk,
            scale=scale,
            cu_seqlens=cu_seqlens,
            N=N,
            T=T,
            H=H,
            K=K,
            BK=BK,
            BT=BT,
            BC=BC,
            BH=_BH,
        ),
    )
    return Aqk, Akk
