# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend backend for GDN2."""

from __future__ import annotations

import torch

from fla.ops.backends import BaseBackend

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_CHUNK_SIZE = 64


def _get_tensor(args, kwargs, name: str, position: int) -> torch.Tensor | None:
    if name in kwargs:
        return kwargs[name]
    if len(args) > position:
        return args[position]
    return None


def _verify_npu_input(args, kwargs, name: str = 'q', position: int = 0) -> tuple[bool, str | None]:
    tensor = _get_tensor(args, kwargs, name, position)
    if tensor is None:
        return False, f"missing required tensor `{name}`"
    if tensor.device.type != 'npu':
        return False, f"`{name}` must be on NPU, got {tensor.device.type}"
    if tensor.dtype not in _SUPPORTED_DTYPES:
        return False, f"unsupported GDN2 Ascend dtype {tensor.dtype}"
    return True, None


def _verify_chunk(args, kwargs, name: str = 'q', position: int = 0) -> tuple[bool, str | None]:
    accepted, reason = _verify_npu_input(args, kwargs, name, position)
    if not accepted:
        return accepted, reason
    chunk_size = kwargs.get('chunk_size', _CHUNK_SIZE)
    if chunk_size != _CHUNK_SIZE:
        return False, f"GDN2 Ascend only supports chunk_size={_CHUNK_SIZE}, got {chunk_size}"
    return True, None


class TritonAscendGDN2Backend(BaseBackend):
    """Ascend NPU backend for GDN2 chunk training and recurrent inference."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def chunk_gdn2_fwd_intra_token_parallel_verifier(self, *args, **kwargs):
        return _verify_chunk(args, kwargs)

    def chunk_gdn2_fwd_intra_token_parallel(self, *args, **kwargs):
        from fla.ops.gdn2.backends.triton_ascend.chunk_intra_token_parallel import (
            chunk_gdn2_fwd_intra_token_parallel_npu,
        )
        return chunk_gdn2_fwd_intra_token_parallel_npu(*args, **kwargs)

    def recompute_w_u_fwd_gdn2_verifier(self, *args, **kwargs):
        return _verify_npu_input(args, kwargs, name='k')

    def recompute_w_u_fwd_gdn2(self, *args, **kwargs):
        from fla.ops.gdn2.backends.triton_ascend.wy_fast import recompute_w_u_fwd_gdn2_npu
        return recompute_w_u_fwd_gdn2_npu(*args, **kwargs)

    def chunk_gdn2_fwd_intra_verifier(self, *args, **kwargs):
        return _verify_chunk(args, kwargs)

    def chunk_gdn2_fwd_intra(self, *args, **kwargs):
        from fla.ops.gdn2.backends.triton_ascend.chunk_intra import chunk_gdn2_fwd_intra_npu
        return chunk_gdn2_fwd_intra_npu(*args, **kwargs)

    def chunk_gdn2_bwd_intra_verifier(self, *args, **kwargs):
        return _verify_chunk(args, kwargs)

    def chunk_gdn2_bwd_intra(self, *args, **kwargs):
        from fla.ops.gdn2.backends.triton_ascend.chunk_intra import chunk_gdn2_bwd_intra_npu
        return chunk_gdn2_bwd_intra_npu(*args, **kwargs)

    def chunk_gdn2_bwd_wy_dqkg_fused_verifier(self, *args, **kwargs):
        return _verify_chunk(args, kwargs)

    def chunk_gdn2_bwd_wy_dqkg_fused(self, *args, **kwargs):
        from fla.ops.gdn2.backends.triton_ascend.chunk_bwd import chunk_gdn2_bwd_wy_dqkg_fused_npu
        return chunk_gdn2_bwd_wy_dqkg_fused_npu(*args, **kwargs)

    def fused_recurrent_gdn2_fwd_verifier(self, *args, **kwargs):
        return _verify_npu_input(args, kwargs)

    def fused_recurrent_gdn2_fwd(self, *args, **kwargs):
        from fla.ops.gdn2.backends.triton_ascend.fused_recurrent import fused_recurrent_gdn2_fwd_npu
        return fused_recurrent_gdn2_fwd_npu(*args, **kwargs)
