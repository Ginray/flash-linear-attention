# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for GLA ops."""

from __future__ import annotations

import torch

from fla.ops.backends import BaseBackend

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _verify_npu_input(args, kwargs, name: str = 'q', position: int = 0) -> tuple[bool, str | None]:
    tensor = kwargs.get(name) if name in kwargs else (args[position] if len(args) > position else None)
    if tensor is None:
        return False, f"missing required tensor `{name}`"
    if tensor.device.type != 'npu':
        return False, f"`{name}` must be on NPU, got {tensor.device.type}"
    if tensor.dtype not in _SUPPORTED_DTYPES:
        return False, f"unsupported GLA Ascend dtype {tensor.dtype}"
    return True, None


class TritonAscendGLABackend(BaseBackend):
    """Ascend NPU backend for GLA forward output composition."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def chunk_gla_fwd_o_gk_verifier(self, *args, **kwargs):
        return _verify_npu_input(args, kwargs)

    def chunk_gla_fwd_o_gk(self, *args, **kwargs):
        from fla.ops.gla.backends.triton_ascend.chunk_o import chunk_gla_fwd_o_gk_npu
        return chunk_gla_fwd_o_gk_npu(*args, **kwargs)
