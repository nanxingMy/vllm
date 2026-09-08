#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
venv_dir=${VLLM_VENV:-"$repo_dir/.venv"}
if [[ "$venv_dir" != /* ]]; then
    venv_dir="$PWD/$venv_dir"
fi
uv_bin=${UV_BIN:-uv}
export UV_CACHE_DIR=${UV_CACHE_DIR:-"$repo_dir/../.cache/uv"}
if ! command -v "$uv_bin" >/dev/null 2>&1; then
    echo "Install uv first, or set UV_BIN to its executable path." >&2
    exit 1
fi
cd -- "$repo_dir"
if [[ ! -x "$venv_dir/bin/python" ]]; then
    "$uv_bin" venv --python "${PYTHON_VERSION:-3.12}" "$venv_dir"
fi

# Python-only patch: use extensions from this exact upstream tag, not main.
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_COMMIT=${VLLM_PRECOMPILED_WHEEL_COMMIT:-d2906cc1958658f296aeee8b248deea684f56add}
"$uv_bin" pip install --python "$venv_dir/bin/python" -e . --torch-backend=auto
"$uv_bin" pip install --python "$venv_dir/bin/python" -r requirements/lint.txt
"$venv_dir/bin/pre-commit" install
if [[ ${INSTALL_TEST_DEPS:-0} == 1 ]]; then
    "$uv_bin" pip install --python "$venv_dir/bin/python" -r requirements/test/cuda.in
fi
"$venv_dir/bin/python" - <<'PY'
import torch
import vllm
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.flashattn_mla_sparse import (
    FlashAttnMLASparseMetadataBuilder,
)

print("vLLM:", vllm.__version__, vllm.__file__)
assert torch.cuda.is_available(), "CUDA is not available"
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index))
    assert torch.cuda.get_device_capability(index)[0] == 9, "Expected SM90 GPU"
assert FlashAttnMLASparseMetadataBuilder.get_cudagraph_support(None, None) == AttentionCGSupport.ALWAYS
PY
