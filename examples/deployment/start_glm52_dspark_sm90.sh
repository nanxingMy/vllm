#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
venv_dir=${VLLM_VENV:-"$repo_dir/.venv"}
if [[ "$venv_dir" != /* ]]; then
    venv_dir="$PWD/$venv_dir"
fi
: "${MODEL_PATH:?Set MODEL_PATH to the SM90-compatible GLM-5.2 checkpoint}"
: "${DRAFT_MODEL_PATH:?Set DRAFT_MODEL_PATH to the DSpark checkpoint with a confidence head}"
if [[ ! -x "$venv_dir/bin/python" ]]; then
    echo "Missing environment: $venv_dir; run tools/install_glm52_dspark_sm90.sh" >&2
    exit 1
fi

export VLLM_USE_V2_MODEL_RUNNER=1
adaptive=${ADAPTIVE_VERIFICATION:-true}
speculative_config=$("$venv_dir/bin/python" - "$DRAFT_MODEL_PATH" "${NUM_SPECULATIVE_TOKENS:-7}" "$adaptive" <<'PY'
import json
import sys

if sys.argv[3] not in ("true", "false"):
    raise SystemExit("ADAPTIVE_VERIFICATION must be true or false")
print(json.dumps({
    "method": "dspark",
    "model": sys.argv[1],
    "attention_backend": "FLASH_ATTN",
    "num_speculative_tokens": int(sys.argv[2]),
    "draft_sample_method": "probabilistic",
    "enable_adaptive_verification": sys.argv[3] == "true",
}))
PY
)
command=(
    "$venv_dir/bin/python" -m vllm.entrypoints.cli.main serve "$MODEL_PATH"
    --served-model-name "${SERVED_MODEL_NAME:-glm-5.2-dspark}"
    --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
    --tensor-parallel-size "${TP_SIZE:-8}"
    --attention-backend FLASH_ATTN_MLA_SPARSE
    --kv-cache-dtype auto --dtype bfloat16 --block-size 64
    --max-model-len "${MAX_MODEL_LEN:-16384}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}"
    --max-num-seqs "${MAX_NUM_SEQS:-128}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
    --compilation-config "{\"max_cudagraph_capture_size\":${MAX_CUDAGRAPH_CAPTURE_SIZE:-2048}}"
    --speculative-config "$speculative_config"
)
if [[ ${ENABLE_EXPERT_PARALLEL:-true} == true ]]; then
    command+=(--enable-expert-parallel)
fi
command+=("$@")
printf '%q ' "${command[@]}"
printf '\n'
if [[ ${DRY_RUN:-0} == 1 ]]; then
    exit 0
fi
cd -- "$repo_dir"
exec "${command[@]}"
