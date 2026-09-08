# GLM-5.2 adaptive DSpark on SM90

This branch is based on vLLM `v0.29.0rc4`
(`d2906cc1958658f296aeee8b248deea684f56add`). It extends the device-boundary
mapping used by [PR #52783](https://github.com/vllm-project/vllm/pull/52783)
to `FLASH_ATTN_MLA_SPARSE`, the SM90 sparse MLA backend.

## Requirements

- H200/H800 or another supported SM90 GPU, with a working NVIDIA driver.
- An SM90-compatible GLM-5.2 checkpoint and sufficient aggregate GPU memory.
- A compatible DSpark checkpoint **with a confidence head**.
- BF16/FP16 KV cache (`--kv-cache-dtype auto` with BF16 model computation).
  This change does not add FP8 KV support to FlashAttention sparse MLA.
- Tensor parallelism is supported; adaptive verification does not support
  pipeline parallelism or `--enforce-eager`.

FlashInfer TRT-LLM sparse MLA remains SM100-only. This branch does not relax
its architecture check. It also does not claim to fix the separate DSv4/H20
hang reported in [issue #54011](https://github.com/vllm-project/vllm/issues/54011).

## Install

With this repository at `/mnt/sfs_turbo/n30008093/vllm`:

```bash
cd /mnt/sfs_turbo/n30008093/vllm
bash tools/install_glm52_dspark_sm90.sh
```

The script creates `.venv` inside the repository and uses `uv` for installation.
Set `UV_BIN` if `uv` is not on PATH. `VLLM_VENV` may override the environment
directory. Precompiled CUDA extensions are pinned to the upstream base commit;
Python code is installed editable from this branch. If that wheel is unavailable
for the host CUDA variant, provide a matching `VLLM_PRECOMPILED_WHEEL_LOCATION`.
The script does not fall back to unrelated main-branch binaries.

## Launch

Set the actual checkpoint paths before launching; these example paths are placeholders:

```bash
export MODEL_PATH=/path/to/glm-5.2
export DRAFT_MODEL_PATH=/path/to/glm-5.2-dspark-with-confidence-head
export TP_SIZE=8
export PORT=8000
bash examples/deployment/start_glm52_dspark_sm90.sh
```

The launcher runs in the foreground and prints the effective command. It selects
the V2 model runner, SM90 FlashAttention sparse MLA, BF16 KV cache, seven draft
tokens, probabilistic draft sampling and adaptive verification. Defaults are
16K context, 128 concurrent sequences and 85% GPU memory utilization. Tune these
to the checkpoint and available memory before serving production traffic.

Set `DRY_RUN=1` to print the command without starting a server. For an A/B baseline,
set `ADAPTIVE_VERIFICATION=false`; keep other settings unchanged. Additional CLI
arguments can be appended to the script invocation. Do not override the KV dtype
to FP8 or disable CUDA graphs for adaptive verification.

For a background process, use normal shell logging:

```bash
mkdir -p /mnt/sfs_turbo/n30008093/logs
nohup bash examples/deployment/start_glm52_dspark_sm90.sh \
  > /mnt/sfs_turbo/n30008093/logs/glm52-dspark-sm90.log 2>&1 &
```

## Validate

Install test dependencies into the same environment:

```bash
INSTALL_TEST_DEPS=1 bash tools/install_glm52_dspark_sm90.sh
.venv/bin/python -m pytest tests/v1/attention/test_sparse_mla_metadata.py -q
.venv/bin/python -m pytest tests/v1/attention/test_dspark_noncausal_sparse_mla.py \
  -k adaptive_varlen -q
```

The regressions cover stale CPU boundaries, zero-length requests and padding,
and a sparse-kernel CUDA graph captured with uniform boundaries then replayed
with adaptive boundaries. The GPU output is compared with an SDPA reference
using GLM-5.2 attention dimensions. Tests for the other architecture skip.

Before considering deployment validated, run fixed/adaptive serving A/B tests,
including batch drain, and compare completion counts, acceptance length,
throughput and model output quality.

Local CPU metadata checks passed (4 cases) after reproducing 3 failures on the
unmodified base. That check used the real metadata methods with a harness that
bypassed the missing CUDA extension import guard. GPU and end-to-end serving
validation has not yet been completed; the original development host had no
working CUDA driver and lacked the full test environment.
