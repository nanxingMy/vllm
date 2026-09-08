# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA request mappings when adaptive budgets change device boundaries."""

import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata

pytest.importorskip("vllm.vllm_flash_attn", exc_type=ImportError)


@pytest.mark.parametrize(
    "device_starts,cpu_starts,num_tokens,expected",
    [
        (
            [0, 1, 8, 11, 16],
            [0, 4, 8, 12, 16],
            16,
            [0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
        ),
        # A drained request contributes no tokens; graph padding is zeroed.
        ([0, 1, 1, 4, 4], [0, 1, 2, 3, 4], 8, [0, 2, 2, 2, 0, 0, 0, 0]),
        # Adaptive decode keeps the prefill split point and lengths intact.
        ([0, 1, 4, 10], [0, 2, 4, 10], 10, [0, 1, 1, 1, 2, 2, 2, 2, 2, 2]),
        ([0, 0, 0], [0, 0, 0], 4, [0, 0, 0, 0]),
    ],
)
def test_flashattn_sparse_mapping_uses_device_boundaries(
    device_starts, cpu_starts, num_tokens, expected
):
    from vllm.v1.attention.backends.mla.flashattn_mla_sparse import (
        FlashAttnMLASparseMetadataBuilder,
    )

    num_reqs = len(device_starts) - 1
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor(device_starts, dtype=torch.int32),
        query_start_loc_cpu=torch.tensor(cpu_starts, dtype=torch.int32),
        seq_lens=torch.full((num_reqs,), 32, dtype=torch.int32),
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=8,
        max_seq_len=32,
        block_table_tensor=torch.arange(num_reqs, dtype=torch.int32).view(-1, 1),
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64),
    )
    # Only request mapping is under test; no model weights or KV workspace needed.
    builder = object.__new__(FlashAttnMLASparseMetadataBuilder)
    builder.req_id_per_token_buffer = torch.full((32,), -1, dtype=torch.int32)
    mapping = builder._build_req_id_per_token(common)
    torch.testing.assert_close(mapping, torch.tensor(expected, dtype=torch.int32))
