# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from argparse import Namespace

import pytest

from examples.offline_inference.ncp_olmo import configure_dflash_env


def _args(**overrides: object) -> Namespace:
    values = {
        "draft_model": "draft",
        "dflash_verification_mode": "intra_chunk_exact",
        "dflash_attention_backend": "sdpa",
        "dflash_context_kv_cache": True,
        "dflash_sparse_context_projection": True,
        "dflash_min_eligible_batch": 2,
        "dflash_min_proposal_tokens_per_row": 2,
        "dflash_min_proposal_tokens_per_batch": 8,
        "dflash_active_batch_widths": "1:2,2:2,4:1",
    }
    values.update(overrides)
    return Namespace(**values)


def test_configure_dflash_env(monkeypatch: pytest.MonkeyPatch) -> None:
    names = (
        "NCP_OLMO_DFLASH_VERIFICATION_MODE",
        "NCP_OLMO_DFLASH_ATTENTION_BACKEND",
        "NCP_OLMO_DFLASH_CONTEXT_KV_CACHE",
        "NCP_OLMO_DFLASH_SPARSE_CONTEXT_PROJECTION",
        "NCP_OLMO_DFLASH_MIN_ELIGIBLE_BATCH",
        "NCP_OLMO_DFLASH_MIN_PROPOSAL_TOKENS_PER_ROW",
        "NCP_OLMO_DFLASH_MIN_PROPOSAL_TOKENS_PER_BATCH",
        "NCP_OLMO_DFLASH_ACTIVE_BATCH_WIDTHS",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)

    configure_dflash_env(_args())

    assert [os.environ[name] for name in names] == [
        "intra_chunk_exact",
        "sdpa",
        "1",
        "1",
        "2",
        "2",
        "8",
        "1:2,2:2,4:1",
    ]


def test_flash_varlen_requires_context_kv_cache() -> None:
    with pytest.raises(ValueError, match="requires the DFlash context KV cache"):
        configure_dflash_env(
            _args(
                dflash_attention_backend="flash_varlen",
                dflash_context_kv_cache=False,
            )
        )
