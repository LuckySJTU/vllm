# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Contract tests for NCP-OLMo's matching DFlash checkpoint."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.models.ncp_olmo.dflash import (
    DRAFT_ARCHITECTURE,
    is_ncp_dflash_config,
    original_draft_config,
    validate_ncp_dflash_config,
)
from vllm.v1.worker.gpu.spec_decode.ncp_dflash import (
    NCPDFlashSpeculator,
    _dflash_sdpa_mask,
    _parse_active_batch_widths,
)
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler


def make_config(**draft_overrides: object) -> SimpleNamespace:
    draft = {
        "model_type": "conceptlm_dflash",
        "proposal_method": "path_selector",
        "hlm_conditioning": "causal_residual",
        "target_layer_ids": [1, 4, 7, 10, 13],
        "block_size": 16,
        "concept_chunk_size": 4,
    }
    draft.update(draft_overrides)
    original = SimpleNamespace(**draft)
    wrapped = SimpleNamespace(model=original)
    draft_model_config = SimpleNamespace(
        architectures=[DRAFT_ARCHITECTURE],
        hf_config=wrapped,
    )
    speculative_config = SimpleNamespace(
        method="dflash",
        num_speculative_tokens=2,
        draft_model_config=draft_model_config,
    )
    return SimpleNamespace(speculative_config=speculative_config)


def test_ncp_dflash_config_is_recognized_and_unwrapped() -> None:
    config = make_config()

    assert is_ncp_dflash_config(config)
    assert original_draft_config(config).model_type == "conceptlm_dflash"
    assert validate_ncp_dflash_config(config) == (1, 4, 7, 10, 13)
    assert not NCPDFlashSpeculator.requires_aux_hidden_states


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_type", "other", "model_type"),
        ("proposal_method", "argmax", "path_selector"),
        ("hlm_conditioning", "none", "causal_residual"),
    ],
)
def test_ncp_dflash_contract_fails_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_ncp_dflash_config(make_config(**{field: value}))


def test_proposal_windows_keep_exact_modes_inside_hlm_chunk() -> None:
    speculator = object.__new__(NCPDFlashSpeculator)
    speculator.num_speculative_steps = 8
    speculator.max_model_len = 64
    speculator.chunk_size = 4

    speculator.verification_mode = "sequential_exact"
    assert [speculator.safe_proposal_count(length) for length in range(1, 6)] == [
        1,
        1,
        0,
        0,
        1,
    ]

    speculator.verification_mode = "intra_chunk_exact"
    assert [speculator.safe_proposal_count(length) for length in range(1, 6)] == [
        2,
        1,
        0,
        0,
        2,
    ]

    speculator.verification_mode = "segmented_kv_approx"
    assert speculator.safe_proposal_count(2) == 8


def test_active_batch_policy_preserves_validated_width_schedule() -> None:
    speculator = object.__new__(NCPDFlashSpeculator)
    speculator.num_speculative_steps = 8
    speculator.active_batch_widths = _parse_active_batch_widths(
        "1:8,2:8,4:4,8:2",
        maximum_width=8,
    )

    assert [speculator._active_batch_width(size) for size in range(0, 10)] == [
        0,
        8,
        8,
        4,
        4,
        2,
        2,
        2,
        2,
        2,
    ]


def test_active_batch_policy_caps_safe_proposal_window() -> None:
    speculator = object.__new__(NCPDFlashSpeculator)
    speculator.num_speculative_steps = 8
    speculator.max_model_len = 64
    speculator.chunk_size = 4
    speculator.verification_mode = "segmented_kv_approx"

    assert speculator.safe_proposal_count(8, proposal_cap=4) == 4
    assert speculator.safe_proposal_count(63, proposal_cap=4) == 1
    assert speculator.safe_proposal_count(64, proposal_cap=4) == 0


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        ("0:1", "upper bounds"),
        ("1:9", "configured speculative width"),
        ("1:8,1:4", "duplicate"),
        ("1", "upper_bound:width"),
    ],
)
def test_active_batch_policy_rejects_invalid_entries(
    policy: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _parse_active_batch_widths(policy, maximum_width=8)


def test_sdpa_mask_matches_dflash_context_and_block_visibility() -> None:
    mask = _dflash_sdpa_mask(
        torch.tensor([[2]]),
        context_length=4,
        block_size=2,
    )

    assert mask.shape == (1, 1, 2, 6)
    assert mask[0, 0].tolist() == [
        [True, True, False, False, True, True],
        [True, True, False, False, True, True],
    ]


def test_variable_draft_rows_trim_only_invalid_suffixes() -> None:
    handler = object.__new__(DraftTokensHandler)
    handler.copy_event = SimpleNamespace(synchronize=lambda: None)
    handler.draft_tokens_np = np.array(
        [[11, -1], [-1, -1], [12, 13]],
        dtype=np.int64,
    )
    handler.req_ids = ["a", "b", "c"]
    handler.num_draft_tokens = 2
    handler.trim_invalid_suffix = True

    output = handler.get_draft_tokens()

    assert output.req_ids == ["a", "b", "c"]
    assert output.draft_token_ids == [[11], [], [12, 13]]


def test_variable_draft_rows_reject_internal_holes() -> None:
    handler = object.__new__(DraftTokensHandler)
    handler.copy_event = SimpleNamespace(synchronize=lambda: None)
    handler.draft_tokens_np = np.array([[11, -1, 12]], dtype=np.int64)
    handler.req_ids = ["a"]
    handler.num_draft_tokens = 3
    handler.trim_invalid_suffix = True

    with pytest.raises(RuntimeError, match="non-suffix invalid token"):
        handler.get_draft_tokens()
