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
    _dflash_context_kv,
    _dflash_sdpa_mask,
    _parse_active_batch_widths,
)
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler


class _CountingProjection(torch.nn.Linear):
    def __init__(self, hidden_size: int) -> None:
        super().__init__(hidden_size, hidden_size, bias=False)
        self.call_count = 0
        self.projected_tokens = 0
        with torch.no_grad():
            self.weight.copy_(torch.eye(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        self.projected_tokens += int(
            hidden_states.numel() // hidden_states.shape[-1]
        )
        return super().forward(hidden_states)


def _context_kv_test_layer() -> SimpleNamespace:
    hidden_size = 4
    layer = SimpleNamespace(
        num_attention_heads=2,
        head_size=2,
        k_proj=_CountingProjection(hidden_size),
        v_proj=_CountingProjection(hidden_size),
        k_norm=torch.nn.Identity(),
    )
    layer._split_heads = lambda hidden: hidden.unflatten(-1, (2, 2))
    layer.rotary = lambda hidden, positions: hidden
    return layer


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


def test_context_kv_cache_survives_continuous_batch_row_reorder() -> None:
    layer = _context_kv_test_layer()
    layer._ncp_dflash_request_ids = ["request-a", "request-b"]
    layer._ncp_dflash_causal_lengths = [3, 2]
    first = torch.zeros(2, 4, 4)
    first[0, :3] = torch.tensor([[1.0] * 4, [2.0] * 4, [3.0] * 4])
    first[1, :2] = torch.tensor([[10.0] * 4, [11.0] * 4])

    _dflash_context_kv(layer, first, torch.tensor([[3], [2]]))

    assert layer.k_proj.projected_tokens == 5
    assert layer.k_proj.call_count == 1

    second = torch.zeros(2, 5, 4)
    second[0, :3] = torch.tensor([[10.0] * 4, [11.0] * 4, [12.0] * 4])
    second[1, :4] = torch.tensor(
        [[1.0] * 4, [2.0] * 4, [3.0] * 4, [4.0] * 4]
    )
    layer._ncp_dflash_request_ids = ["request-b", "request-a"]
    layer._ncp_dflash_causal_lengths = [3, 4]
    cached_key, cached_value = _dflash_context_kv(
        layer,
        second,
        torch.tensor([[3], [4]]),
    )

    assert layer.k_proj.projected_tokens == 7
    assert layer.v_proj.projected_tokens == 7
    assert layer.k_proj.call_count == 2
    reference = _context_kv_test_layer()
    expected_key, expected_value = _dflash_context_kv(
        reference,
        second[:, :4],
        torch.tensor([[3], [4]]),
    )
    torch.testing.assert_close(cached_key, expected_key)
    torch.testing.assert_close(cached_value, expected_value)
    assert set(layer._ncp_dflash_context_kv_cache) == {
        "request-a",
        "request-b",
    }


def test_context_kv_cache_prunes_released_requests() -> None:
    speculator = object.__new__(NCPDFlashSpeculator)
    speculator.context_kv_cache = True
    layer = SimpleNamespace(
        _ncp_dflash_context_kv_cache={
            "request-a": object(),
            "request-b": object(),
        }
    )
    speculator._draft_model = SimpleNamespace(layers=[layer])

    speculator._prune_context_kv_cache(["request-b"])

    assert set(layer._ncp_dflash_context_kv_cache) == {"request-b"}


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
