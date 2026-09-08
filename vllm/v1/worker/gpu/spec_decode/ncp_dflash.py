# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NCP-OLMo parallel DFlash proposer for the V2 GPU model runner."""

from __future__ import annotations

import math
from contextlib import nullcontext
from types import FunctionType, MethodType, SimpleNamespace
from typing import Any

import torch
from torch.nn import functional as F

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.ncp_olmo.dflash import (
    VERIFICATION_MODES,
    DFlashConceptLMDFlashModel,
    original_draft_config,
    validate_ncp_dflash_config,
)
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

logger = init_logger(__name__)


def _dflash_sdpa_mask(
    anchor_positions: torch.Tensor,
    *,
    context_length: int,
    block_size: int,
) -> torch.Tensor:
    """Materialize the checkpoint's small-query DFlash visibility rule."""

    batch_size, anchor_count = anchor_positions.shape
    query_length = anchor_count * block_size
    query_blocks = torch.arange(
        anchor_count,
        device=anchor_positions.device,
    ).repeat_interleave(block_size)
    valid_queries = (anchor_positions >= 0).repeat_interleave(block_size, dim=1)
    anchor_by_query = anchor_positions.repeat_interleave(block_size, dim=1)
    context_indices = torch.arange(context_length, device=anchor_positions.device)
    context_allowed = (
        context_indices.view(1, 1, context_length) < anchor_by_query.unsqueeze(-1)
    ) & valid_queries.unsqueeze(-1)
    draft_blocks = (
        torch.arange(query_length, device=anchor_positions.device) // block_size
    )
    draft_allowed = (
        query_blocks.view(1, query_length, 1) == draft_blocks.view(1, 1, query_length)
    ) & valid_queries.unsqueeze(-1)
    return torch.cat((context_allowed, draft_allowed), dim=-1).view(
        batch_size,
        1,
        query_length,
        context_length + query_length,
    )


def _dflash_cached_context_kv_rows(
    layer: torch.nn.Module,
    context: torch.Tensor,
    anchor_positions: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return request-local context KV rows, projecting only new suffixes.

    The final target context row is the unprocessed anchor placeholder. Rows
    strictly before it are immutable, so their projected K/V may be retained
    across decode steps. New suffixes from all active requests are packed into
    one projection to preserve batching after request removal or row reorder.
    """

    batch_size, supplied_length, _ = context.shape
    request_ids = getattr(layer, "_ncp_dflash_request_ids", None)
    causal_lengths = getattr(layer, "_ncp_dflash_causal_lengths", None)
    if request_ids is None or causal_lengths is None:
        raise ValueError("NCP DFlash context KV cache metadata is missing")
    if len(request_ids) != batch_size or len(causal_lengths) != batch_size:
        raise ValueError("NCP DFlash context KV cache metadata is misaligned")
    context_offsets = getattr(layer, "_ncp_dflash_context_offsets", None)
    if context_offsets is not None and len(context_offsets) != batch_size:
        raise ValueError("NCP DFlash context offsets are misaligned")
    if int(anchor_positions.shape[0]) != batch_size:
        raise ValueError("NCP DFlash anchor positions are misaligned")

    cache = getattr(layer, "_ncp_dflash_context_kv_cache", None)
    if cache is None:
        cache = {}
        layer._ncp_dflash_context_kv_cache = cache
    projection_dtype = getattr(layer.k_proj.weight, "dtype", context.dtype)
    empty_shape = (1, layer.num_attention_heads, 0, layer.head_size)
    records: list[tuple[str, int, int, torch.Tensor, torch.Tensor]] = []
    suffixes: list[torch.Tensor] = []
    suffix_positions: list[torch.Tensor] = []
    suffix_lengths: list[int] = []
    projected_tokens = 0
    reused_tokens = 0

    for row_index, raw_request_id in enumerate(request_ids):
        request_id = str(raw_request_id)
        causal_length = int(causal_lengths[row_index])
        context_offset = (
            0 if context_offsets is None else int(context_offsets[row_index])
        )
        if not (
            0 <= context_offset <= causal_length
            and causal_length - context_offset <= supplied_length
        ):
            raise ValueError(
                "NCP DFlash causal context is outside the supplied slice: "
                f"request={request_id!r} offset={context_offset} "
                f"causal={causal_length} supplied={supplied_length}"
            )

        cached = cache.get(request_id)
        if cached is None:
            cached_length = 0
            cached_key = torch.empty(
                empty_shape,
                device=context.device,
                dtype=projection_dtype,
            )
            cached_value = cached_key.clone()
        else:
            cached_length, cached_key, cached_value = cached
            if (
                int(cached_length) > causal_length
                or cached_key.device != context.device
                or cached_key.dtype != projection_dtype
            ):
                cached_length = 0
                cached_key = torch.empty(
                    empty_shape,
                    device=context.device,
                    dtype=projection_dtype,
                )
                cached_value = cached_key.clone()

        suffix_length = causal_length - int(cached_length)
        if suffix_length:
            if int(cached_length) < context_offset:
                raise RuntimeError(
                    "NCP DFlash compact context starts after its valid cache: "
                    f"request={request_id!r} cached={cached_length} "
                    f"offset={context_offset}"
                )
            relative_start = int(cached_length) - context_offset
            relative_end = causal_length - context_offset
            suffixes.append(context[row_index, relative_start:relative_end])
            suffix_positions.append(
                torch.arange(
                    int(cached_length),
                    causal_length,
                    device=context.device,
                )
            )
            suffix_lengths.append(suffix_length)
            projected_tokens += suffix_length
        reused_tokens += min(int(cached_length), causal_length)
        records.append(
            (
                request_id,
                causal_length,
                suffix_length,
                cached_key,
                cached_value,
            )
        )

    projected_keys: list[torch.Tensor] = []
    projected_values: list[torch.Tensor] = []
    if suffixes:
        packed_suffix = torch.cat(suffixes, dim=0).unsqueeze(0)
        packed_positions = torch.cat(suffix_positions, dim=0).unsqueeze(0)
        packed_key = layer._split_heads(layer.k_norm(layer.k_proj(packed_suffix)))
        packed_value = layer._split_heads(layer.v_proj(packed_suffix))
        packed_key = layer.rotary(packed_key, packed_positions).transpose(1, 2)
        packed_value = packed_value.transpose(1, 2)
        projected_keys = list(torch.split(packed_key, suffix_lengths, dim=2))
        projected_values = list(torch.split(packed_value, suffix_lengths, dim=2))

    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    projected_index = 0
    for request_id, causal_length, suffix_length, cached_key, cached_value in records:
        if suffix_length:
            required_length = causal_length
            capacity = int(cached_key.shape[2])
            if required_length > capacity:
                target_capacity = max(16, required_length, capacity * 2)
                new_capacity = 1 << (target_capacity - 1).bit_length()
                key_storage = cached_key.new_empty(
                    (
                        1,
                        layer.num_attention_heads,
                        new_capacity,
                        layer.head_size,
                    )
                )
                value_storage = cached_value.new_empty(key_storage.shape)
                cached_prefix = required_length - suffix_length
                if cached_prefix:
                    key_storage[:, :, :cached_prefix].copy_(
                        cached_key[:, :, :cached_prefix]
                    )
                    value_storage[:, :, :cached_prefix].copy_(
                        cached_value[:, :, :cached_prefix]
                    )
                cached_key = key_storage
                cached_value = value_storage
            suffix_start = required_length - suffix_length
            cached_key[:, :, suffix_start:required_length].copy_(
                projected_keys[projected_index].to(dtype=cached_key.dtype)
            )
            cached_value[:, :, suffix_start:required_length].copy_(
                projected_values[projected_index].to(dtype=cached_value.dtype)
            )
            projected_index += 1
        cache[request_id] = (
            causal_length,
            cached_key.detach(),
            cached_value.detach(),
        )
        keys.append(cached_key[:, :, :causal_length])
        values.append(cached_value[:, :, :causal_length])
    if projected_index != len(projected_keys):
        raise RuntimeError("NCP DFlash context KV suffix accounting drifted")
    layer._ncp_dflash_context_kv_projected_tokens = (
        int(getattr(layer, "_ncp_dflash_context_kv_projected_tokens", 0))
        + projected_tokens
    )
    layer._ncp_dflash_context_kv_reused_tokens = (
        int(getattr(layer, "_ncp_dflash_context_kv_reused_tokens", 0)) + reused_tokens
    )
    return keys, values


def _dflash_context_kv(
    layer: torch.nn.Module,
    context: torch.Tensor,
    anchor_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded context K/V, reusing request-local rows when enabled."""

    batch_size, supplied_length, _ = context.shape
    causal_lengths = getattr(layer, "_ncp_dflash_causal_lengths", None)
    sequence_length = (
        max((int(length) for length in causal_lengths), default=0)
        if causal_lengths is not None
        else supplied_length
    )
    request_ids = getattr(layer, "_ncp_dflash_request_ids", None)
    if request_ids is None:
        positions = torch.arange(sequence_length, device=context.device)
        positions = positions.view(1, sequence_length).expand(batch_size, -1)
        context_key = layer._split_heads(layer.k_norm(layer.k_proj(context)))
        context_value = layer._split_heads(layer.v_proj(context))
        context_key = layer.rotary(context_key, positions).transpose(1, 2)
        return context_key, context_value.transpose(1, 2)

    key_rows, value_rows = _dflash_cached_context_kv_rows(
        layer,
        context,
        anchor_positions,
    )
    keys = []
    values = []
    for key, value in zip(key_rows, value_rows, strict=True):
        padding = sequence_length - int(key.shape[2])
        keys.append(key if padding == 0 else F.pad(key, (0, 0, 0, padding)))
        values.append(value if padding == 0 else F.pad(value, (0, 0, 0, padding)))
    return torch.cat(keys, dim=0), torch.cat(values, dim=0)


def _sdpa_dflash_attention(
    layer: torch.nn.Module,
    slots: torch.Tensor,
    context: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_mask: Any,
) -> torch.Tensor:
    """Inference-only SDPA equivalent of the checkpoint's FlexAttention."""

    del block_mask
    batch_size, anchor_count, block_size, hidden_size = slots.shape
    causal_lengths = getattr(layer, "_ncp_dflash_causal_lengths", None)
    context_length = (
        max((int(length) for length in causal_lengths), default=0)
        if causal_lengths is not None
        else int(context.shape[1])
    )
    slot_positions = anchor_positions.clamp(min=0).unsqueeze(-1) + torch.arange(
        block_size,
        device=slots.device,
    )

    query = layer._split_heads(layer.q_norm(layer.q_proj(slots)))
    slot_key = layer._split_heads(layer.k_norm(layer.k_proj(slots)))
    slot_value = layer._split_heads(layer.v_proj(slots))
    query = layer.rotary(query, slot_positions)
    slot_key = layer.rotary(slot_key, slot_positions)

    context_key, context_value = _dflash_context_kv(
        layer,
        context,
        anchor_positions,
    )

    query_length = anchor_count * block_size
    query = (
        query.reshape(
            batch_size,
            query_length,
            layer.num_attention_heads,
            layer.head_size,
        )
        .transpose(1, 2)
        .contiguous()
    )
    slot_key = slot_key.reshape(
        batch_size,
        query_length,
        layer.num_attention_heads,
        layer.head_size,
    ).transpose(1, 2)
    slot_value = slot_value.reshape(
        batch_size,
        query_length,
        layer.num_attention_heads,
        layer.head_size,
    ).transpose(1, 2)
    key = torch.cat((context_key, slot_key), dim=2)
    value = torch.cat((context_value, slot_value), dim=2)
    attention_mask = _dflash_sdpa_mask(
        anchor_positions,
        context_length=context_length,
        block_size=block_size,
    )
    attended = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        scale=layer.head_size**-0.5,
    )
    return attended.transpose(1, 2).reshape(
        batch_size,
        anchor_count,
        block_size,
        hidden_size,
    )


def _flash_varlen_dflash_attention(
    layer: torch.nn.Module,
    slots: torch.Tensor,
    context: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_mask: Any,
) -> torch.Tensor:
    """Run packed FlashAttention for one DFlash anchor per request."""

    del block_mask
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    batch_size, anchor_count, block_size, hidden_size = slots.shape
    if anchor_count != 1:
        raise ValueError("packed NCP DFlash attention requires one anchor per request")
    slot_positions = anchor_positions.clamp(min=0).unsqueeze(-1) + torch.arange(
        block_size,
        device=slots.device,
    )
    query = layer._split_heads(layer.q_norm(layer.q_proj(slots)))
    slot_key = layer._split_heads(layer.k_norm(layer.k_proj(slots)))
    slot_value = layer._split_heads(layer.v_proj(slots))
    query = layer.rotary(query, slot_positions)
    slot_key = layer.rotary(slot_key, slot_positions)

    context_keys, context_values = _dflash_cached_context_kv_rows(
        layer,
        context,
        anchor_positions,
    )
    query_length = anchor_count * block_size
    query = query.reshape(
        batch_size * query_length,
        layer.num_attention_heads,
        layer.head_size,
    ).contiguous()
    slot_key = slot_key.reshape(
        batch_size,
        query_length,
        layer.num_attention_heads,
        layer.head_size,
    )
    slot_value = slot_value.reshape(
        batch_size,
        query_length,
        layer.num_attention_heads,
        layer.head_size,
    )
    key_rows = [
        torch.cat(
            (
                context_key.squeeze(0).transpose(0, 1).to(query.dtype),
                slot_key[row_index].to(query.dtype),
            ),
            dim=0,
        )
        for row_index, context_key in enumerate(context_keys)
    ]
    value_rows = [
        torch.cat(
            (
                context_value.squeeze(0).transpose(0, 1).to(query.dtype),
                slot_value[row_index].to(query.dtype),
            ),
            dim=0,
        )
        for row_index, context_value in enumerate(context_values)
    ]
    key_lengths = [int(key.shape[0]) for key in key_rows]
    cu_seqlens_q = torch.arange(
        0,
        (batch_size + 1) * query_length,
        query_length,
        device=slots.device,
        dtype=torch.int32,
    )
    cu_seqlens_k = torch.tensor(
        [0, *key_lengths],
        device=slots.device,
        dtype=torch.int32,
    ).cumsum(dim=0, dtype=torch.int32)
    attended = flash_attn_varlen_func(
        q=query,
        k=torch.cat(key_rows, dim=0),
        v=torch.cat(value_rows, dim=0),
        max_seqlen_q=query_length,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max(key_lengths),
        cu_seqlens_k=cu_seqlens_k,
        dropout_p=0.0,
        softmax_scale=layer.head_size**-0.5,
        causal=False,
        fa_version=3,
    )
    return attended.reshape(
        batch_size,
        anchor_count,
        block_size,
        hidden_size,
    )


def _install_draft_attention_backend(
    model: torch.nn.Module,
    backend: str,
) -> str:
    """Install an attention implementation only on this draft instance."""

    if backend not in {"flash_varlen", "sdpa", "flex_attention"}:
        raise ValueError(
            "NCP DFlash attention backend must be flash_varlen, sdpa, or flex_attention"
        )
    model.config.flex_attention_compile = False
    if backend == "flex_attention":
        return backend

    forward = getattr(model.forward, "__func__", model.forward)
    forward_globals = getattr(forward, "__globals__", None)
    if not isinstance(forward_globals, dict) or (
        "_create_dflash_block_mask" not in forward_globals
    ):
        raise RuntimeError(
            "DFlash remote model no longer exposes its block-mask helper"
        )
    local_globals = dict(forward_globals)
    local_globals["_create_dflash_block_mask"] = lambda *_args, **_kwargs: None
    local_forward = FunctionType(
        forward.__code__,
        local_globals,
        forward.__name__,
        forward.__defaults__,
        forward.__closure__,
    )
    local_forward.__kwdefaults__ = forward.__kwdefaults__
    local_forward.__annotations__ = forward.__annotations__
    local_forward.__dict__.update(forward.__dict__)
    local_forward.__module__ = forward.__module__
    local_forward.__qualname__ = forward.__qualname__
    model.forward = MethodType(local_forward, model)

    for layer in model.layers:
        layer.flex_attention_compile = False
        attention = (
            _flash_varlen_dflash_attention
            if backend == "flash_varlen"
            else _sdpa_dflash_attention
        )
        layer._attention = MethodType(attention, layer)
    return backend


class NCPDFlashSpeculator(DraftModelSpeculator):
    """Draft from NCP target features while target verification stays authoritative."""

    supports_mm_inputs = False
    requires_aux_hidden_states = False
    variable_draft_lengths = True

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        self.vllm_config = vllm_config
        self.device = device
        speculative_config = vllm_config.speculative_config
        if speculative_config is None:
            raise ValueError("NCP DFlash requires speculative_config")
        self.speculative_config = speculative_config
        self.draft_config = original_draft_config(vllm_config)
        self.target_layer_ids = validate_ncp_dflash_config(vllm_config)
        self.num_speculative_steps = int(speculative_config.num_speculative_tokens)
        self.max_model_len = int(vllm_config.model_config.max_model_len)
        self.max_num_reqs = int(vllm_config.scheduler_config.max_num_seqs)
        self.dtype = vllm_config.model_config.dtype
        if not speculative_config.draft_model_config.trust_remote_code:
            raise ValueError(
                "NCP DFlash checkpoints contain their Hugging Face draft model; "
                "pass --trust-remote-code to opt in to loading it"
            )

        self.verification_mode = speculative_config.ncp_dflash_verification_mode
        if self.verification_mode not in VERIFICATION_MODES:
            raise ValueError(
                "NCP DFlash verification mode must be one of "
                f"{sorted(VERIFICATION_MODES)!r}"
            )
        schedule = speculative_config.num_speculative_tokens_per_batch_size
        self.active_batch_width_lookup = (
            None
            if schedule is None
            else build_dynamic_sd_schedule_lookup(
                schedule,
                vllm_max_batch_size=self.max_num_reqs,
                vllm_num_speculative_tokens=self.num_speculative_steps,
            )
        )
        self.context_kv_cache = speculative_config.ncp_dflash_context_kv_cache
        self.sparse_context_projection = (
            speculative_config.ncp_dflash_sparse_context_projection
        )
        if self.sparse_context_projection and not self.context_kv_cache:
            raise ValueError(
                "NCP DFlash sparse context projection requires context KV cache"
            )
        self.min_eligible_batch = int(speculative_config.ncp_dflash_min_eligible_batch)
        self.min_proposal_tokens_per_row = int(
            speculative_config.ncp_dflash_min_proposal_tokens_per_row
        )
        self.min_proposal_tokens_per_batch = int(
            speculative_config.ncp_dflash_min_proposal_tokens_per_batch
        )
        self.chunk_size = int(self.draft_config.concept_chunk_size)
        self.draft_tokens = torch.full(
            (self.max_num_reqs, self.num_speculative_steps),
            -1,
            dtype=torch.int64,
            device=device,
        )
        # Path-selector proposals are deterministic token IDs, so there are no
        # draft logits to pass to the rejection sampler.  ModelRunner consumes
        # this attribute for every speculator implementation.
        self.draft_logits: torch.Tensor | None = None
        self._draft_model: torch.nn.Module | None = None
        self.target_model: torch.nn.Module | None = None
        # The remote NCP drafter owns request-local attention state instead of
        # registering vLLM AttentionLayerBase instances in the target config.
        # ModelRunner still reads this attribute for every DraftModelSpeculator.
        self.draft_attn_layer_names: set[str] = set()
        self._logged_first_proposal = False
        self._last_context_kv_cache_stats: dict[str, Any] = {
            "enabled": self.context_kv_cache,
            "projected_tokens": 0,
            "reused_tokens": 0,
        }
        self._last_proposal_stats: dict[str, Any] = {}
        logger.info(
            "NCP DFlash enabled with verification_mode=%s, proposal_width=%d, "
            "dynamic_width_schedule=%s, context_kv_cache=%s, and "
            "sparse_context_projection=%s, min_eligible_batch=%d, "
            "min_tokens_per_row=%d, and min_tokens_per_batch=%d",
            self.verification_mode,
            self.num_speculative_steps,
            schedule or "static",
            self.context_kv_cache,
            self.sparse_context_projection,
            self.min_eligible_batch,
            self.min_proposal_tokens_per_row,
            self.min_proposal_tokens_per_batch,
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        del cudagraph_mode

    def capture(self) -> None:
        return None

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        """Load the draft through vLLM and bind its explicit target instance."""

        del target_attn_layer_names
        self.target_model = target_model
        wrapper = get_model(
            vllm_config=self.vllm_config,
            model_config=self.speculative_config.draft_model_config,
            load_config=self.speculative_config.draft_load_config,
        )
        if not isinstance(wrapper, DFlashConceptLMDFlashModel):
            raise TypeError(
                "NCP DFlash loader returned an unexpected model type: "
                f"{type(wrapper).__name__}"
            )
        model = wrapper.model
        if hasattr(model, "gradient_checkpointing"):
            model.gradient_checkpointing = False
        model.config.gradient_checkpointing = False
        if hasattr(model.config, "flex_attention_compile"):
            model.config.flex_attention_compile = False
        if getattr(model.config, "model_type", None) != "conceptlm_dflash":
            raise ValueError("loaded draft is not a conceptlm_dflash checkpoint")
        if tuple(int(value) for value in model.config.target_layer_ids) != (
            self.target_layer_ids
        ):
            raise ValueError(
                "loaded draft target_layer_ids changed after config loading"
            )
        if int(model.config.block_size) < self.num_speculative_steps:
            raise ValueError(
                "loaded draft block_size is smaller than the proposal width"
            )
        self._draft_runtime_max_block_size = int(model.config.block_size)
        if str(model.config.proposal_method) != "path_selector":
            raise ValueError("loaded NCP drafter must use path_selector")
        if str(model.config.hlm_conditioning) != "causal_residual":
            raise ValueError("loaded NCP drafter must use causal_residual HLM state")
        attention_backend = _install_draft_attention_backend(
            model,
            self.speculative_config.ncp_dflash_attention_backend,
        )
        if attention_backend == "flash_varlen" and not self.context_kv_cache:
            raise ValueError(
                "flash_varlen NCP DFlash attention requires request-local "
                "context KV cache"
            )
        logger.info("NCP DFlash draft attention backend: %s", attention_backend)
        self._draft_model = model
        return wrapper

    def load_model(self, target_model: torch.nn.Module) -> None:
        """Load the draft once during the runner's accounted loading phase."""

        self.draft_attn_layer_names = set()
        self.model = self.load_draft_model(target_model, set())

    def set_attn(self, *args: Any, **kwargs: Any) -> None:
        """Keep the remote drafter outside vLLM's target KV-cache plumbing."""

        del args, kwargs

    def _require_draft_model(self) -> torch.nn.Module:
        if self._draft_model is None:
            raise RuntimeError("NCP DFlash draft model has not been loaded")
        return self._draft_model

    def _require_target_model(self) -> torch.nn.Module:
        if self.target_model is None:
            raise RuntimeError("NCP DFlash target model has not been bound")
        return self.target_model

    def _active_batch_width(self, active_batch_size: int) -> int:
        """Return the proposal cap for the current continuous-batch step."""

        active_batch_size = int(active_batch_size)
        if active_batch_size <= 0:
            return 0
        if self.active_batch_width_lookup is None:
            return self.num_speculative_steps
        lookup_index = min(active_batch_size, len(self.active_batch_width_lookup) - 1)
        return self.active_batch_width_lookup[lookup_index]

    def safe_proposal_count(
        self,
        prefix_length: int,
        proposal_cap: int | None = None,
    ) -> int:
        """Return the proposal width allowed by the selected state contract."""

        configured_maximum = self.num_speculative_steps
        if proposal_cap is not None:
            configured_maximum = min(configured_maximum, max(0, int(proposal_cap)))
        maximum = max(
            0,
            min(
                configured_maximum,
                self.max_model_len - int(prefix_length),
            ),
        )
        if self.verification_mode != "segmented_kv_approx":
            anchor_position = int(prefix_length) - 1
            before_chunk = self.chunk_size - 2 - (anchor_position % self.chunk_size)
            maximum = min(maximum, max(0, before_chunk))
            if self.verification_mode == "sequential_exact":
                maximum = min(maximum, 1)
        if maximum < getattr(self, "min_proposal_tokens_per_row", 1):
            return 0
        return maximum

    def _proposal_batch_skip_reason(
        self,
        proposal_counts: list[int],
    ) -> str | None:
        """Return why this draft batch is too small to amortize safely."""

        if not proposal_counts:
            return "no_eligible_rows"
        if self.max_num_reqs > 1 and len(proposal_counts) < getattr(
            self, "min_eligible_batch", 1
        ):
            return "eligible_batch_too_small"
        if sum(proposal_counts) < getattr(
            self,
            "min_proposal_tokens_per_batch",
            1,
        ):
            return "proposal_budget_too_small"
        return None

    @staticmethod
    def _pad_context_batch(contexts: list[torch.Tensor]) -> torch.Tensor:
        if not contexts:
            raise ValueError("cannot pad an empty NCP DFlash context batch")
        max_length = max(int(context.shape[1]) for context in contexts)
        padded = []
        for context in contexts:
            if context.ndim != 4 or int(context.shape[0]) != 1:
                raise ValueError(
                    "NCP DFlash target context must be [1, sequence, layers, hidden]"
                )
            padding = max_length - int(context.shape[1])
            padded.append(torch.nn.functional.pad(context, (0, 0, 0, 0, 0, padding)))
        return torch.cat(padded, dim=0)

    @staticmethod
    def _valid_cached_context_length(
        layer: torch.nn.Module,
        request_id: str,
        *,
        causal_length: int,
        device: torch.device,
    ) -> int:
        cache = getattr(layer, "_ncp_dflash_context_kv_cache", None)
        cached = None if cache is None else cache.get(str(request_id))
        if cached is None:
            return 0
        cached_length, cached_key, _ = cached
        projection_dtype = getattr(layer.k_proj.weight, "dtype", cached_key.dtype)
        if (
            int(cached_length) > int(causal_length)
            or cached_key.device != device
            or cached_key.dtype != projection_dtype
        ):
            return 0
        return int(cached_length)

    def _forward_sparse_context(
        self,
        draft: torch.nn.Module,
        contexts: list[torch.Tensor],
        *,
        request_ids: list[str],
        prefix_lengths: list[int],
        anchor_embeddings: torch.Tensor,
        mask_embedding: torch.Tensor,
        anchor_positions: torch.Tensor,
        hlm_hidden_states: torch.Tensor,
    ) -> Any:
        """Run the remote drafter using only uncached context suffixes."""

        batch_size = len(contexts)
        if not (len(request_ids) == batch_size == len(prefix_lengths)):
            raise ValueError("NCP DFlash sparse context metadata is misaligned")
        suffix_starts: list[int] = []
        suffixes: list[torch.Tensor] = []
        suffix_lengths: list[int] = []
        for row_index, (request_id, prefix_length) in enumerate(
            zip(request_ids, prefix_lengths, strict=True)
        ):
            causal_length = int(prefix_length) - 1
            cached_lengths = [
                self._valid_cached_context_length(
                    layer,
                    request_id,
                    causal_length=causal_length,
                    device=contexts[row_index].device,
                )
                for layer in draft.layers
            ]
            suffix_start = min(cached_lengths, default=0)
            suffix_length = causal_length - suffix_start
            suffix_starts.append(suffix_start)
            suffix_lengths.append(suffix_length)
            if suffix_length:
                suffixes.append(contexts[row_index][0, suffix_start:causal_length])

        hidden_size = int(draft.config.draft_hidden_size)
        if suffixes:
            packed_features = torch.cat(suffixes, dim=0)
            shared_suffix = draft.feature_norm(
                draft.feature_projection(packed_features.flatten(start_dim=1))
            )
            layer_suffixes = [
                draft._context_for_layer(
                    packed_features.unsqueeze(0),
                    shared_suffix.unsqueeze(0),
                    layer_index,
                ).squeeze(0)
                for layer_index in range(len(draft.layers))
            ]
        else:
            layer_suffixes = [
                anchor_embeddings.new_empty((0, hidden_size)) for _ in draft.layers
            ]

        compact_length = max(suffix_lengths, default=0)
        block_size = int(draft.config.block_size)
        mask_slots = mask_embedding.view(1, 1, 1, -1).expand(
            batch_size,
            int(anchor_embeddings.shape[1]),
            block_size - 1,
            -1,
        )
        target_slots = torch.cat((anchor_embeddings.unsqueeze(2), mask_slots), dim=2)
        slots = draft.input_projection(target_slots)
        uniform_suffix_length = (
            suffix_lengths[0]
            if suffix_lengths
            and all(length == suffix_lengths[0] for length in suffix_lengths)
            else None
        )
        for layer_index, layer in enumerate(draft.layers):
            layer._ncp_dflash_context_offsets = suffix_starts
            if uniform_suffix_length is not None:
                layer_context = layer_suffixes[layer_index].reshape(
                    batch_size,
                    uniform_suffix_length,
                    hidden_size,
                )
            else:
                layer_context = slots.new_zeros(
                    (batch_size, compact_length, hidden_size)
                )
                packed_offset = 0
                for row_index, suffix_length in enumerate(suffix_lengths):
                    if suffix_length:
                        layer_context[row_index, :suffix_length] = layer_suffixes[
                            layer_index
                        ][packed_offset : packed_offset + suffix_length]
                    packed_offset += suffix_length
            layer_hlm = draft._hlm_for_layer(
                hlm_hidden_states,
                slots,
                layer_index,
            )
            slots = layer(
                slots,
                layer_context,
                layer_hlm,
                anchor_positions,
                None,
            )
        hidden = draft.output_projection(draft.final_norm(slots))
        self._last_sparse_context_projection = {
            "enabled": True,
            "projected_target_tokens": sum(suffix_lengths),
            "total_causal_target_tokens": sum(
                prefix_length - 1 for prefix_length in prefix_lengths
            ),
        }
        return SimpleNamespace(last_hidden_state=hidden)

    def _prune_context_kv_cache(self, request_ids: list[str]) -> None:
        if not self.context_kv_cache or self._draft_model is None:
            return
        active_request_ids = {str(request_id) for request_id in request_ids}
        for layer in self._draft_model.layers:
            cache = getattr(layer, "_ncp_dflash_context_kv_cache", None)
            if cache is None:
                continue
            for request_id in tuple(cache):
                if request_id not in active_request_ids:
                    del cache[request_id]

    @staticmethod
    def _path_selector_batch(
        model: torch.nn.Module,
        hidden: torch.Tensor,
        output_weight: torch.Tensor,
        embedding_weight: torch.Tensor,
        previous: torch.Tensor,
        proposal_counts: list[int],
    ) -> torch.Tensor:
        batch_size = int(hidden.shape[0])
        max_count = max(proposal_counts, default=0)
        result = torch.full(
            (batch_size, max_count),
            -1,
            dtype=torch.long,
            device=hidden.device,
        )
        if max_count == 0:
            return result

        base_logits = torch.matmul(
            hidden[:, :max_count],
            output_weight.transpose(0, 1),
        )
        selector_top_k = int(model.config.selector_top_k)
        if not 0 < selector_top_k <= int(base_logits.shape[-1]):
            raise ValueError("selector_top_k is outside the target vocabulary")
        topk_values, topk_ids = torch.topk(
            base_logits.float(),
            k=selector_top_k,
            dim=-1,
        )
        del base_logits
        previous = previous.to(device=hidden.device, dtype=torch.long).clone()
        count_tensor = torch.tensor(
            proposal_counts,
            dtype=torch.long,
            device=hidden.device,
        )
        selector_scale = math.sqrt(
            float(model.selector_previous_projection.out_features)
        )
        for position in range(max_count):
            active = count_tensor > position
            position_topk_values = topk_values[active, position]
            position_topk_ids = topk_ids[active, position]
            previous_embeddings = embedding_weight[previous[active]]
            candidate_embeddings = embedding_weight[position_topk_ids]
            previous_features = model.selector_previous_projection(
                previous_embeddings
            ).float()
            candidate_features = model.selector_candidate_projection(
                candidate_embeddings
            ).float()
            context_gates = torch.sigmoid(
                model.selector_context_projection(hidden[active, position]).float()
            )
            compatibility = (
                torch.sum(
                    candidate_features
                    * (previous_features * context_gates).unsqueeze(1),
                    dim=-1,
                )
                / selector_scale
            )
            selected_indices = (position_topk_values + compatibility).argmax(dim=-1)
            selected = position_topk_ids.gather(
                1, selected_indices.unsqueeze(1)
            ).squeeze(1)
            previous[active] = selected
            result[active, position] = selected
        return result

    def _propose_rows(
        self,
        request_ids: list[str],
        anchor_ids: torch.Tensor,
        proposal_counts: list[int],
    ) -> torch.Tensor:
        target = self._require_target_model()
        contexts = []
        hlm_states = []
        embedding_weight: torch.Tensor | None = None
        output_weight: torch.Tensor | None = None
        prefix_lengths = []
        for request_id in request_ids:
            context, hlm_state, row_embedding, row_output = (
                target.ncp_dflash_proposal_context(request_id)
            )
            state = target.request_states.states[request_id]
            prefix_lengths.append(int(state.next_token_position) + 1)
            if embedding_weight is None:
                embedding_weight = row_embedding
                output_weight = row_output
            elif (
                embedding_weight.data_ptr() != row_embedding.data_ptr()
                or output_weight is None
                or output_weight.data_ptr() != row_output.data_ptr()
            ):
                raise RuntimeError(
                    "NCP DFlash request rows do not share target weights"
                )
            contexts.append(context)
            hlm_states.append(hlm_state)
        assert embedding_weight is not None and output_weight is not None

        draft = self._require_draft_model()
        if int(draft.config.vocab_size) != int(embedding_weight.shape[0]):
            raise ValueError("NCP target and draft vocabulary sizes do not match")
        proposal_block_size = max(proposal_counts)
        runtime_max_block_size = int(self._draft_runtime_max_block_size)
        if not 0 < proposal_block_size <= runtime_max_block_size:
            raise RuntimeError(
                "NCP DFlash proposal block exceeds the loaded runtime block: "
                f"proposal={proposal_block_size} runtime={runtime_max_block_size}"
            )
        # Keep the trained checkpoint contract immutable. The remote model
        # computes its fixed block and the selector consumes only the scheduler-
        # selected prefix below. A future variable-width kernel can optimize
        # this without mutating shared model configuration at runtime.
        anchor_embeddings = embedding_weight[anchor_ids].unsqueeze(1)
        mask_embedding = embedding_weight[int(draft.config.mask_token_id)]
        anchor_positions = torch.tensor(
            [[prefix_length - 1] for prefix_length in prefix_lengths],
            dtype=torch.long,
            device=self.device,
        )
        sequence_lengths = torch.tensor(
            prefix_lengths,
            dtype=torch.long,
            device=self.device,
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=self.dtype)
            if self.device.type == "cuda"
            and self.dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        cache_counters_before = [
            (
                int(
                    getattr(
                        layer,
                        "_ncp_dflash_context_kv_projected_tokens",
                        0,
                    )
                ),
                int(
                    getattr(
                        layer,
                        "_ncp_dflash_context_kv_reused_tokens",
                        0,
                    )
                ),
            )
            for layer in draft.layers
        ]
        self._last_sparse_context_projection = {
            "enabled": False,
            "projected_target_tokens": 0,
            "total_causal_target_tokens": 0,
        }
        for layer in draft.layers:
            if self.context_kv_cache:
                layer._ncp_dflash_request_ids = request_ids
                layer._ncp_dflash_causal_lengths = [
                    prefix_length - 1 for prefix_length in prefix_lengths
                ]
                layer._ncp_dflash_context_offsets = None
        with torch.inference_mode(), autocast:
            try:
                if self.sparse_context_projection:
                    output = self._forward_sparse_context(
                        draft,
                        contexts,
                        request_ids=request_ids,
                        prefix_lengths=prefix_lengths,
                        anchor_embeddings=anchor_embeddings,
                        mask_embedding=mask_embedding,
                        anchor_positions=anchor_positions,
                        hlm_hidden_states=torch.cat(hlm_states, dim=0),
                    )
                else:
                    output = draft(
                        aux_hidden_states=self._pad_context_batch(contexts),
                        anchor_embeddings=anchor_embeddings,
                        mask_embedding=mask_embedding,
                        anchor_positions=anchor_positions,
                        sequence_lengths=sequence_lengths,
                        hlm_hidden_states=torch.cat(hlm_states, dim=0),
                        return_dict=True,
                    )
            finally:
                for layer in draft.layers:
                    for attribute in (
                        "_ncp_dflash_request_ids",
                        "_ncp_dflash_causal_lengths",
                        "_ncp_dflash_context_offsets",
                    ):
                        if hasattr(layer, attribute):
                            delattr(layer, attribute)
            selected = self._path_selector_batch(
                draft,
                output.last_hidden_state[:, 0],
                output_weight,
                embedding_weight,
                anchor_ids,
                proposal_counts,
            )
        projected_tokens = 0
        reused_tokens = 0
        for layer, (projected_before, reused_before) in zip(
            draft.layers,
            cache_counters_before,
            strict=True,
        ):
            projected_tokens += (
                int(
                    getattr(
                        layer,
                        "_ncp_dflash_context_kv_projected_tokens",
                        0,
                    )
                )
                - projected_before
            )
            reused_tokens += (
                int(
                    getattr(
                        layer,
                        "_ncp_dflash_context_kv_reused_tokens",
                        0,
                    )
                )
                - reused_before
            )
        self._last_context_kv_cache_stats = {
            "enabled": self.context_kv_cache,
            "projected_tokens": projected_tokens,
            "reused_tokens": reused_tokens,
            "sparse_context_projection": self._last_sparse_context_projection,
            "proposal_block_size": proposal_block_size,
            "runtime_block_size": int(draft.config.block_size),
        }
        return selected

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        del (
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_rejected,
            next_prefill_tokens,
            temperature,
            seeds,
            dp_sync,
            skip_attn_for_dummy_run,
            mm_inputs,
            is_profile,
        )
        num_reqs = int(input_batch.num_reqs)
        result = self.draft_tokens[:num_reqs]
        result.fill_(-1)
        self._require_draft_model()
        if dummy_run:
            return result
        if input_batch.req_ids and all(
            str(request_id).startswith("_warmup_") for request_id in input_batch.req_ids
        ):
            # Kernel warmup schedules the configured maximum width directly
            # and does not consume per-row proposal lengths from the scheduler.
            # Use valid synthetic IDs; no warmup output is user-visible.
            result.zero_()
            return result

        self._prune_context_kv_cache(
            [str(request_id) for request_id in input_batch.req_ids]
        )
        self._last_context_kv_cache_stats = {
            "enabled": self.context_kv_cache,
            "projected_tokens": 0,
            "reused_tokens": 0,
        }

        sampled_counts = num_sampled[:num_reqs].detach().cpu().tolist()
        active_decode_batch_size = sum(
            int(sampled_count) > 0 for sampled_count in sampled_counts
        )
        active_batch_width = self._active_batch_width(active_decode_batch_size)
        slots = input_batch.idx_mapping[:num_reqs].to(
            device=last_sampled.device,
            dtype=torch.long,
        )
        anchor_ids = last_sampled[slots]
        if anchor_ids.ndim == 2 and int(anchor_ids.shape[1]) == 1:
            anchor_ids = anchor_ids[:, 0]
        if anchor_ids.ndim != 1:
            raise RuntimeError("NCP DFlash expected one sampled anchor per request")
        anchor_ids = anchor_ids.to(dtype=torch.long)
        target = self._require_target_model()
        eligible_rows = []
        eligible_request_ids = []
        eligible_anchor_ids = []
        proposal_counts = []
        for row_index, (request_id, sampled_count) in enumerate(
            zip(input_batch.req_ids, sampled_counts, strict=True)
        ):
            if int(sampled_count) <= 0:
                continue
            state = target.request_states.states.get(str(request_id))
            if state is None:
                continue
            proposal_count = self.safe_proposal_count(
                int(state.next_token_position) + 1,
                proposal_cap=active_batch_width,
            )
            if proposal_count <= 0:
                continue
            eligible_rows.append(row_index)
            eligible_request_ids.append(str(request_id))
            eligible_anchor_ids.append(anchor_ids[row_index])
            proposal_counts.append(proposal_count)

        eligible_before_gate = len(eligible_rows)
        requested_proposal_tokens = sum(proposal_counts)
        skip_reason = self._proposal_batch_skip_reason(proposal_counts)
        if skip_reason is not None:
            eligible_rows = []
            eligible_request_ids = []
            eligible_anchor_ids = []
            proposal_counts = []

        if eligible_rows:
            selected = self._propose_rows(
                eligible_request_ids,
                torch.stack(eligible_anchor_ids),
                proposal_counts,
            )
            for selected_row, output_row, proposal_count in zip(
                selected,
                eligible_rows,
                proposal_counts,
                strict=True,
            ):
                result[output_row, :proposal_count].copy_(selected_row[:proposal_count])
            if not self._logged_first_proposal:
                logger.info(
                    "NCP DFlash emitted its first proposal batch: requests=%d "
                    "tokens=%d",
                    len(eligible_rows),
                    sum(proposal_counts),
                )
                self._logged_first_proposal = True
        self._last_proposal_stats = {
            "active_decode_batch_size": active_decode_batch_size,
            "active_batch_width": active_batch_width,
            "eligible_before_gate": eligible_before_gate,
            "eligible_request_count": len(eligible_rows),
            "requested_proposal_tokens": requested_proposal_tokens,
            "emitted_proposal_tokens": sum(proposal_counts),
            "skip_reason": skip_reason,
            "context_kv_cache": self._last_context_kv_cache_stats,
        }
        return result
