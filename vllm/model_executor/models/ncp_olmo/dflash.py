# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NCP-OLMo DFlash configuration and target-model rendezvous helpers."""

from __future__ import annotations

import weakref
from typing import Any

import torch
import torch.nn as nn

DRAFT_ARCHITECTURE = "DFlashConceptLMDFlashModel"
VERIFICATION_MODES = {
    "sequential_exact",
    "intra_chunk_exact",
    "segmented_kv_approx",
}

_target_model_ref: weakref.ReferenceType[nn.Module] | None = None


def is_ncp_dflash_config(vllm_config: Any) -> bool:
    """Return whether this is the supported NCP target/drafter pairing."""

    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is None or speculative_config.method != "dflash":
        return False
    draft_config = getattr(speculative_config, "draft_model_config", None)
    architectures = getattr(draft_config, "architectures", ()) or ()
    return DRAFT_ARCHITECTURE in architectures


def original_draft_config(vllm_config: Any) -> Any:
    """Return the ConceptLM config wrapped by vLLM's DFlash EAGLEConfig."""

    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.draft_model_config is None:
        raise ValueError("NCP DFlash requires a draft model configuration")
    wrapped = speculative_config.draft_model_config.hf_config
    return getattr(wrapped, "model", wrapped)


def validate_ncp_dflash_config(vllm_config: Any) -> tuple[int, ...]:
    """Validate the trained NCP DFlash contract and return target layer IDs."""

    if not is_ncp_dflash_config(vllm_config):
        raise ValueError(
            f"NCP DFlash requires draft architecture {DRAFT_ARCHITECTURE!r}"
        )
    config = original_draft_config(vllm_config)
    if getattr(config, "model_type", None) != "conceptlm_dflash":
        raise ValueError("NCP DFlash requires model_type='conceptlm_dflash'")
    if getattr(config, "proposal_method", None) != "path_selector":
        raise ValueError("NCP DFlash requires proposal_method='path_selector'")
    if getattr(config, "hlm_conditioning", None) != "causal_residual":
        raise ValueError("NCP DFlash requires hlm_conditioning='causal_residual'")
    target_layer_ids = tuple(int(value) for value in config.target_layer_ids)
    if not target_layer_ids:
        raise ValueError("NCP DFlash target_layer_ids must not be empty")
    if len(set(target_layer_ids)) != len(target_layer_ids):
        raise ValueError("NCP DFlash target_layer_ids must be unique")
    if int(config.block_size) < int(
        vllm_config.speculative_config.num_speculative_tokens
    ):
        raise ValueError(
            "num_speculative_tokens exceeds the trained NCP DFlash block_size"
        )
    return target_layer_ids


def register_ncp_dflash_target(model: nn.Module) -> None:
    """Register the process-local target consumed by the V2 speculator."""

    global _target_model_ref
    existing = None if _target_model_ref is None else _target_model_ref()
    if existing is not None and existing is not model:
        raise RuntimeError("only one NCP DFlash target may be active per process")
    _target_model_ref = weakref.ref(model)


def get_ncp_dflash_target() -> nn.Module:
    """Return the live process-local NCP target or fail closed."""

    target = None if _target_model_ref is None else _target_model_ref()
    if target is None:
        raise RuntimeError("the NCP DFlash target model has not been constructed")
    return target


class DFlashConceptLMDFlashModel(nn.Module):
    """Registry-only marker for the remote-code NCP DFlash checkpoint.

    The V2 NCP speculator loads the self-contained Hugging Face draft model
    directly. This class lets ``ModelConfig`` inspect the wrapped architecture
    without accidentally routing it through the incompatible generic Qwen
    DFlash implementation.
    """

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__()
        del vllm_config, prefix
        raise RuntimeError(
            "DFlashConceptLMDFlashModel is loaded by NCPDFlashSpeculator, "
            "not by the generic vLLM model loader"
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        raise NotImplementedError

