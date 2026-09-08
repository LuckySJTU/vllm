# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NCP-OLMo DFlash configuration and draft-model wrapper."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from vllm.model_executor.models.utils import AutoWeightsLoader

DRAFT_ARCHITECTURE = "DFlashConceptLMDFlashModel"
VERIFICATION_MODES = {
    "sequential_exact",
    "intra_chunk_exact",
    "segmented_kv_approx",
}


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


class DFlashConceptLMDFlashModel(nn.Module):
    """Load the self-contained NCP DFlash checkpoint through vLLM.

    The checkpoint provides a Transformers remote-code model. Constructing it
    here keeps draft allocation, checkpoint loading, dummy initialization, and
    peak-memory accounting inside vLLM's normal model-loader lifecycle.
    """

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__()
        del prefix
        from transformers import AutoModel

        config = original_draft_config(vllm_config)
        self.model = AutoModel.from_config(
            config,
            trust_remote_code=True,
        )
        self.config = self.model.config

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load the flat remote-code checkpoint with vLLM tracking enabled."""

        loaded = AutoWeightsLoader(self.model).load_weights(weights)
        return {f"model.{name}" for name in loaded}

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self.model(*args, **kwargs)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        raise NotImplementedError
