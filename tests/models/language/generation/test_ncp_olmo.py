# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import pytest
import torch

from vllm import SamplingParams

from ...utils import check_logprobs_close, check_outputs_equal

MODEL = os.environ.get("NCP_OLMO_TEST_MODEL", "")
DFLASH_MODEL = os.environ.get("NCP_OLMO_DFLASH_TEST_MODEL", "")
PROMPTS = [
    "A short request checks the model state.",
    "A longer request checks that batching preserves the request-local HLM state. " * 8,
    "Name the capital of France.",
]
MAX_NUM_SEQS = 2
PRESSURE_PROMPTS = [
    "The following numbers of the sequence "
    + ", ".join(str(i) for i in range(10))
    + " are:",
    "In one word, the capital of France is ",
] + [f"Tell me about the number {index}: " for index in range(32)]

pytestmark = pytest.mark.skipif(
    not MODEL,
    reason="NCP_OLMO_TEST_MODEL must point to a pure-HF NCP-OLMo checkpoint",
)


def _check_exact_vllm_outputs(
    reference: list[tuple[list[int], str, Any]],
    candidate: list[tuple[list[int], str, Any]],
    *,
    reference_name: str,
    candidate_name: str,
) -> None:
    """Require exact tokens/text and matching top-k choices."""

    check_outputs_equal(
        outputs_0_lst=[(output[0], output[1]) for output in reference],
        outputs_1_lst=[(output[0], output[1]) for output in candidate],
        name_0=reference_name,
        name_1=candidate_name,
    )
    check_logprobs_close(
        outputs_0_lst=reference,
        outputs_1_lst=candidate,
        name_0=reference_name,
        name_1=candidate_name,
        always_check_logprobs=True,
    )


def _generate_hf_greedy_logprobs(
    hf_model: Any,
    prompts: list[str],
    max_tokens: int,
    num_logprobs: int,
) -> list[tuple[list[int], str, list[dict[int, float]]]]:
    """Collect HF generation logprobs without relying on hidden states.

    NCP-OLMo's pure-HF generation path returns per-step scores but intentionally
    does not materialize generation hidden states. Consuming ``scores`` also
    compares the logits after the same generation-time processors that selected
    each greedy token.
    """
    outputs = []
    for inputs in hf_model.get_inputs(prompts):
        generated = hf_model.model.generate(
            **hf_model.wrap_device(inputs),
            tokenizer=hf_model.tokenizer,
            use_cache=True,
            do_sample=False,
            max_new_tokens=max_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )

        scores = generated.scores
        output_ids = generated.sequences[0, -len(scores) :].tolist() if scores else []
        output_logprobs = []
        for score in scores:
            logprobs = torch.log_softmax(score[0].float(), dim=-1)
            topk = logprobs.topk(num_logprobs)
            output_logprobs.append(
                dict(
                    zip(
                        topk.indices.tolist(),
                        topk.values.tolist(),
                        strict=True,
                    )
                )
            )

        outputs.append(
            (
                output_ids,
                hf_model.tokenizer.decode(output_ids),
                output_logprobs,
            )
        )

    return outputs


@pytest.mark.parametrize("max_tokens", [16])
@pytest.mark.parametrize("num_logprobs", [5])
def test_greedy_logprobs(
    hf_runner,
    vllm_runner,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    with hf_runner(
        MODEL,
        dtype="bfloat16",
        model_kwargs={"attn_implementation": "eager"},
    ) as hf_model:
        hf_outputs = _generate_hf_greedy_logprobs(
            hf_model,
            PROMPTS,
            max_tokens,
            num_logprobs,
        )

    with vllm_runner(
        MODEL,
        dtype="bfloat16",
        max_num_seqs=MAX_NUM_SEQS,
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            PROMPTS,
            max_tokens,
            num_logprobs,
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
        always_check_logprobs=True,
    )


@pytest.mark.parametrize("max_tokens", [16])
@pytest.mark.parametrize("num_logprobs", [5])
def test_batched_matches_sequential(
    vllm_runner,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    with vllm_runner(
        MODEL,
        dtype="bfloat16",
        max_num_seqs=MAX_NUM_SEQS,
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
    ) as vllm_model:
        sequential_outputs = [
            vllm_model.generate_greedy_logprobs(
                [prompt],
                max_tokens,
                num_logprobs,
            )[0]
            for prompt in PROMPTS
        ]
        batched_outputs = vllm_model.generate_greedy_logprobs(
            PROMPTS,
            max_tokens,
            num_logprobs,
        )

    _check_exact_vllm_outputs(
        sequential_outputs,
        batched_outputs,
        reference_name="sequential_vllm",
        candidate_name="batched_vllm",
    )


@pytest.mark.parametrize("max_tokens", [40])
@pytest.mark.parametrize("num_logprobs", [5])
def test_chunked_prefill_preemption_and_refill_match_sequential(
    vllm_runner,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    """Preserve request-local HLM state across scheduler pressure."""

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        min_tokens=20,
        logprobs=num_logprobs,
    )

    with vllm_runner(
        MODEL,
        dtype="bfloat16",
        max_model_len=512,
        max_num_batched_tokens=48,
        # NCP uses four KV-cache groups. Scale the upstream preemption fixture's
        # 33-block cache by that factor: one 512-token request remains legal,
        # while 34 concurrent requests force scheduler recompute/preemption.
        num_gpu_blocks_override=132,
        disable_log_stats=False,
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
    ) as vllm_model:
        sequential_outputs = [
            vllm_model.generate_w_logprobs(
                [prompt],
                sampling_params=sampling_params,
            )[0]
            for prompt in PRESSURE_PROMPTS
        ]
        metrics_before = vllm_model.llm.get_metrics()
        pressured_outputs = vllm_model.generate_w_logprobs(
            PRESSURE_PROMPTS,
            sampling_params=sampling_params,
        )
        metrics_after = vllm_model.llm.get_metrics()

    _check_exact_vllm_outputs(
        sequential_outputs,
        pressured_outputs,
        reference_name="sequential_vllm",
        candidate_name="pressured_vllm",
    )
    preemptions_before = next(
        (
            metric.value
            for metric in metrics_before
            if metric.name == "vllm:num_preemptions"
        ),
        0,
    )
    preemptions_after = next(
        (
            metric.value
            for metric in metrics_after
            if metric.name == "vllm:num_preemptions"
        ),
        0,
    )
    assert preemptions_after > preemptions_before


@pytest.mark.skipif(
    not DFLASH_MODEL,
    reason="NCP_OLMO_DFLASH_TEST_MODEL must point to the matching draft checkpoint",
)
def test_dflash_continuous_refill_matches_target_only(
    monkeypatch: pytest.MonkeyPatch,
    vllm_runner,
) -> None:
    """A finished slot is refilled while a longer request remains active."""

    prompts = [
        "A",
        "Continue this longer request with a few factual words.",
        "B",
        "Write a short sentence about Paris.",
        "C",
    ]
    max_tokens = [1, 12, 2, 8, 3]
    sampling_params = [
        SamplingParams(
            temperature=0.0,
            max_tokens=count,
            ignore_eos=True,
        )
        for count in max_tokens
    ]
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")

    def generate(speculative_config: dict[str, object] | None) -> list[list[int]]:
        with vllm_runner(
            MODEL,
            dtype="bfloat16",
            max_num_seqs=2,
            enforce_eager=True,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            speculative_config=speculative_config,
            kernel_config={
                "enable_jit_warmup": False,
                "enable_cutedsl_warmup": False,
            },
        ) as model:
            outputs = model.llm.generate(
                prompts,
                sampling_params,
                use_tqdm=False,
            )
        return [list(output.outputs[0].token_ids) for output in outputs]

    target_only = generate(None)
    with_dflash = generate(
        {
            "model": DFLASH_MODEL,
            "method": "dflash",
            "num_speculative_tokens": 8,
            "num_speculative_tokens_per_batch_size": [
                (1, 1, 8),
                (2, 2, 8),
                (3, 4, 4),
                (5, 8, 2),
            ],
            "ncp_dflash_verification_mode": "sequential_exact",
        }
    )

    assert [len(output) for output in with_dflash] == max_tokens
    assert with_dflash == target_only


@pytest.mark.skipif(
    not DFLASH_MODEL,
    reason="NCP_OLMO_DFLASH_TEST_MODEL must point to the matching draft checkpoint",
)
def test_dflash_preemption_matches_target_only(
    monkeypatch: pytest.MonkeyPatch,
    vllm_runner,
) -> None:
    """Preserve exact target tokens through scheduler recompute preemption."""

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=24,
        min_tokens=20,
        logprobs=5,
    )
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")

    def generate(
        speculative_config: dict[str, object] | None,
    ) -> tuple[list[tuple[list[int], str, Any]], float, float]:
        with vllm_runner(
            MODEL,
            dtype="bfloat16",
            max_model_len=512,
            max_num_batched_tokens=48,
            num_gpu_blocks_override=132,
            disable_log_stats=False,
            enforce_eager=True,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            speculative_config=speculative_config,
            kernel_config={
                "enable_jit_warmup": False,
                "enable_cutedsl_warmup": False,
            },
        ) as model:
            metrics_before = model.llm.get_metrics()
            outputs = model.generate_w_logprobs(
                PRESSURE_PROMPTS,
                sampling_params=sampling_params,
            )
            metrics_after = model.llm.get_metrics()

        def preemptions(metrics: list[Any]) -> float:
            return next(
                (
                    metric.value
                    for metric in metrics
                    if metric.name == "vllm:num_preemptions"
                ),
                0,
            )

        return outputs, preemptions(metrics_before), preemptions(metrics_after)

    target_only, _, _ = generate(None)
    with_dflash, preemptions_before, preemptions_after = generate(
        {
            "model": DFLASH_MODEL,
            "method": "dflash",
            "num_speculative_tokens": 8,
            "num_speculative_tokens_per_batch_size": [
                (1, 1, 8),
                (2, 2, 8),
                (3, 4, 4),
                (5, 8, 2),
            ],
            "ncp_dflash_verification_mode": "sequential_exact",
        }
    )

    _check_exact_vllm_outputs(
        target_only,
        with_dflash,
        reference_name="target_only",
        candidate_name="dflash_preempted",
    )
    assert preemptions_after > preemptions_before
