# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run an NCP-OLMo Hugging Face checkpoint with vLLM."""

import argparse
import os


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_attention_config(mode: str) -> dict[str, object] | None:
    if mode == "auto":
        return None
    if mode == "flash-attn":
        return {"backend": "FLASH_ATTN", "flash_attn_version": 3}
    if mode == "mixed-flashinfer":
        return {
            "backend": "FLASH_ATTN",
            "flash_attn_version": 3,
            "use_trtllm_attention": False,
            "backend_per_kind": {
                "full_attention": "FLASHINFER",
                "sliding_window": "FLASH_ATTN",
            },
        }
    raise ValueError(f"Unsupported attention mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path or Hub ID of a pure HF NCP-OLMo model")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--draft-model",
        help="Path or Hub ID of a conceptlm_dflash Hugging Face checkpoint.",
    )
    parser.add_argument("--num-speculative-tokens", type=int, default=2)
    parser.add_argument(
        "--dflash-active-batch-widths",
        help=(
            "Optional upper-bound policy such as '1:8,2:8,4:4,8:2'. "
            "The policy counts active decode requests, not prefill rows."
        ),
    )
    parser.add_argument(
        "--dflash-verification-mode",
        choices=("sequential_exact", "intra_chunk_exact", "segmented_kv_approx"),
        default="sequential_exact",
        help=(
            "NCP DFlash state contract. Cross-chunk segmented_kv_approx is an "
            "explicit experimental opt-in."
        ),
    )
    parser.add_argument(
        "--dflash-attention-backend",
        choices=("sdpa", "flash_varlen", "flex_attention"),
        default="sdpa",
        help=(
            "Draft attention backend. SDPA is the correctness-first default; "
            "FlashAttention varlen is an explicit performance opt-in."
        ),
    )
    parser.add_argument(
        "--dflash-context-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache request-local draft context projections.",
    )
    parser.add_argument(
        "--dflash-sparse-context-projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Project only uncached request-local context suffixes.",
    )
    parser.add_argument(
        "--dflash-min-eligible-batch",
        type=positive_int,
        default=1,
        help="Skip DFlash unless at least this many decode rows are eligible.",
    )
    parser.add_argument(
        "--dflash-min-proposal-tokens-per-row",
        type=positive_int,
        default=1,
        help="Skip DFlash for rows whose draft width is below this value.",
    )
    parser.add_argument(
        "--dflash-min-proposal-tokens-per-batch",
        type=positive_int,
        default=1,
        help="Skip DFlash unless the eligible rows propose this many tokens.",
    )
    parser.add_argument(
        "--attention-mode",
        choices=("auto", "flash-attn", "mixed-flashinfer"),
        default="auto",
        help=(
            "Attention backend selection. The mixed mode uses FlashInfer for "
            "full-attention layers and FlashAttention for sliding-window layers."
        ),
    )
    parser.add_argument(
        "--flashinfer-sampler",
        action="store_true",
        help="Use FlashInfer top-k/top-p sampling kernels.",
    )
    return parser.parse_args()


def configure_dflash_env(args: argparse.Namespace) -> None:
    if not args.draft_model:
        return
    if (
        args.dflash_attention_backend == "flash_varlen"
        and not args.dflash_context_kv_cache
    ):
        raise ValueError("flash_varlen requires the DFlash context KV cache")

    settings = {
        "NCP_OLMO_DFLASH_VERIFICATION_MODE": args.dflash_verification_mode,
        "NCP_OLMO_DFLASH_ATTENTION_BACKEND": args.dflash_attention_backend,
        "NCP_OLMO_DFLASH_CONTEXT_KV_CACHE": str(int(args.dflash_context_kv_cache)),
        "NCP_OLMO_DFLASH_SPARSE_CONTEXT_PROJECTION": str(
            int(args.dflash_sparse_context_projection)
        ),
        "NCP_OLMO_DFLASH_MIN_ELIGIBLE_BATCH": str(args.dflash_min_eligible_batch),
        "NCP_OLMO_DFLASH_MIN_PROPOSAL_TOKENS_PER_ROW": str(
            args.dflash_min_proposal_tokens_per_row
        ),
        "NCP_OLMO_DFLASH_MIN_PROPOSAL_TOKENS_PER_BATCH": str(
            args.dflash_min_proposal_tokens_per_batch
        ),
    }
    if args.dflash_active_batch_widths:
        settings["NCP_OLMO_DFLASH_ACTIVE_BATCH_WIDTHS"] = (
            args.dflash_active_batch_widths
        )
    os.environ.update(settings)


def main() -> None:
    args = parse_args()
    if args.flashinfer_sampler:
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1"
    configure_dflash_env(args)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        enforce_eager=True,
        enable_prefix_caching=False,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        attention_config=build_attention_config(args.attention_mode),
        speculative_config=(
            {
                "model": args.draft_model,
                "method": "dflash",
                "num_speculative_tokens": args.num_speculative_tokens,
            }
            if args.draft_model
            else None
        ),
        seed=args.seed,
    )
    output = llm.generate(
        [args.prompt],
        SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ),
        use_tqdm=False,
    )[0]
    print(output.outputs[0].text)


if __name__ == "__main__":
    main()
