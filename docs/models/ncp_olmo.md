# NCP-ArchPreview

NCP-ArchPreview is a decoder-only language model with token-level encoder and
decoder towers around a chunk-level high-level model (HLM).

## Checkpoint contract

The implementation accepts pure Hugging Face checkpoints with:

- `architectures: ["NCPOlmo3ForCausalLM"]`
- `model_type: "ncp_olmo3"`
- `weight_key_format: "huggingface_state_dict"`
- `conceptlm_chunk_merge_method: "meanpooling"`
- `conceptlm_layer_norm_option: "normed_add"`
- `conceptlm_hlm_attention_mode: "backbone_window"`
- `conceptlm_v22_vq_merge_mode: "raw_logits"`
- `conceptlm_v21_dd_self_dd_mode: "dd"`
- `qk_layernorm: true` with full-hidden Q/K normalization
- `position_embedding_type: "rope"` and `rotary_interleaved: false`
- `conceptlm_shift_feature: true` and `conceptlm_hlm_ffn_hidden_size: null`
- split `q_proj`, `k_proj`, `v_proj` tensors
- split `gate_proj` and `up_proj` tensors

Native Megatron/DCP state-dict keys are intentionally not accepted. Convert a
training checkpoint into this standalone HF schema before loading it with vLLM.

All non-DFlash checkpoints in the public
[NCP-ArchPreview collection](https://huggingface.co/collections/ArchSpace-Collection/ncp-archpreview)
are supported, including Stage 1 (and its intermediate checkpoints) and Stage 2
v1/v2/v3. The public
[Stage 2 DFlash2 NCPFlash checkpoint](https://huggingface.co/ArchSpace-Collection/NCP_ArchPreview_dolma3_8.9B_Stage2_DFlash2_NCPFlash)
is supported only by this DFlash integration and is bound to the Stage 2 v1
target.
The model-registry initialization test uses the Stage 2 v1 configuration with
a contract-preserving layer-count override and vLLM's dummy loader; a separate
dummy-weight checkpoint is not required.

The loader maps those tensors into vLLM fused QKV and SwiGLU parameters. The
weight-loading test checks every tensor in the NCP-ArchPreview graph shared by
the published Stage 1 and Stage 2 checkpoints.

## Attention backends

The token encoder and decoder instantiate vLLM `Attention` layers. Their KV
cache therefore uses the regular vLLM block table and PagedAttention lifecycle.
The concrete CUDA implementation is not fixed in the model. vLLM selects the
compatible backend for the device and can use FlashAttention decode kernels
when available.

FlashInfer sampling is independent of the attention backend. Enable its
top-k/top-p kernels with `VLLM_USE_FLASHINFER_SAMPLER=1` or use the offline
example's `--flashinfer-sampler` option. The FlashInfer package and AOT kernel
cache must match the vLLM CUDA and PyTorch build.

NCP-ArchPreview interleaves full and sliding-window attention. On Hopper (SM90), vLLM
currently rejects FlashInfer attention for sliding-window groups because that
combination is not reliable. A supported mixed configuration keeps
sliding-window groups on FlashAttention 3 and assigns only full-attention
groups to FlashInfer:

```python
attention_config = {
    "backend": "FLASH_ATTN",
    "flash_attn_version": 3,
    "use_trtllm_attention": False,
    "backend_per_kind": {
        "full_attention": "FLASHINFER",
        "sliding_window": "FLASH_ATTN",
    },
}
```

Disabling TRTLLM attention in this configuration selects native FlashInfer and
avoids a remote cubin lookup in air-gapped deployments. Do not force every
attention group to FlashInfer on SM90 until the upstream sliding-window issue
is resolved.

The HLM currently retains request-owned dense state. Its add, remove, reorder,
and replay lifecycle is connected to the V2 runner through the model-state
interface, while its tensors remain outside the vLLM block manager. Prefix
caching is rejected because a cached token prefix does not yet carry the
corresponding HLM state snapshot. Enabling it requires a cache payload for the
pending encoder fragment, per-layer HLM K/V and raw states, and predicted
concepts at the same token boundary, together with copy-on-write ownership when
requests share that prefix. Token-tower KV blocks alone are insufficient.
Tensor parallelism and pipeline parallelism remain disabled in this first
implementation. Quantized checkpoints are also rejected until their fused
weight mappings and numerical parity are validated.

## NCP DFlash speculative decoding

NCP-ArchPreview supports its matching self-contained Hugging Face DFlash
checkpoint. The draft must declare `model_type: "conceptlm_dflash"`, use the
`ConceptLMDFlashModel` architecture, `path_selector` proposals,
`causal_residual` HLM conditioning, and target-layer IDs compatible with the
NCP target. Other speculative checkpoints fail closed rather than being sent
through the NCP-specific path.

The target captures the routed decoder layers requested by the draft. After
rejection sampling, request-local token, HLM, and captured-feature state is
truncated or restored and replayed to the accepted boundary before the next
proposal. Prefix caching must still remain disabled.

`sequential_exact` is the default verification mode. It proposes at most one
token and never completes an HLM chunk inside one speculative verifier pass.
`intra_chunk_exact` widens proposals within the same safe boundary.
`segmented_kv_approx` permits cross-chunk proposals and uses transactional
rollback/replay; it is an explicit experimental opt-in because proposal-side
HLM conditioning and batched target numerics can differ from target-only
decoding. Target verification remains authoritative in every mode.

## Offline inference

```bash
python examples/offline_inference/ncp_olmo.py /path/to/pure-hf-model \
  --prompt "The capital of France is" \
  --max-tokens 32
```

Run with a matching DFlash checkpoint:

```bash
python examples/offline_inference/ncp_olmo.py /path/to/pure-hf-model \
  --draft-model /path/to/conceptlm-dflash \
  --num-speculative-tokens 2
```

The DFlash checkpoint supplies executable Hugging Face model code, so loading
it is covered by the example's explicit `trust_remote_code=True`. Review the
checkpoint code before using an untrusted source. vLLM constructs and loads the
draft during the runner's normal model-loading phase, so its allocation, dummy
initialization, and peak memory are accounted with the target rather than being
hidden behind a lazy first request. To reproduce the experimental cross-chunk
mode, add `--dflash-verification-mode segmented_kv_approx`.
An experimental active-batch schedule uses a maximum width of eight and caps it
by the number of active decode requests: `1 -> 8`, `2 -> 8`, `3-4 -> 4`, and
`5-8 -> 2`. This is an opt-in performance heuristic for the approximate
cross-chunk state contract; it is not a quality preset or a checkpoint property:

```bash
python examples/offline_inference/ncp_olmo.py /path/to/pure-hf-model \
  --draft-model /path/to/conceptlm-dflash \
  --num-speculative-tokens 8 \
  --dflash-verification-mode segmented_kv_approx \
  --dflash-attention-backend flash_varlen \
  --dflash-active-batch-widths '1:8,2:8,4:4,8:2'
```

The active batch is recomputed from decode rows on every proposal step, so
request completion and refill immediately select a new width. The scheduler
and proposer share vLLM's typed dynamic speculative-decoding schedule; invalid
suffix entries remain `-1` and are trimmed before target verification. The
remote drafter keeps its trained fixed block size immutable and the selector
consumes the scheduled prefix. A future variable-width draft kernel can reduce
this compute without mutating shared checkpoint configuration at runtime.
Exact modes still cap every row at the remaining safe positions in its current
HLM chunk. With the current four-token chunk contract this normally emits one
or two draft tokens, even when the configured adaptive width is larger. Only
`segmented_kv_approx` uses the full cross-chunk adaptive widths.

Three typed speculative-config fields can bypass proposal work when the active
decode set is too small. Their defaults are all one, which preserves the normal
proposal path: `ncp_dflash_min_eligible_batch`,
`ncp_dflash_min_proposal_tokens_per_row`, and
`ncp_dflash_min_proposal_tokens_per_batch`.

The first gate applies when the engine is configured for more than one request.
The other two gates remove rows with too little safe work and then require a
minimum total proposal budget across the remaining rows. Tune them only from
matched target/DFlash traces; they trade draft coverage for lower overhead.
The offline example exposes the same settings as
`--dflash-min-eligible-batch`, `--dflash-min-proposal-tokens-per-row`, and
`--dflash-min-proposal-tokens-per-batch`.

The DFlash attention path keeps request-owned context K/V and projects only
the newly committed target-feature suffix on later proposal steps. Cache keys
are request IDs rather than row indices, so continuous-batch removal, refill,
and row reordering cannot exchange state. Released requests are pruned. Set
`ncp_dflash_context_kv_cache=false` and
`ncp_dflash_sparse_context_projection=false` together only for a reference run
that recomputes the complete draft context. The equivalent example options are
`--no-dflash-context-kv-cache` and `--no-dflash-sparse-context-projection`.

The draft uses an inference-only SDPA implementation of the checkpoint's
FlexAttention visibility rule by default. Set the typed
`ncp_dflash_attention_backend` field to `flash_varlen` to pack request-local
context K/V and run the same one-anchor visibility rule with FlashAttention
Varlen. This backend requires the request-local context K/V cache. It is opt-in:
kernel-level numerical differences can change draft accept/reject telemetry
even when exact-mode final target tokens are unchanged. Use `flex_attention`
only when debugging the checkpoint's original attention implementation. The
offline example exposes these choices as `--dflash-attention-backend`.

To use FlashInfer sampling while keeping vLLM's default attention selection:

```bash
python examples/offline_inference/ncp_olmo.py /path/to/pure-hf-model \
  --flashinfer-sampler
```

To test the mixed FlashAttention/FlashInfer attention configuration:

```bash
python examples/offline_inference/ncp_olmo.py /path/to/pure-hf-model \
  --attention-mode mixed-flashinfer \
  --flashinfer-sampler
```

The example uses the default V2 model runner. Request-local HLM state is
initialized, reordered, and released by the V2 model-state lifecycle. The first
version requires `enforce_eager=True` and prefix caching to remain disabled.
CUDA graph capture will be enabled only after request-scoped HLM state has an
explicit graph-safe contract.

## Validation before upstream review

The registry test uses the public Stage 2 v1 checkpoint with small dummy-weight
overrides. Before opening a pull request, run:

- model registry import and dummy-weight initialization
- full pure-HF checkpoint loading
- Hugging Face versus vLLM logprob and greedy-output parity
- mixed-length continuous batching with more queued requests than
  `max_num_seqs`, in-flight slot refill/reordering, and request cleanup
- target-only versus DFlash token parity in `sequential_exact` mode
- rejection rollback at same-chunk and cross-chunk boundaries
- DFlash acceptance and throughput on the intended target checkpoint
- chunked-prefill and preemption/recompute tests
- GPU confirmation of the selected PagedAttention/FlashAttention backend
- FlashInfer-sampler parity and mixed-backend parity on a sliding-window model
