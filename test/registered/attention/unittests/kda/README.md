# KDA Attention Capability Matrix

This folder covers KDA (Kimi Delta Attention) linear attention. The actual
path drives `KDAAttnBackend` through `HybridLinearAttnBackend` on a
`RadixLinearAttention` layer. Expected outputs come from an independent
pure-PyTorch sigmoid-gated delta-rule reference using
`KimiLinearCacheParams` / `KimiLinearStateShape` (per-head-channel `dt_bias`,
`silu` activation on conv1d output, per-channel gate broadcast), not the KDA
Triton kernel.

## Coverage Matrix

Columns are runner modes; rows are the linear-attention kernel backend. Cells use:
- **✓ \<variants\>** — exercised, with the config variants listed in the cell
- **—** — not applicable / not exercised
- **blocked: \<reason\>** — production-unsupported, not a follow-up
- **deferred: \<reason\>** — could land later, currently disabled

| Linear-attn kernel | Eager Phase 2 | CG decode | PCG extend | BCG extend | Verify eager | Verify CG | DE eager | DE CG | DE-V2 CG | EAGLE-draft runner | EAGLE-DE runner | FKVMTP runner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `triton` | ✓ 10 input layouts (page 1/16/32, prefix/decode edges) | ✓ decode page-boundary (uses `KDA_GRAPH_ATOL=1e-1` to absorb Triton recurrent-kernel CG-replay drift; eager `KDA_ATOL=3e-2` kept for non-graph cases) | ✓ ragged page-boundary extend | ✓ ragged page-boundary extend | ✓ EAGLE chain (topk=1) + EAGLE tree (topk=2) (`atol=1e-1` because the verify reference's pure-Python per-token recurrence drifts ~0.07 vs the Triton kernel even before CG capture/replay) | ✓ EAGLE chain CG + EAGLE tree CG (same `1e-1` tolerance) | — | blocked: HybridLinearAttnBackend `_replay_metadata` rejects modes outside `DECODE_OR_IDLE` / `TARGET_VERIFY` (`hybrid_linear_attn_backend.py:509,572`) | blocked: same `_replay_metadata` reject | deferred | blocked: same `_replay_metadata` reject | — |
| `flashinfer` prefill (SM100/SM103, FlashInfer 0.7.0) | ✓ KDA backend extend with beta logits, 128-D heads, and aligned/unaligned radix-cache snapshots; kernel tests cover BF16/FP32 and indexed state pools | — | — | — | — | — | — | — | — | — | — | — |
| FlashInfer batch-invariant class (`--enable-linear-attn-batch-invariant`; SM100/SM103/SM107) | ✓ `test_kda_flashinfer_batch_invariant.py`: backend extend and decode, 128-D heads, convolution included; every comparison is bitwise | ✓ kernel-level decode step captured and replayed with padded rows | blocked: the prefill plans on the host | ✓ eager at the layer (not captured) | blocked: speculative decoding refused | blocked | blocked | blocked | blocked | blocked | blocked | blocked |

## Input And Config Coverage

- 10 input variants from `make_kda_cases('triton')`: page 1, exact-page,
  crossing-page, ragged page-boundary, page-size-32 crossing, decode
  page-boundary, batch-size-1 decode.
- `num_k_heads=2, num_v_heads=2` with head dims defaulted by
  `DEFAULT_HEAD_K_DIM = DEFAULT_HEAD_V_DIM = 32`.
- `test_kda_prefill_flashinfer.py` exercises the 128-D FlashInfer path against
  Triton on B200/GB300, including the 130-token to 128-token cache boundary.

## FlashInfer Prefill Contract

Safe-gate KDA models default to FlashInfer prefill on SM100/SM103. Explicit
`--linear-attn-prefill-backend` choices take precedence. Other hardware, unbounded
gates, deterministic inference, TBO, and full prefill CUDA graphs retain Triton.

FlashInfer prefill execution requires SM100/SM103,
BF16 model activations, BF16 or FP32 SSM state, equal Q/K/V head counts with
128-D heads, and a finite negative safe-gate lower bound.
FlashInfer validates this tensor contract when the kernel runs, including
during server warmup. Radix-cache checkpoints require a positive mamba chunk
size divisible by 32. Incompatible intervals are rejected at backend
initialization; explicit FlashInfer selection rejects unbounded gates during
warmup. Original Kimi Linear retains Triton prefill by default.

The backend maps radix-cache checkpoint destinations once per tracked batch,
before layer execution. The adapter consumes raw gate/beta projections and
normalizes projection strides before calling `recurrent_kda(backend="cute-dsl")`
directly. FlashInfer owns execution scheduling and workspace allocation; its
direct API may read packed sequence offsets on the host to size that workspace.
Packed batches may include single-token sequences and zero-length padding rows;
only the total packed token count must exceed one.

KDA models pass raw beta logits in every phase; activation belongs to the
kernel adapters. A packed batch with only one token uses Triton with the same
raw beta format. Invalid checkpoint metadata raises an assertion instead of changing
kernels. TBO and full prefill CUDA graphs are rejected at initialization; BCG
keeps linear attention eager. Verify uses its separately resolved backend.

## Production-Unsupported

- **CUDA-graph capture/replay outside `DECODE_OR_IDLE` / `TARGET_VERIFY`** —
  KDA inherits the same `MambaAttnBackendBase` capture/replay path as GDN,
  so `ValueError("Invalid forward mode")` at
  `hybrid_linear_attn_backend.py:509, 572` rejects `DRAFT_EXTEND` /
  `DRAFT_EXTEND_V2` / `EXTEND` graph runners. Any Phase 4 KDA draft-extend
  graph runner is structurally unreachable.

## Next Work

- Add model-level Kimi K3 coverage when weights are available in the runner.

## FlashInfer Batch-Invariant Contract

`--enable-linear-attn-batch-invariant` replaces the decode and prefill kernels of
every KDA layer by one FlashInfer class (`kda.bf16.c16`): a token's output bits
depend on its own sequence only, and a decode step returns the bits a prefill of
the same tokens returns. `test_kda_flashinfer_batch_invariant.py` asserts this
with `torch.equal` for decode against teacher-forced prefill in changing
batches, chunked prefill, prefix-cache snapshots, slot reuse and CUDA-graph
replay, at the kernel adapter and through `KDAAttnBackend`.

A state slot holds the state at the request's last 16-token boundary and
FlashInfer keeps the open chunk beside the pool, so a slot may only be copied or
restored at a multiple of 16 tokens. The backend refuses the configurations that
break this (`kda_batch_invariant_refusal`), the scheduler keeps prefill cuts on
the checkpoint chunk when the prefix cache is on, and the adapter raises instead
of resuming or snapshotting off the grid. The class needs a safe gate, 128-D
heads, equal key and value head counts and a float32 state pool; whether it
covers the head count per rank is asked of the installed FlashInfer at startup.
