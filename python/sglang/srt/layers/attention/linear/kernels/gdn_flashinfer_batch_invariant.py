"""FlashInfer's batch-invariant GDN class (``gdn.bf16.c64``) as an SGLang kernel.

Prefill, prefill in pieces and decode return the same bits for the same token,
whatever shares the call. A state-pool slot holds the fp32 state at the
sequence's last 64-token boundary; FlashInfer keeps the open chunk's rows next
to the pool, keyed by the pool tensor it is given.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import torch

from sglang.kernels.ops.attention.linear.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.linear.kernels.gdn_batch_invariant_spec import (
    GDN_BATCH_INVARIANT_ARCHS,
    GDN_BATCH_INVARIANT_CHUNK,
    GDN_BATCH_INVARIANT_HEAD_DIM,
)
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    ExtendStateSnapshots,
    LinearAttnKernelBase,
)

logger = logging.getLogger(__name__)


def _load_flashinfer_gdn_batch_invariant():
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    try:
        from flashinfer.batch_invariant.gdn.pool import get_pool
        from flashinfer.gdn_prefill import chunk_gated_delta_rule
    except ImportError as e:
        raise RuntimeError(
            "--enable-linear-attn-batch-invariant needs a FlashInfer build with "
            "the batch-invariant GDN class gdn.bf16.c64 "
            "(flashinfer.batch_invariant.gdn, flashinfer-ai/flashinfer#9405); "
            f"the installed FlashInfer lacks it: {e}"
        ) from e
    return chunk_gated_delta_rule, get_pool


def _check_device() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("--enable-linear-attn-batch-invariant requires CUDA.")
    capability = tuple(torch.cuda.get_device_capability())
    cuda_major = int((torch.version.cuda or "0").split(".", 1)[0])
    if capability not in GDN_BATCH_INVARIANT_ARCHS or cuda_major < 13:
        raise RuntimeError(
            "--enable-linear-attn-batch-invariant supports SM100, SM103 and "
            "SM107 GPUs with a CUDA 13 build of torch; got compute capability "
            f"{capability[0]}.{capability[1]} and CUDA {torch.version.cuda}."
        )


class FlashInferBatchInvariantGDNKernel(LinearAttnKernelBase):
    """GDN decode and extend through ``chunk_gated_delta_rule(batch_invariant=True)``."""

    supports_packed_decode = True
    # extend() takes the prefix-cache snapshots itself (`state_snapshots`).
    snapshots_extend_state = True

    def __init__(self):
        _check_device()
        self._prefill_fn, self._get_pool = _load_flashinfer_gdn_batch_invariant()
        self._layer_pools: dict[int, torch.Tensor] = {}
        self._decode_offsets: dict[tuple[int, torch.device], torch.Tensor] = {}
        logger.info("Using FlashInfer batch-invariant GDN kernels (gdn.bf16.c64)")

    def _layer_pool(self, ssm_states: torch.Tensor) -> torch.Tensor:
        # FlashInfer sizes its side memory by the storage behind the pool it is
        # given; a DLPack alias gives one layer's slots a storage of their own.
        pool = self._layer_pools.get(ssm_states.data_ptr())
        if pool is None or pool.shape != ssm_states.shape:
            if ssm_states.dtype != torch.float32 or not ssm_states.is_contiguous():
                raise ValueError(
                    "Batch-invariant GDN needs a contiguous float32 state pool "
                    "(--mamba-ssm-dtype float32, no unified memory); got "
                    f"{ssm_states.dtype}, contiguous={ssm_states.is_contiguous()}."
                )
            pool = torch.from_dlpack(ssm_states)
            self._layer_pools[ssm_states.data_ptr()] = pool
        return pool

    def _decode_cu_seqlens(
        self, *, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        key = (batch_size, device)
        offsets = self._decode_offsets.get(key)
        if offsets is None:
            offsets = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
            self._decode_offsets[key] = offsets
        return offsets

    def _side_pool(self, *, pool: torch.Tensor, num_qk_heads: int):
        return self._get_pool(pool, num_qk_heads)

    def _step(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = cache_indices.shape[0]
        g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
        # A padded graph row carries slot -1: FlashInfer skips an out-of-pool slot
        # (NaN row, nothing written); other kernels leave such rows uninitialized.
        out = self._prefill_fn(
            q=q,
            k=k,
            v=v,
            g=g[0],
            beta=beta[0],
            initial_state=self._layer_pool(ssm_states),
            state_indices=cache_indices,
            cu_seqlens=self._decode_cu_seqlens(batch_size=batch_size, device=q.device),
            use_qk_l2norm_in_kernel=True,
            gate_domain="log",
            batch_invariant=True,
        )
        return out.view(1, batch_size, v.shape[1], v.shape[2])

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> torch.Tensor:
        from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
            fused_qkv_split_gdn_prefill,
        )

        head_dim = GDN_BATCH_INVARIANT_HEAD_DIM
        num_qk_heads = (mixed_qkv.shape[-1] - num_v_heads * head_v_dim) // (
            2 * head_dim
        )
        # The split prefill uses, so a token's q / k / v bits match across paths.
        q, k, v = fused_qkv_split_gdn_prefill(
            mixed_qkv,
            num_qk_heads,
            num_qk_heads,
            num_v_heads,
            head_dim,
            head_dim,
            head_v_dim,
        )
        return self._step(
            q=q[0],
            k=k[0],
            v=v[0],
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self._step(
            q=q[0].contiguous(),
            k=k[0].contiguous(),
            v=v[0].contiguous(),
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
        )

    def _prefill(
        self,
        *,
        rows: tuple[torch.Tensor, ...],
        pool: torch.Tensor,
        slots: torch.Tensor,
        seq_lens: Sequence[int],
        prefix_lens: Sequence[int],
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        offsets = [0]
        for length in seq_lens:
            offsets.append(offsets[-1] + length)
        chunk = GDN_BATCH_INVARIANT_CHUNK
        if any(prefix % chunk for prefix in prefix_lens):
            raise ValueError(
                "Batch-invariant GDN extends a request only from a multiple of "
                f"{chunk} tokens; got prefix lengths {list(prefix_lens)}."
            )
        # FlashInfer reads one token per sequence without prefix_lens as a decode
        # step of whatever the slot holds; with it, as a start from the pool state.
        prefix = None
        if offsets[-1] == len(seq_lens):
            prefix = torch.tensor(prefix_lens, dtype=torch.int32, device=pool.device)
        q, k, v, g, beta = rows
        return self._prefill_fn(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=pool,
            state_indices=slots,
            cu_seqlens=torch.tensor(offsets, dtype=torch.int32, device=pool.device),
            cu_seqlens_cpu=offsets,
            prefix_lens=prefix,
            output=output,
            use_qk_l2norm_in_kernel=True,
            gate_domain="log",
            batch_invariant=True,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        extend_prefix_lens: Optional[torch.Tensor] = None,
        extend_seq_lens_cpu: Optional[Sequence[int]] = None,
        extend_prefix_lens_cpu: Optional[Sequence[int]] = None,
        state_snapshots: Optional[ExtendStateSnapshots] = None,
        **kwargs,
    ) -> tuple:
        rows = (
            q[0].contiguous(),
            k[0].contiguous(),
            v[0].contiguous(),
            g[0].to(torch.float32).contiguous(),
            beta[0].to(torch.float32).contiguous(),
        )
        pool = self._layer_pool(ssm_states)
        slots = cache_indices.to(torch.int32)
        if extend_seq_lens_cpu is None or sum(extend_seq_lens_cpu) != rows[0].shape[0]:
            offsets = query_start_loc.tolist()
            extend_seq_lens_cpu = [b - a for a, b in zip(offsets, offsets[1:])]
        if extend_prefix_lens_cpu is None:
            extend_prefix_lens_cpu = (
                [0] * len(extend_seq_lens_cpu)
                if extend_prefix_lens is None
                else extend_prefix_lens.tolist()
            )
        mid_call = _mid_call_snapshots(
            state_snapshots=state_snapshots,
            seq_lens=extend_seq_lens_cpu,
            prefix_lens=extend_prefix_lens_cpu,
        )
        output = _packed_output(output=kwargs.get("output"), v=rows[2])
        if not mid_call:
            out = self._prefill(
                rows=rows,
                pool=pool,
                slots=slots,
                seq_lens=extend_seq_lens_cpu,
                prefix_lens=extend_prefix_lens_cpu,
                output=output,
            )
            _copy_snapshots(pool=pool, slots=slots, state_snapshots=state_snapshots)
        else:
            out = self._extend_around_snapshots(
                rows=rows,
                pool=pool,
                slots=slots,
                seq_lens=extend_seq_lens_cpu,
                prefix_lens=extend_prefix_lens_cpu,
                state_snapshots=state_snapshots,
                mid_call=mid_call,
                output=output,
            )
        # A captured decode step cannot build the cached rows of an open tail.
        self._side_pool(pool=pool, num_qk_heads=rows[0].shape[1]).rebuild(pool)
        return out.view(1, *out.shape), None, None

    def _extend_around_snapshots(
        self,
        *,
        rows: tuple[torch.Tensor, ...],
        pool: torch.Tensor,
        slots: torch.Tensor,
        seq_lens: Sequence[int],
        prefix_lens: Sequence[int],
        state_snapshots: ExtendStateSnapshots,
        mid_call: dict[int, int],
        output: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Two pieces cut where a snapshot lies inside its sequence: the pool only
        # ever holds a sequence's last boundary state.
        out = output if output is not None else torch.empty_like(rows[2])
        starts = [0]
        for length in seq_lens:
            starts.append(starts[-1] + length)
        first = [
            (i, starts[i], mid_call.get(i, seq_lens[i]), prefix_lens[i])
            for i in range(len(seq_lens))
            if mid_call.get(i, seq_lens[i]) > 0
        ]
        second = [
            (i, starts[i] + cut, seq_lens[i] - cut, prefix_lens[i] + cut)
            for i, cut in mid_call.items()
        ]
        self._prefill_pieces(rows=rows, pool=pool, slots=slots, pieces=first, out=out)
        _copy_snapshots(pool=pool, slots=slots, state_snapshots=state_snapshots)
        self._prefill_pieces(rows=rows, pool=pool, slots=slots, pieces=second, out=out)
        return out

    def _prefill_pieces(
        self,
        *,
        rows: tuple[torch.Tensor, ...],
        pool: torch.Tensor,
        slots: torch.Tensor,
        pieces: list[tuple[int, int, int, int]],
        out: torch.Tensor,
    ) -> None:
        # pieces: (sequence, first packed row, rows, prefix).
        if not pieces:
            return
        device = pool.device
        token_rows = torch.cat(
            [torch.arange(start, start + n, device=device) for _, start, n, _ in pieces]
        )
        seqs = torch.tensor([i for i, _, _, _ in pieces], device=device)
        piece_out = self._prefill(
            rows=tuple(x.index_select(0, token_rows) for x in rows),
            pool=pool,
            slots=slots.index_select(0, seqs),
            seq_lens=[n for _, _, n, _ in pieces],
            prefix_lens=[prefix for _, _, _, prefix in pieces],
        )
        out.index_copy_(0, token_rows, piece_out)

    def prewarm(
        self,
        *,
        num_qk_heads: int,
        num_v_heads: int,
        io_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Compile the prefill and step kernels on a scratch pool, outside graph
        capture, so no serving batch waits on FlashInfer's first-use build."""
        assert not torch.cuda.is_current_stream_capturing()
        chunk, dim = GDN_BATCH_INVARIANT_CHUNK, GDN_BATCH_INVARIANT_HEAD_DIM
        pool = torch.zeros(
            (3, num_v_heads, dim, dim), dtype=torch.float32, device=device
        )
        slots = torch.tensor([1, 2], dtype=torch.int32, device=device)

        def rows(num_tokens: int) -> tuple[torch.Tensor, ...]:
            q = torch.zeros(
                (num_tokens, num_qk_heads, dim), dtype=io_dtype, device=device
            )
            v = torch.zeros(
                (num_tokens, num_v_heads, dim), dtype=io_dtype, device=device
            )
            g = torch.zeros(
                (num_tokens, num_v_heads), dtype=torch.float32, device=device
            )
            return q, torch.zeros_like(q), v, g, torch.zeros_like(g)

        # Whole chunks, then open tails (with their rebuild), then a decode step.
        for seq_len in (chunk, chunk + 3):
            self._prefill(
                rows=rows(2 * seq_len),
                pool=pool,
                slots=slots,
                seq_lens=[seq_len, seq_len],
                prefix_lens=[0, 0],
            )
        self._side_pool(pool=pool, num_qk_heads=num_qk_heads).rebuild(pool)
        q, k, v, g, beta = rows(2)
        self._prefill_fn(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=pool,
            state_indices=slots,
            cu_seqlens=self._decode_cu_seqlens(batch_size=2, device=device),
            use_qk_l2norm_in_kernel=True,
            gate_domain="log",
            batch_invariant=True,
        )
        torch.cuda.current_stream(device).synchronize()


def _packed_output(
    *, output: Optional[torch.Tensor], v: torch.Tensor
) -> Optional[torch.Tensor]:
    if output is None:
        return None
    if tuple(output.shape) != (1, *v.shape) or output.dtype != v.dtype:
        raise ValueError(
            "Batch-invariant GDN prefill output must match v: expected "
            f"{(1, *v.shape)} {v.dtype}, got {tuple(output.shape)} {output.dtype}."
        )
    return output[0]


def _mid_call_snapshots(
    *,
    state_snapshots: Optional[ExtendStateSnapshots],
    seq_lens: Sequence[int],
    prefix_lens: Sequence[int],
) -> dict[int, int]:
    # Sequence row -> boundary, for snapshots before the row's last boundary.
    if state_snapshots is None:
        return {}
    chunk = GDN_BATCH_INVARIANT_CHUNK
    mid_call = {}
    for row, boundary in zip(state_snapshots.rows, state_snapshots.boundaries):
        if (prefix_lens[row] + boundary) % chunk:
            raise ValueError(
                "Batch-invariant GDN can snapshot a state only at a multiple of "
                f"{chunk} tokens; got prefix {prefix_lens[row]} + {boundary}. "
                "Prefill chunks and prefix-cache checkpoints must stay on that grid."
            )
        if boundary < seq_lens[row] // chunk * chunk:
            mid_call[row] = boundary
    return mid_call


def _copy_snapshots(
    *,
    pool: torch.Tensor,
    slots: torch.Tensor,
    state_snapshots: Optional[ExtendStateSnapshots],
) -> None:
    if state_snapshots is None or not state_snapshots.rows:
        return
    rows = torch.tensor(state_snapshots.rows, device=pool.device)
    pool.index_copy_(
        0,
        state_snapshots.destinations.index_select(0, rows).to(torch.int64),
        pool.index_select(0, slots.index_select(0, rows).to(torch.int64)),
    )
