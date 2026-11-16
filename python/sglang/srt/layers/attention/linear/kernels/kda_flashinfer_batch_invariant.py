"""FlashInfer batch-invariant KDA: one kernel class for prefill and decode.

Wraps ``flashinfer.recurrent_kda(..., batch_invariant=True)`` (class
``kda.bf16.c16``, SM100 / SM103 / SM107). A token's output bits depend on its
own sequence only, and a decode step returns the bits a prefill of the whole
sequence gives at that position.

State contract, which differs from every other KDA kernel:
  - ``ssm_states[slot]`` is fp32 and holds the state at the sequence's last
    16-token boundary, not the state after the last token;
  - FlashInfer keeps the rows of the open chunk next to the pool, keyed by the
    pool's storage, so a slot is only meaningful together with that side state;
  - the 16-token grid is anchored at token 0 of the request: an extend that
    does not continue the same slot must start at a multiple of 16 tokens.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)

# The class's chunk; boundaries are absolute multiples of it.
KDA_BATCH_INVARIANT_CHUNK = 16
KDA_BATCH_INVARIANT_HEAD_DIM = 128
KDA_BATCH_INVARIANT_ARCHS = ((10, 0), (10, 3), (10, 7))
_FLASHINFER_REQUIREMENT = (
    "a FlashInfer build with the batch-invariant KDA class kda.bf16.c16 "
    "(flashinfer.recurrent_kda(batch_invariant=True))"
)


# Bytes FlashInfer keeps per (slot, head) beside the pool: a bf16 copy of the
# boundary state plus the open chunk's cached rows, count and stamp (the
# buffers of flashinfer.batch_invariant.kda.pool.ExactPool).
_SIDE_STATE_BYTES_PER_HEAD = (
    KDA_BATCH_INVARIANT_HEAD_DIM
    * KDA_BATCH_INVARIANT_HEAD_DIM
    * 2  # bf16 boundary state
    + KDA_BATCH_INVARIANT_CHUNK
    * KDA_BATCH_INVARIANT_HEAD_DIM
    * (2 + 4 + 2 + 2)  # key, rhs, update rows
    + KDA_BATCH_INVARIANT_CHUNK * KDA_BATCH_INVARIANT_CHUNK * 2  # rows of L
    + KDA_BATCH_INVARIANT_HEAD_DIM * 4  # decay prefix
    + 4  # open-row count
    + 8 * 4  # stamp
)


def kda_batch_invariant_state_bytes_per_slot(*, num_layers: int, num_heads: int) -> int:
    return num_layers * num_heads * _SIDE_STATE_BYTES_PER_HEAD


class KDABatchInvariantSegment(msgspec.Struct, frozen=True):
    """Packed token rows ``[start, start + length)`` of batch row ``row``."""

    row: int
    start: int
    length: int
    # True when the rows start at a multiple of 16 tokens of the request.
    at_boundary: bool


def kda_batch_invariant_unsupported_reason(
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    lower_bound: Optional[float],
    state_dtype: torch.dtype,
) -> Optional[str]:
    """Why a KDA layer is outside the class, or None when it is covered."""
    if (
        head_k_dim != KDA_BATCH_INVARIANT_HEAD_DIM
        or head_v_dim != KDA_BATCH_INVARIANT_HEAD_DIM
    ):
        return (
            f"head dims K={head_k_dim}, V={head_v_dim}; the class needs "
            f"{KDA_BATCH_INVARIANT_HEAD_DIM}"
        )
    if num_k_heads != num_v_heads:
        return (
            f"{num_k_heads} key heads and {num_v_heads} value heads; the class "
            "needs equal head counts"
        )
    if lower_bound is None or not lower_bound < 0.0:
        return (
            f"gate lower bound {lower_bound!r}; the class needs the safe gate "
            "(a finite negative lower bound)"
        )
    if state_dtype != torch.float32:
        return f"a {state_dtype} recurrent state; the class needs float32"
    return None


def plan_kda_batch_invariant_extend(
    *,
    extend_lens: list[int],
    prefix_lens: list[int],
    cut_lens: list[int],
) -> tuple[list[KDABatchInvariantSegment], list[KDABatchInvariantSegment]]:
    """Split each sequence at its snapshot cut (0 for none) into two phases.

    The slot holds the state at ``prefix + cut`` between the phases, which is
    only that boundary's state when ``prefix + cut`` is a multiple of 16.
    """
    first: list[KDABatchInvariantSegment] = []
    second: list[KDABatchInvariantSegment] = []
    start = 0
    for row, (length, prefix, cut) in enumerate(
        zip(extend_lens, prefix_lens, cut_lens, strict=True)
    ):
        if not 0 <= cut <= length:
            raise ValueError(
                f"KDA snapshot cut {cut} is outside the {length} extended tokens"
            )
        if cut and (prefix + cut) % KDA_BATCH_INVARIANT_CHUNK:
            raise ValueError(
                "batch-invariant KDA can only snapshot a state at a multiple of "
                f"{KDA_BATCH_INVARIANT_CHUNK} tokens; got token {prefix + cut} "
                f"(prefix {prefix}, cut {cut})"
            )
        head = cut or length
        if head:
            first.append(
                KDABatchInvariantSegment(
                    row=row,
                    start=start,
                    length=head,
                    at_boundary=prefix % KDA_BATCH_INVARIANT_CHUNK == 0,
                )
            )
        if length - head:
            second.append(
                KDABatchInvariantSegment(
                    row=row, start=start + head, length=length - head, at_boundary=True
                )
            )
        start += length
    return first, second


class FlashInferKDABatchInvariantKernel(LinearAttnKernelBase):
    """KDA prefill, chunked prefill and decode on FlashInfer's batch-invariant class."""

    supports_track_state_snapshot = True
    supports_safe_gate = True
    supports_safe_gate_decode = True

    def __init__(self, state_chunk_size: int):
        if state_chunk_size <= 0 or state_chunk_size % KDA_BATCH_INVARIANT_CHUNK:
            raise ValueError(
                "batch-invariant KDA needs a state chunk size that is a multiple "
                f"of {KDA_BATCH_INVARIANT_CHUNK}; got {state_chunk_size}"
            )
        capability = torch.cuda.get_device_capability()
        if capability not in KDA_BATCH_INVARIANT_ARCHS:
            raise ValueError(
                "batch-invariant KDA runs on SM100, SM103 and SM107; this GPU is "
                f"SM{capability[0]}{capability[1]}"
            )
        try:
            from flashinfer import recurrent_kda
            from flashinfer.batch_invariant import fingerprint
            from flashinfer.batch_invariant.kda.api import (
                recurrent_kda_batch_invariant_extend,
                recurrent_kda_batch_invariant_reset,
            )

            self.fingerprint = fingerprint("recurrent_kda")
        except (ImportError, ValueError, KeyError) as exc:
            raise RuntimeError(
                f"--enable-linear-attn-batch-invariant needs {_FLASHINFER_REQUIREMENT}: "
                f"{exc}"
            ) from exc
        self._recurrent_kda = recurrent_kda
        self._append = recurrent_kda_batch_invariant_extend
        self._reset = recurrent_kda_batch_invariant_reset
        self._state_chunk_size = state_chunk_size
        self._checked_head_counts: set[int] = set()
        self._gate_cache: dict = {}
        # (pool address, slot) -> tokens the slot holds after its last extend;
        # the only state an extend that starts inside a chunk may continue.
        self._extend_end: dict[tuple[int, int], int] = {}

    def register_state_pool(self, ssm_states: torch.Tensor) -> None:
        """Create FlashInfer's side state for a pool; must run outside a graph capture."""
        if ssm_states.dtype != torch.float32 or not ssm_states.is_contiguous():
            raise ValueError(
                "batch-invariant KDA needs a contiguous float32 recurrent state pool; "
                f"got {ssm_states.dtype}, strides {tuple(ssm_states.stride())} "
                "(use --mamba-ssm-dtype float32 and a static, non-unified state pool)"
            )
        self._check_head_count(
            num_heads=ssm_states.shape[1], device=ssm_states.device
        )
        self._reset(
            ssm_states, torch.zeros(1, dtype=torch.int32, device=ssm_states.device)
        )

    def _check_head_count(self, *, num_heads: int, device: torch.device) -> None:
        """Raise unless the installed FlashInfer runs the class at this head count."""
        if num_heads in self._checked_head_counts:
            return
        from flashinfer.batch_invariant import BatchInvariantUnsupportedError

        dim = KDA_BATCH_INVARIANT_HEAD_DIM
        rows = torch.zeros(1, 1, num_heads, dim, dtype=torch.bfloat16, device=device)
        # The head counts the class covers are FlashInfer's to state;
        # one decode step on a scratch pool is the only public way to ask.
        try:
            self._recurrent_kda(
                q=rows,
                k=rows,
                v=rows,
                g=rows,
                beta=rows[..., 0].contiguous(),
                A_log=torch.zeros(num_heads, device=device),
                dt_bias=torch.zeros(num_heads * dim, device=device),
                initial_state=torch.zeros(1, num_heads, dim, dim, device=device),
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                lower_bound=-1.0,
                beta_is_logit=True,
                batch_invariant=True,
            )
        except BatchInvariantUnsupportedError as exc:
            raise ValueError(
                "--enable-linear-attn-batch-invariant: the installed FlashInfer "
                f"refused the class for this model ({num_heads} KDA heads per "
                f"rank): {exc}"
            ) from exc
        self._checked_head_counts.add(num_heads)

    def _gate_params(self, A_log: torch.Tensor, dt_bias: torch.Tensor):
        key = (id(A_log), id(dt_bias))
        cached = self._gate_cache.get(key)
        if cached is None:
            cached = (
                A_log.reshape(-1).float().contiguous(),
                dt_bias.reshape(-1).float().contiguous(),
            )
            self._gate_cache[key] = cached
        return cached

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
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = cache_indices.shape[0]
        num_heads, head_k_dim = q.shape[-2], q.shape[-1]
        num_v_heads, head_v_dim = v.shape[-2], v.shape[-1]
        A_log_fi, dt_bias_fi = self._gate_params(A_log, dt_bias)
        # q, k, v arrive as strided slices of one conv output; the class has equal
        # widths, so one stacked copy replaces three.
        qkv = torch.stack((q, k, v)).view(3, batch_size, 1, num_heads, head_k_dim)
        # One launch of the step kernel; it skips rows whose slot is negative.
        output, _ = self._recurrent_kda(
            q=qkv[0],
            k=qkv[1],
            v=qkv[2],
            g=a.reshape(batch_size, 1, num_v_heads, head_k_dim),
            beta=b.reshape(batch_size, 1, num_v_heads),
            A_log=A_log_fi,
            dt_bias=dt_bias_fi,
            initial_state=ssm_states,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            lower_bound=lower_bound,
            ssm_state_indices=cache_indices.to(torch.int32),
            beta_is_logit=True,
            batch_invariant=True,
        )
        return output.view(1, batch_size, num_v_heads, head_v_dim)

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
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        lower_bound: float,
        extend_seq_lens_cpu: list[int],
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: Optional[list[int]] = None,
        return_intermediate_states: bool = False,
        track_state: Optional[torch.Tensor] = None,
        track_chunk_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]] | torch.Tensor:
        num_tokens = q.shape[1]
        rows = (
            q.reshape(num_tokens, *q.shape[2:]),
            k.reshape(num_tokens, *k.shape[2:]),
            v.reshape(num_tokens, *v.shape[2:]),
            g[:, :num_tokens].reshape(num_tokens, v.shape[2], q.shape[-1]),
            beta[:, :num_tokens].reshape(num_tokens, v.shape[2]),
        )
        gates = dict(zip(("A_log", "dt_bias"), self._gate_params(A_log, dt_bias)))
        gates["lower_bound"] = lower_bound
        extend_lens = list(extend_seq_lens_cpu)
        prefix_lens = (
            extend_prefix_lens.tolist()
            if extend_prefix_lens_cpu is None
            else list(extend_prefix_lens_cpu)
        )
        if return_intermediate_states and any(
            p % KDA_BATCH_INVARIANT_CHUNK for p in prefix_lens
        ):
            # The caller snapshots slots after this call, and a slot only holds
            # its sequence's state at a multiple of 16 tokens.
            raise RuntimeError(
                "batch-invariant KDA cannot checkpoint a prefill that starts at "
                f"tokens {prefix_lens}: every start must be a multiple of "
                f"{KDA_BATCH_INVARIANT_CHUNK}"
            )
        # -1 marks a row without a snapshot; 0 snapshots the state it starts from.
        track_chunks = (
            [-1] * len(extend_lens) if track_state is None else track_chunk_idx.tolist()
        )
        first, second = plan_kda_batch_invariant_extend(
            extend_lens=extend_lens,
            prefix_lens=prefix_lens,
            cut_lens=[max(c, 0) * self._state_chunk_size for c in track_chunks],
        )
        slots = cache_indices.to(torch.int32)
        slot_list = slots.tolist()
        self._check_continuations(
            pool=ssm_states.data_ptr(),
            segments=first,
            slot_list=slot_list,
            prefix_lens=prefix_lens,
        )

        run = dict(rows=rows, ssm_states=ssm_states, slots=slots, gates=gates)
        self._snapshot(track_state, ssm_states, slots, track_chunks, at_start=True)
        output = self._run_phase(segments=first, output=None, **run)
        self._snapshot(track_state, ssm_states, slots, track_chunks, at_start=False)
        output = self._run_phase(segments=second, output=output, **run)

        pool = ssm_states.data_ptr()
        for slot, prefix, length in zip(slot_list, prefix_lens, extend_lens):
            self._extend_end[(pool, slot)] = prefix + length
        if output is None:
            output = torch.empty_like(rows[2])
        output = output.unsqueeze(0)
        return (output, None) if return_intermediate_states else output

    @staticmethod
    def _snapshot(
        track_state: Optional[torch.Tensor],
        ssm_states: torch.Tensor,
        slots: torch.Tensor,
        track_chunks: list[int],
        at_start: bool,
    ) -> None:
        tracked = [
            row for row, c in enumerate(track_chunks) if (c == 0 if at_start else c > 0)
        ]
        if tracked:
            index = torch.tensor(tracked, dtype=torch.int64, device=slots.device)
            track_state.index_copy_(
                0, index, ssm_states.index_select(0, slots.long()[index])
            )

    def _check_continuations(
        self,
        *,
        pool: int,
        segments: list[KDABatchInvariantSegment],
        slot_list: list[int],
        prefix_lens: list[int],
    ) -> None:
        for segment in segments:
            prefix = prefix_lens[segment.row]
            if segment.at_boundary:
                continue
            if self._extend_end.get((pool, slot_list[segment.row])) != prefix:
                raise RuntimeError(
                    "batch-invariant KDA cannot resume a request at token "
                    f"{prefix}: inside a 16-token chunk a slot can only continue "
                    "its own previous prefill, and this state was restored, "
                    "copied or advanced by decode steps since"
                )

    def _run_phase(
        self,
        *,
        segments: list[KDABatchInvariantSegment],
        rows: tuple[torch.Tensor, ...],
        output: Optional[torch.Tensor],
        ssm_states: torch.Tensor,
        slots: torch.Tensor,
        gates: dict,
    ) -> Optional[torch.Tensor]:
        """Run the segments and return the packed output rows; ``output`` is None
        until a launch has produced rows."""
        run = dict(rows=rows, ssm_states=ssm_states, slots=slots, gates=gates)
        boundary = [s for s in segments if s.at_boundary]
        if boundary:
            output = self._prefill_from_boundary(
                segments=boundary, output=output, **run
            )
        # A row that continues an open chunk appends to it; equal lengths share a launch.
        by_length: dict[int, list[KDABatchInvariantSegment]] = {}
        for segment in segments:
            if not segment.at_boundary:
                by_length.setdefault(segment.length, []).append(segment)
        for group in by_length.values():
            output = self._append_to_open_chunk(segments=group, output=output, **run)
        return output

    def _prefill_from_boundary(
        self,
        *,
        segments: list[KDABatchInvariantSegment],
        rows: tuple[torch.Tensor, ...],
        output: Optional[torch.Tensor],
        ssm_states: torch.Tensor,
        slots: torch.Tensor,
        gates: dict,
    ) -> torch.Tensor:
        segment_slots = _segment_slots(segments=segments, slots=slots)
        if all(s.length == 1 for s in segments):
            # recurrent_kda reads one token per sequence as a decode step, which
            # would continue the slot's open chunk; start the chunk explicitly.
            self._reset(ssm_states, segment_slots)
            return self._append_to_open_chunk(
                segments=segments,
                rows=rows,
                output=output,
                ssm_states=ssm_states,
                slots=slots,
                gates=gates,
            )
        token_index = _segment_token_index(
            segments=segments, num_tokens=rows[0].shape[0], device=slots.device
        )
        packed = rows if token_index is None else _gather(rows, token_index)
        cu_seqlens = [0]
        for segment in segments:
            cu_seqlens.append(cu_seqlens[-1] + segment.length)
        out, _ = self._recurrent_kda(
            q=packed[0].unsqueeze(0),
            k=packed[1].unsqueeze(0),
            v=packed[2].unsqueeze(0),
            g=packed[3].unsqueeze(0),
            beta=packed[4].unsqueeze(0),
            initial_state=ssm_states,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int64, device=slots.device),
            ssm_state_indices=segment_slots,
            beta_is_logit=True,
            batch_invariant=True,
            **gates,
        )
        if token_index is None:
            # The call covered every packed row in order: its rows are the output.
            return out[0]
        if output is None:
            output = torch.empty_like(rows[2])
        return output.index_copy_(0, token_index, out[0])

    def _append_to_open_chunk(
        self,
        *,
        segments: list[KDABatchInvariantSegment],
        rows: tuple[torch.Tensor, ...],
        output: Optional[torch.Tensor],
        ssm_states: torch.Tensor,
        slots: torch.Tensor,
        gates: dict,
    ) -> torch.Tensor:
        length = segments[0].length
        # num_tokens=-1 never matches: the [N, T] call always needs its own rows.
        token_index = _segment_token_index(
            segments=segments, num_tokens=-1, device=slots.device
        )
        packed = [
            t.unflatten(0, (len(segments), length)) for t in _gather(rows, token_index)
        ]
        out = self._append(
            *packed,
            initial_state=ssm_states,
            ssm_state_indices=_segment_slots(segments=segments, slots=slots),
            **gates,
        )
        if output is None:
            output = torch.empty_like(rows[2])
        return output.index_copy_(0, token_index, out.flatten(0, 1))


def _segment_slots(
    *, segments: list[KDABatchInvariantSegment], slots: torch.Tensor
) -> torch.Tensor:
    if [s.row for s in segments] == list(range(slots.shape[0])):
        return slots
    index = torch.tensor([s.row for s in segments], device=slots.device)
    return slots.index_select(0, index)


def _segment_token_index(
    *, segments: list[KDABatchInvariantSegment], num_tokens: int, device: torch.device
) -> Optional[torch.Tensor]:
    # None when the segments are all `num_tokens` packed rows in order.
    end = 0
    for segment in segments:
        if segment.start != end:
            break
        end = segment.start + segment.length
    else:
        if end == num_tokens:
            return None
    return torch.cat(
        [torch.arange(s.start, s.start + s.length, device=device) for s in segments]
    )


def _gather(
    rows: tuple[torch.Tensor, ...], token_index: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    return tuple(t.index_select(0, token_index) for t in rows)
