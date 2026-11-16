"""Constants and host-side arithmetic of FlashInfer's batch-invariant GDN class
(``gdn.bf16.c64``); no kernel imports, so sizing and scheduling code can use it."""

from typing import Sequence

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    ExtendStateSnapshots,
)

# Fixed by the FlashInfer class: chunk boundaries are absolute multiples of 64
# tokens and every head is 128 wide.
GDN_BATCH_INVARIANT_CHUNK = 64
GDN_BATCH_INVARIANT_HEAD_DIM = 128
# SM100 (B200, GB200), SM103 (B300, GB300), SM107 (Rubin).
GDN_BATCH_INVARIANT_ARCHS = ((10, 0), (10, 3), (10, 7))


def gdn_batch_invariant_side_bytes_per_slot(
    *, num_q_heads: int, num_v_heads: int, num_layers: int
) -> int:
    """Bytes FlashInfer keeps next to one state-pool slot, over all layers:
    the open chunk's cached rows, a bf16 copy of the boundary state and the raw
    rows of a prefill's open tail."""
    chunk, dim = GDN_BATCH_INVARIANT_CHUNK, GDN_BATCH_INVARIANT_HEAD_DIM
    bf16, fp32 = 2, 4
    cached_rows = (
        num_v_heads * dim * dim * bf16  # boundary state copy
        + chunk * num_v_heads * dim * bf16  # normalized keys
        + 2 * num_v_heads * chunk * chunk * bf16  # blocked-inverse rows
        + 2 * num_v_heads * chunk * dim * bf16  # VKS and NV rows
        + 2 * num_v_heads * chunk * fp32  # gates and betas
        + num_v_heads * fp32  # open-row count
    )
    tail_rows = chunk * (
        num_q_heads * dim * bf16 + num_v_heads * dim * bf16 + 2 * num_v_heads * fp32
    )
    return (cached_rows + tail_rows) * num_layers


def build_extend_state_snapshots(
    *,
    track_mask: Sequence[bool],
    track_seqlens: Sequence[int],
    prefix_lens: Sequence[int],
    destinations: torch.Tensor,
) -> ExtendStateSnapshots:
    """The boundaries `_init_track_ssm_indices` reads from the final state or
    from the per-chunk states: the last whole chunk at or before the tracked
    length past the prefix (a tracked length one past a boundary names it)."""
    chunk = GDN_BATCH_INVARIANT_CHUNK
    rows = [row for row, tracked in enumerate(track_mask) if tracked]
    return ExtendStateSnapshots(
        rows=rows,
        boundaries=[
            (track_seqlens[row] - prefix_lens[row]) // chunk * chunk for row in rows
        ],
        destinations=destinations,
    )
