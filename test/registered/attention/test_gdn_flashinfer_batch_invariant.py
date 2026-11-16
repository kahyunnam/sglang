"""--enable-linear-attn-batch-invariant: the GDN layer path (conv, q / k / v
split, gating, FlashInfer's gdn.bf16.c64 kernel) gives a token the same bits
from prefill, prefill in pieces and decode, alone or in a mixed batch."""

import unittest

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    ExtendStateSnapshots,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="4-gpu-b200")

HEAD_DIM = 128
NUM_QK_HEADS = 4
NUM_V_HEADS = 8
CONV_KERNEL = 4
QKV_DIM = (2 * NUM_QK_HEADS + NUM_V_HEADS) * HEAD_DIM
NUM_LAYERS = 2
NUM_SLOTS = 12


def _class_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import flashinfer.batch_invariant.gdn  # noqa: F401
    except ImportError:
        return False
    capability = tuple(torch.cuda.get_device_capability())
    cuda_major = int((torch.version.cuda or "0").split(".", 1)[0])
    return capability in ((10, 0), (10, 3), (10, 7)) and cuda_major >= 13


class _Sequence:
    """One request's GDN layer inputs: pre-conv packed q / k / v rows and the
    raw gate inputs, as the layer's projections hand them to the backend."""

    def __init__(self, num_tokens: int, seed: int):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.x = torch.randn(
            (num_tokens, QKV_DIM), generator=gen, device="cuda"
        ).bfloat16()
        self.a = torch.randn(
            (num_tokens, NUM_V_HEADS), generator=gen, device="cuda"
        ).bfloat16()
        self.b = torch.randn(
            (num_tokens, NUM_V_HEADS), generator=gen, device="cuda"
        ).bfloat16()

    def __len__(self) -> int:
        return self.x.shape[0]


class _GDNLayer:
    """One GDN layer's state and the backend's kernel sequence around it
    (`GDNAttnBackend.forward_extend` / `forward_decode` without the scheduler)."""

    def __init__(self, kernel, layer: int = 0):
        gen = torch.Generator(device="cuda").manual_seed(7)
        self.kernel = kernel
        # The MambaPool layout: every layer's slots in one tensor, slot 0 padding.
        self.temporal = torch.zeros(
            (NUM_LAYERS, NUM_SLOTS, NUM_V_HEADS, HEAD_DIM, HEAD_DIM),
            dtype=torch.float32,
            device="cuda",
        )
        self.ssm_states = self.temporal[layer]
        self.conv_states = torch.zeros(
            (NUM_SLOTS, QKV_DIM, CONV_KERNEL - 1), dtype=torch.bfloat16, device="cuda"
        )
        self.conv_weights = (
            torch.randn((QKV_DIM, CONV_KERNEL), generator=gen, device="cuda") * 0.5
        ).bfloat16()
        self.A_log = torch.randn(NUM_V_HEADS, generator=gen, device="cuda")
        self.dt_bias = torch.randn(NUM_V_HEADS, generator=gen, device="cuda").bfloat16()

    def extend(self, pieces, slots, prefix_lens, snapshots=None) -> list[torch.Tensor]:
        """Prefill ``pieces`` (one `(x, a, b)` per sequence) in one call; returns
        each sequence's output rows."""
        from sglang.kernels.ops.attention.linear.fused_gdn_gating import (
            fused_gdn_gating,
        )
        from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
            fused_qkv_split_gdn_prefill,
        )
        from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn

        seq_lens = [x.shape[0] for x, _, _ in pieces]
        starts = torch.tensor(
            [sum(seq_lens[:i]) for i in range(len(seq_lens) + 1)],
            dtype=torch.int32,
            device="cuda",
        )
        cache_indices = torch.tensor(slots, dtype=torch.int32, device="cuda")
        prefix = torch.tensor(prefix_lens, dtype=torch.int32, device="cuda")
        x = torch.cat([x for x, _, _ in pieces])
        mixed_qkv = causal_conv1d_fn(
            x.transpose(0, 1),
            self.conv_weights,
            None,
            activation="silu",
            conv_states=self.conv_states,
            has_initial_state=prefix > 0,
            cache_indices=cache_indices,
            query_start_loc=starts,
            seq_lens_cpu=seq_lens,
        ).transpose(0, 1)[: x.shape[0]]
        q, k, v = fused_qkv_split_gdn_prefill(
            mixed_qkv,
            NUM_QK_HEADS,
            NUM_QK_HEADS,
            NUM_V_HEADS,
            HEAD_DIM,
            HEAD_DIM,
            HEAD_DIM,
        )
        g, beta = fused_gdn_gating(
            self.A_log,
            torch.cat([a for _, a, _ in pieces]),
            torch.cat([b for _, _, b in pieces]),
            self.dt_bias,
        )
        out, _, _ = self.kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=self.ssm_states,
            cache_indices=cache_indices,
            query_start_loc=starts,
            extend_prefix_lens=prefix,
            extend_seq_lens_cpu=seq_lens,
            extend_prefix_lens_cpu=list(prefix_lens),
            state_snapshots=snapshots,
        )
        return list(out[0].split(seq_lens))

    def decode(self, x, a, b, cache_indices) -> torch.Tensor:
        from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

        mixed_qkv = causal_conv1d_update(
            x,
            self.conv_states,
            self.conv_weights,
            None,
            "silu",
            conv_state_indices=cache_indices,
        )
        return self.kernel.packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=HEAD_DIM**-0.5,
            ssm_states=self.ssm_states,
            cache_indices=cache_indices,
            num_v_heads=NUM_V_HEADS,
            head_v_dim=HEAD_DIM,
        )[0]

    def decode_tokens(self, seq: _Sequence, slot: int, first: int, last: int):
        index = torch.tensor([slot], dtype=torch.int32, device="cuda")
        return torch.cat(
            [
                self.decode(seq.x[p : p + 1], seq.a[p : p + 1], seq.b[p : p + 1], index)
                for p in range(first, last)
            ]
        )

    def reference(self, seq: _Sequence, slot: int) -> torch.Tensor:
        """The whole sequence as one prefill, alone in the call."""
        return self.extend([(seq.x, seq.a, seq.b)], [slot], [0])[0]


@unittest.skipUnless(
    _class_available(),
    "needs SM100 / SM103 / SM107, CUDA 13 and FlashInfer's gdn.bf16.c64 class",
)
class TestGDNFlashInferBatchInvariant(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer_batch_invariant import (
            FlashInferBatchInvariantGDNKernel,
        )

        cls.kernel = FlashInferBatchInvariantGDNKernel()

    def test_decode_in_mixed_batches_equals_teacher_forced_prefill(self):
        """Ragged prefill, then decode steps in a padded bucket under a CUDA
        graph while requests finish: every row equals the request's own
        one-call prefill."""
        layer = _GDNLayer(self.kernel)
        seqs = [_Sequence(200, 1), _Sequence(150, 2), _Sequence(70, 3)]
        refs = [layer.reference(seq, slot) for seq, slot in zip(seqs, (9, 10, 11))]
        prompt_lens, slots, bucket = (37, 64, 5), (3, 1, 5), 4
        outs = layer.extend(
            [(s.x[:n], s.a[:n], s.b[:n]) for s, n in zip(seqs, prompt_lens)],
            slots,
            [0, 0, 0],
        )
        for out, ref, n in zip(outs, refs, prompt_lens):
            self.assertTrue(torch.equal(out, ref[:n]))
        self.assertGreater(float(refs[0].float().abs().sum()), 0.0)

        x = torch.zeros((bucket, QKV_DIM), dtype=torch.bfloat16, device="cuda")
        a = torch.zeros((bucket, NUM_V_HEADS), dtype=torch.bfloat16, device="cuda")
        b = torch.zeros_like(a)
        indices = torch.full((bucket,), -1, dtype=torch.int32, device="cuda")
        out = torch.empty(
            (bucket, NUM_V_HEADS, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
        )

        def load(step: int) -> list[tuple[int, int]]:
            live = []
            indices.fill_(-1)
            for i, (seq, n, slot) in enumerate(zip(seqs, prompt_lens, slots)):
                if n + step < len(seq):
                    x[i], a[i], b[i] = seq.x[n + step], seq.a[n + step], seq.b[n + step]
                    indices[i] = slot
                    live.append((i, n + step))
            return live

        def run():
            out.copy_(layer.decode(x, a, b, indices))

        # Captured right after the ragged prefill: the first step is a replay.
        load(0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for step in range(max(len(s) - n for s, n in zip(seqs, prompt_lens))):
            live = load(step)
            graph.replay()
            for i, pos in live:
                self.assertTrue(torch.equal(out[i], refs[i][pos]), (step, i))

    def test_prefill_pieces_and_cache_snapshots_resume_exactly(self):
        """A prefill cut on the 64-token grid, snapshots at the end of a piece
        and inside one, and a request resumed from a snapshot all reproduce the
        one-call prefill."""
        layer = _GDNLayer(self.kernel, layer=1)
        seq = _Sequence(300, 4)
        ref = layer.reference(seq, 11)
        destinations = torch.tensor([7], dtype=torch.int64, device="cuda")

        def piece(first, last):
            return (seq.x[first:last], seq.a[first:last], seq.b[first:last])

        first = layer.extend(
            [piece(0, 128)],
            [2],
            [0],
            ExtendStateSnapshots(rows=[0], boundaries=[128], destinations=destinations),
        )[0]
        self.assertTrue(torch.equal(first, ref[:128]))
        second = layer.extend(
            [piece(128, 300)],
            [2],
            [128],
            ExtendStateSnapshots(
                rows=[0], boundaries=[64], destinations=destinations + 1
            ),
        )[0]
        self.assertTrue(torch.equal(second, ref[128:]))

        # A cache hit at 192 tokens: the snapshot and the conv window before it.
        layer.ssm_states[4] = layer.ssm_states[8]
        layer.conv_states[4] = seq.x[189:192].transpose(0, 1)
        resumed = layer.extend([piece(192, 260)], [4], [192])[0]
        self.assertTrue(torch.equal(resumed, ref[192:260]))
        self.assertTrue(
            torch.equal(layer.decode_tokens(seq, 4, 260, 300), ref[260:300])
        )

    def test_snapshots_inside_and_at_the_end_of_one_call(self):
        """One extend whose sequences snapshot inside the call (leaving a
        one-token second piece), at its last boundary and not at all."""
        layer = _GDNLayer(self.kernel)
        seqs = [_Sequence(129, 11), _Sequence(200, 12), _Sequence(90, 13)]
        refs = [layer.reference(seq, slot) for seq, slot in zip(seqs, (9, 10, 11))]
        outs = layer.extend(
            [(s.x, s.a, s.b) for s in seqs],
            [1, 2, 3],
            [0, 0, 0],
            ExtendStateSnapshots(
                rows=[0, 1],
                boundaries=[64, 192],
                destinations=torch.tensor([5, 6, 0], device="cuda"),
            ),
        )
        for out, ref in zip(outs, refs):
            self.assertTrue(torch.equal(out, ref))
        # Each snapshot resumes its sequence: slot 5 at 64 tokens, slot 6 at 192.
        for seq, ref, slot, at in (
            (seqs[0], refs[0], 5, 64),
            (seqs[1], refs[1], 6, 192),
        ):
            layer.conv_states[slot] = seq.x[at - 3 : at].transpose(0, 1)
            n = len(seq)
            resumed = layer.extend(
                [(seq.x[at:n], seq.a[at:n], seq.b[at:n])], [slot], [at]
            )
            self.assertTrue(torch.equal(resumed[0], ref[at:]))

    def test_reused_slot_starts_from_its_own_request(self):
        """A slot freed with an open chunk (abort, retraction) carries nothing
        into the next request, including a one-token prompt."""
        layer = _GDNLayer(self.kernel)
        old, new = _Sequence(60, 5), _Sequence(90, 6)
        ref = layer.reference(new, 11)
        for prompt_len in (1, 20):
            with self.subTest(prompt_len=prompt_len):
                layer.extend([(old.x[:37], old.a[:37], old.b[:37])], [6], [0])
                layer.decode_tokens(old, 6, 37, 41)
                layer.ssm_states[6].zero_()
                layer.conv_states[6].zero_()
                n = prompt_len
                out = layer.extend([(new.x[:n], new.a[:n], new.b[:n])], [6], [0])[0]
                self.assertTrue(torch.equal(out, ref[:n]))
                self.assertTrue(
                    torch.equal(layer.decode_tokens(new, 6, n, 90), ref[n:])
                )

    def test_side_memory_is_per_layer_and_within_the_pool_charge(self):
        """FlashInfer sizes its side memory by the pool it is handed; the pool
        sizing charges `gdn_batch_invariant_side_bytes_per_slot` for it."""
        from flashinfer.batch_invariant.gdn.pool import get_pool

        from sglang.srt.layers.attention.linear.kernels.gdn_batch_invariant_spec import (
            gdn_batch_invariant_side_bytes_per_slot,
        )

        layer = _GDNLayer(self.kernel)
        seq = _Sequence(70, 8)
        layer.reference(seq, 3)
        side = get_pool(self.kernel._layer_pool(layer.ssm_states), NUM_QK_HEADS)
        self.assertEqual(side.slots, NUM_SLOTS)
        kept = sum(t.numel() * t.element_size() for t in side._side())
        kept += sum(t.numel() * t.element_size() for t in side._hist)
        charged = NUM_SLOTS * gdn_batch_invariant_side_bytes_per_slot(
            num_q_heads=NUM_QK_HEADS, num_v_heads=NUM_V_HEADS, num_layers=1
        )
        self.assertEqual(kept, charged)

    def test_refuses_what_the_class_cannot_hold(self):
        layer = _GDNLayer(self.kernel)
        seq = _Sequence(130, 9)
        with self.assertRaisesRegex(ValueError, "multiple of 64"):
            layer.extend(
                [(seq.x, seq.a, seq.b)],
                [2],
                [0],
                ExtendStateSnapshots(
                    rows=[0],
                    boundaries=[100],
                    destinations=torch.tensor([7], device="cuda"),
                ),
            )
        with self.assertRaisesRegex(ValueError, "from a multiple of 64"):
            layer.extend([(seq.x[:10], seq.a[:10], seq.b[:10])], [2], [37])
        with self.assertRaisesRegex(ValueError, "float32 state pool"):
            self.kernel._layer_pool(layer.ssm_states.bfloat16())


if __name__ == "__main__":
    unittest.main()
