"""Batch-invariant FlashInfer KDA: decode, prefill and chunked prefill give one set of bits."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
    (10, 0),
    (10, 3),
    (10, 7),
):
    pytest.skip(
        "batch-invariant KDA requires SM100, SM103 or SM107", allow_module_level=True
    )

pytest.importorskip("flashinfer")
pytest.importorskip(
    "flashinfer.batch_invariant.kda.api",
    reason="FlashInfer build lacks the batch-invariant KDA class",
)

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (  # noqa: E402
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.kda_backend import (  # noqa: E402
    KDAAttnBackend,
)
from sglang.srt.layers.attention.linear.kernels.kda_flashinfer_batch_invariant import (  # noqa: E402
    FlashInferKDABatchInvariantKernel,
)
from sglang.srt.model_executor.forward_context import (  # noqa: E402
    ForwardContext,
    forward_context,
)
from sglang.srt.model_executor.forward_mode import ForwardMode  # noqa: E402
from sglang.srt.runtime_context import (  # noqa: E402
    get_exec,
    get_memory,
    get_parallel,
)
from sglang.test.kits.attention_unittest.attention_methods.kda_attention import (  # noqa: E402
    KDAAttentionCase,
    _make_forward_batch,
    build_kda_attention_fixture,
)
from sglang.test.test_utils import CustomTestCase  # noqa: E402

HEADS, DIM, SLOTS, STATE_CHUNK = 4, 128, 12, 64
DEVICE = "cuda"


@pytest.fixture(scope="module")
def gates():
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    return dict(
        A_log=torch.randn(HEADS, device=DEVICE, generator=generator) * 0.2,
        dt_bias=torch.randn(HEADS * DIM, device=DEVICE, generator=generator) * 0.1,
        lower_bound=-5.0,
    )


def _sequence(seed: int, length: int, heads: int = HEADS):
    """q, k, v, raw gate, beta logit rows of one request; gates near 0 keep a long memory."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device=DEVICE, generator=generator)

    return (
        randn(length, heads, DIM).bfloat16(),
        randn(length, heads, DIM).bfloat16(),
        randn(length, heads, DIM).bfloat16(),
        (randn(length, heads, DIM) - 4.0).bfloat16(),
        randn(length, heads).bfloat16(),
    )


def _new_pool(
    kernel: FlashInferKDABatchInvariantKernel, heads: int = HEADS
) -> torch.Tensor:
    pool = torch.zeros(SLOTS, heads, DIM, DIM, device=DEVICE)
    kernel.register_state_pool(pool)
    return pool


def _extend(kernel, pool, gates, pieces, slots, prefix_lens, **track):
    """Extend ``slots`` with one piece of rows each; returns the per-request outputs."""
    lens = [piece[0].shape[0] for piece in pieces]
    packed = [torch.cat([piece[i] for piece in pieces]).unsqueeze(0) for i in range(5)]
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    result = kernel.extend(
        *packed,
        ssm_states=pool,
        cache_indices=torch.tensor(slots, dtype=torch.int32, device=DEVICE),
        query_start_loc=torch.tensor(cu, dtype=torch.int32, device=DEVICE),
        extend_seq_lens_cpu=lens,
        extend_prefix_lens=torch.tensor(prefix_lens, dtype=torch.int32, device=DEVICE),
        extend_prefix_lens_cpu=list(prefix_lens),
        **gates,
        **track,
    )
    output = result[0] if isinstance(result, tuple) else result
    return list(output[0].split(lens))


def _decode(kernel, pool, gates, tokens, slots):
    """One decode step; ``tokens[i]`` is a 5-tuple of single rows for ``slots[i]``."""
    q, k, v, a, b = (torch.stack([token[i] for token in tokens]) for i in range(5))
    batch = len(slots)
    heads = q.shape[-2]
    return kernel.decode(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        a.reshape(batch, heads * DIM),
        b.unsqueeze(0),
        ssm_states=pool,
        cache_indices=torch.tensor(slots, dtype=torch.int32, device=DEVICE),
        query_start_loc=torch.arange(batch + 1, dtype=torch.int32, device=DEVICE),
        **gates,
    )[0]


def _rows(sequence, start, stop):
    return tuple(t[start:stop] for t in sequence)


def _teacher_forced(kernel, gates, sequence):
    """Prefill of the whole request alone in a fresh pool."""
    pool = _new_pool(kernel)
    return _extend(kernel, pool, gates, [sequence], [1], [0])[0]


@pytest.fixture(scope="module")
def kernel():
    return FlashInferKDABatchInvariantKernel(state_chunk_size=STATE_CHUNK)


def test_decode_equals_teacher_forced_prefill_in_mixed_batches(kernel, gates):
    """A request's decode rows are its prefill rows, alone or beside changing neighbours."""
    # (prompt tokens, total tokens, first decode step in the shared schedule)
    plan = {
        1: (1, 40, 0),
        2: (15, 50, 0),
        3: (16, 49, 3),
        4: (37, 70, 5),
        5: (64, 81, 9),
    }
    sequences = {
        slot: _sequence(100 + slot, total) for slot, (_, total, _) in plan.items()
    }
    reference = {slot: _teacher_forced(kernel, gates, sequences[slot]) for slot in plan}

    pool = _new_pool(kernel)
    got = {slot: [] for slot in plan}
    position = {}
    step = 0
    while len(position) < len(plan) or any(position[s] < plan[s][1] for s in position):
        arriving = [s for s in plan if s not in position and plan[s][2] <= step]
        if arriving:
            outputs = _extend(
                kernel,
                pool,
                gates,
                [_rows(sequences[s], 0, plan[s][0]) for s in arriving],
                arriving,
                [0] * len(arriving),
            )
            for slot, output in zip(arriving, outputs):
                got[slot].append(output)
                position[slot] = plan[slot][0]
        running = [s for s in position if position[s] < plan[s][1]]
        if running:
            # Rotate the batch order so a request's batch row changes every step.
            running = running[step % len(running) :] + running[: step % len(running)]
            output = _decode(
                kernel,
                pool,
                gates,
                [_rows(sequences[s], position[s], position[s] + 1) for s in running],
                running,
            )
            for row, slot in enumerate(running):
                got[slot].append(output[row : row + 1].reshape(1, HEADS, DIM))
                position[slot] += 1
        step += 1

    for slot in plan:
        assert torch.equal(torch.cat(got[slot]), reference[slot]), f"slot {slot}"


def test_chunked_prefill_equals_single_prefill(kernel, gates):
    """Cuts inside and on 16-token chunks, with a neighbour, change no bit."""
    sequence = _sequence(11, 150)
    neighbour = _sequence(12, 90)
    reference = _teacher_forced(kernel, gates, sequence)

    pool = _new_pool(kernel)
    cuts = [0, 5, 16, 47, 48, 113, 150]
    outputs = []
    for index, (start, stop) in enumerate(zip(cuts[:-1], cuts[1:])):
        pieces, slots, prefixes = [_rows(sequence, start, stop)], [3], [start]
        if index % 2 == 0:
            n_start = 16 * index
            pieces.append(_rows(neighbour, n_start, n_start + 32))
            slots.append(8)
            prefixes.append(n_start)
        outputs.append(_extend(kernel, pool, gates, pieces, slots, prefixes)[0])
    assert torch.equal(torch.cat(outputs), reference)


def test_snapshots_restore_bitwise(kernel, gates):
    """Every prefix-cache snapshot is a whole state: a request restored from one,
    taken inside a prefill, before it, or after a decode step, matches."""
    sequence = _sequence(21, 200)
    short = _sequence(22, 70)
    reference = _teacher_forced(kernel, gates, sequence)
    short_reference = _teacher_forced(kernel, gates, short)

    pool = _new_pool(kernel)
    track_state = torch.zeros(3, HEADS, DIM, DIM, device=DEVICE)
    # Row 0 snapshots after two 64-token state chunks (token 128), row 1 is
    # untracked, row 2 snapshots the state it starts from (chunk 0).
    outputs = _extend(
        kernel,
        pool,
        gates,
        [_rows(sequence, 0, 150), short, _rows(short, 0, 40)],
        [2, 6, 7],
        [0, 0, 0],
        return_intermediate_states=True,
        track_state=track_state,
        track_chunk_idx=torch.tensor([2, -1, 0], dtype=torch.int32, device=DEVICE),
    )
    assert torch.equal(outputs[0], reference[:150])
    assert torch.equal(outputs[1], short_reference)
    assert torch.equal(outputs[2], short_reference[:40])
    assert not track_state[2].any()

    # A second request reuses the cached 128-token prefix in another slot.
    pool[9].copy_(track_state[0])
    resumed = _extend(kernel, pool, gates, [_rows(sequence, 128, 200)], [9], [128])[0]
    assert torch.equal(resumed, reference[128:])

    # Decode slot 2 up to token 192, where SGLang's decode tracking copies the slot.
    for position in range(150, 192):
        _decode(kernel, pool, gates, [_rows(sequence, position, position + 1)], [2])
    pool[10].copy_(pool[2])
    resumed = _extend(kernel, pool, gates, [_rows(sequence, 192, 200)], [10], [192])[0]
    assert torch.equal(resumed, reference[192:])


def test_side_state_bytes_match_flashinfer(kernel):
    """The memory budget charges what FlashInfer allocates beside the pool."""
    from flashinfer.batch_invariant.kda.pool import get_pool

    from sglang.srt.layers.attention.linear.kernels.kda_flashinfer_batch_invariant import (
        kda_batch_invariant_state_bytes_per_slot,
    )

    pool = _new_pool(kernel)
    side = get_pool(pool)
    allocated = sum(
        t.numel() * t.element_size()
        for t in vars(side).values()
        if isinstance(t, torch.Tensor)
    )
    assert allocated == SLOTS * kda_batch_invariant_state_bytes_per_slot(
        num_layers=1, num_heads=HEADS
    )


def test_reused_slot_forgets_the_previous_request(kernel, gates):
    """A slot freed inside its first chunk still holds the zero boundary state, so
    only the explicit reset tells the next request's rows from the old ones."""
    first, second = _sequence(31, 10), _sequence(32, 30)
    reference = _teacher_forced(kernel, gates, second)

    pool = _new_pool(kernel)
    _extend(kernel, pool, gates, [_rows(first, 0, 9)], [4], [0])
    _decode(kernel, pool, gates, [_rows(first, 9, 10)], [4])
    pool[4].zero_()
    # One prompt token: the call shape FlashInfer would read as a decode step.
    outputs = [_extend(kernel, pool, gates, [_rows(second, 0, 1)], [4], [0])[0]]
    for position in range(1, 30):
        output = _decode(
            kernel, pool, gates, [_rows(second, position, position + 1)], [4]
        )
        outputs.append(output.reshape(1, HEADS, DIM))
    assert torch.equal(torch.cat(outputs), reference)


def test_resume_inside_a_chunk_from_a_foreign_state_is_refused(kernel, gates):
    """Inside a chunk a slot may only continue its own prefill: not a state that
    decode steps advanced, and not another slot's."""
    pool = _new_pool(kernel)
    sequence = _sequence(41, 40)
    _extend(kernel, pool, gates, [_rows(sequence, 0, 10)], [5], [0])
    _decode(kernel, pool, gates, [_rows(sequence, 10, 11)], [5])
    for slot, prefix in ((5, 11), (6, 10)):
        with pytest.raises(RuntimeError, match="continue its own previous prefill"):
            _extend(
                kernel, pool, gates, [_rows(sequence, prefix, 40)], [slot], [prefix]
            )


def test_decode_cuda_graph_replay_skips_padded_rows(kernel, gates):
    """A captured decode step replays the eager bits; padded rows (slot -1) touch nothing."""
    sequences = [_sequence(51 + i, 40) for i in range(3)]
    reference = [_teacher_forced(kernel, gates, sequence) for sequence in sequences]

    pool = _new_pool(kernel)
    batch = 4
    _extend(
        kernel, pool, gates, [_rows(s, 0, 9) for s in sequences], [1, 2, 3], [0] * 3
    )
    static = dict(
        q=torch.zeros(1, batch, HEADS, DIM, device=DEVICE, dtype=torch.bfloat16),
        k=torch.zeros(1, batch, HEADS, DIM, device=DEVICE, dtype=torch.bfloat16),
        v=torch.zeros(1, batch, HEADS, DIM, device=DEVICE, dtype=torch.bfloat16),
        a=torch.zeros(batch, HEADS * DIM, device=DEVICE, dtype=torch.bfloat16),
        b=torch.zeros(1, batch, HEADS, device=DEVICE, dtype=torch.bfloat16),
    )
    slots = torch.full((batch,), -1, dtype=torch.int32, device=DEVICE)
    query_start_loc = torch.arange(batch + 1, dtype=torch.int32, device=DEVICE)

    def step():
        return kernel.decode(
            static["q"],
            static["k"],
            static["v"],
            static["a"],
            static["b"],
            ssm_states=pool,
            cache_indices=slots,
            query_start_loc=query_start_loc,
            **gates,
        )

    untouched = pool.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = step()
    graph.replay()
    assert torch.equal(pool, untouched)

    # Requests sit in rows 2, 0, 1; row 3 stays padded.
    slots.copy_(torch.tensor([2, 3, 1, -1], dtype=torch.int32, device=DEVICE))
    row_of = {0: 2, 1: 0, 2: 1}
    for position in range(9, 40):
        for index, sequence in enumerate(sequences):
            row = row_of[index]
            q, k, v, a, b = _rows(sequence, position, position + 1)
            static["q"][0, row].copy_(q[0])
            static["k"][0, row].copy_(k[0])
            static["v"][0, row].copy_(v[0])
            static["a"][row].copy_(a[0].reshape(-1))
            static["b"][0, row].copy_(b[0])
        graph.replay()
        for index in range(3):
            assert torch.equal(output[0, row_of[index]], reference[index][position])


BACKEND_HEADS, BACKEND_CONTEXT = 2, 256


class _BackendHarness:
    """KDAAttnBackend in batch-invariant mode over the attention unit-test fixture."""

    def __init__(self):
        case = self.case(ForwardMode.EXTEND, prefix_lens=(0,), extend_lens=(1,))
        self.fixture = build_kda_attention_fixture(
            CustomTestCase(),
            case,
            head_k_dim=DIM,
            head_v_dim=DIM,
            max_context_len=BACKEND_CONTEXT,
            runner_batch_size=6,
        )
        self.fixture.actual_module.attn.lower_bound = -5.0
        with (
            get_exec().mamba.override(enable_linear_attn_batch_invariant=True),
            get_memory().override(disable_radix_cache=True),
        ):
            self.fixture.backend = HybridLinearAttnBackend(
                self.fixture.backend.full_attn_backend,
                KDAAttnBackend(self.fixture.runner),
                full_attn_layers=[],
            )
        self.cache = (
            self.fixture.runner.req_to_token_pool.mamba_pool.mamba2_layer_cache(0)
        )
        self.mixed_dim = self.fixture.actual_module.mixed_qkv_dim

    @staticmethod
    def case(forward_mode, *, prefix_lens, extend_lens=()):
        return KDAAttentionCase(
            name="kda_batch_invariant",
            backend="triton",
            forward_mode=forward_mode,
            num_k_heads=BACKEND_HEADS,
            num_v_heads=BACKEND_HEADS,
            page_size=16,
            prefix_lens=tuple(prefix_lens),
            extend_lens=tuple(extend_lens),
        )

    def request(self, seed: int, length: int):
        """mixed_qkv, raw gate and beta logit rows of one request."""
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        def randn(width):
            return torch.randn(length, width, device=DEVICE, generator=generator)

        return (
            randn(self.mixed_dim).bfloat16(),
            (randn(BACKEND_HEADS * DIM) - 4.0).bfloat16(),
            randn(BACKEND_HEADS).bfloat16(),
        )

    def clear(self):
        self.cache.conv[0].zero_()
        self.cache.temporal.zero_()

    def forward(self, forward_mode, pieces, slots, prefix_lens, track=None):
        """One forward over ``pieces`` (row slices of requests) held in ``slots``."""
        lens = [piece[0].shape[0] for piece in pieces]
        runner = self.fixture.runner
        batch = _make_forward_batch(
            self.case(forward_mode, prefix_lens=prefix_lens, extend_lens=lens),
            runner,
            max_context_len=BACKEND_CONTEXT,
            device=DEVICE,
        )
        runner.req_to_token_pool.req_index_to_mamba_index_mapping[
            batch.req_pool_indices
        ] = torch.tensor(slots, dtype=torch.int32, device=DEVICE)
        if track is not None:
            tracked, track_slots, track_seqlens = track
            batch.mamba_track_mask = torch.tensor(tracked, device=DEVICE)
            batch.mamba_track_indices = torch.tensor(
                track_slots, dtype=torch.int32, device=DEVICE
            )
            batch.mamba_track_seqlens = torch.tensor(
                track_seqlens, dtype=torch.int32, device=DEVICE
            )
            batch.mamba_prefill_track_mask_cpu = list(tracked)
            batch.mamba_track_seqlens_cpu = list(track_seqlens)
        mixed_qkv = torch.cat([piece[0] for piece in pieces])
        a = torch.cat([piece[1] for piece in pieces])
        if not forward_mode.is_decode():
            a = a.unflatten(-1, (BACKEND_HEADS, DIM)).unsqueeze(0)
        b = torch.cat([piece[2] for piece in pieces]).unsqueeze(0)
        with (
            torch.no_grad(),
            get_parallel().override(attn_dcp_rank=0, attn_dcp_size=1),
            forward_context(ForwardContext(attn_backend=self.fixture.backend)),
        ):
            self.fixture.backend.init_forward_metadata(batch)
            output = self.fixture.actual_module(batch, mixed_qkv, a, b)
        return list(output[0].split(lens))

    def teacher_forced(self, request):
        self.clear()
        return self.forward(ForwardMode.EXTEND, [request], [1], [0])[0].clone()


@pytest.fixture(scope="module")
def harness():
    return _BackendHarness()


def test_backend_decode_equals_teacher_forced_prefill(harness):
    """KDA layer output through the backend, convolution included: decode-stepped
    requests in a shrinking batch equal their own prefill bit for bit."""
    prompts, totals, slots = (20, 1, 33), (45, 30, 52), (3, 5, 2)
    requests = [harness.request(200 + i, total) for i, total in enumerate(totals)]
    reference = [harness.teacher_forced(request) for request in requests]

    harness.clear()
    got = [
        [output]
        for output in harness.forward(
            ForwardMode.EXTEND,
            [_rows(r, 0, p) for r, p in zip(requests, prompts)],
            slots,
            [0, 0, 0],
        )
    ]
    position = list(prompts)
    while any(p < t for p, t in zip(position, totals)):
        running = [i for i in range(3) if position[i] < totals[i]]
        outputs = harness.forward(
            ForwardMode.DECODE,
            [_rows(requests[i], position[i], position[i] + 1) for i in running],
            [slots[i] for i in running],
            [position[i] for i in running],
        )
        for i, output in zip(running, outputs):
            got[i].append(output)
            position[i] += 1

    for i in range(3):
        assert torch.equal(torch.cat(got[i]), reference[i]), f"request {i}"


def test_backend_prefix_checkpoint_restores_bitwise(harness):
    """The backend's prefix-cache snapshot (state and conv window) of a tracked
    prefill lets a second request resume from it with the first one's bits."""
    request = harness.request(300, 200)
    reference = harness.teacher_forced(request)

    harness.clear()
    # 150 tokens with a 64-token state chunk: the snapshot is taken at token 128.
    first = harness.forward(
        ForwardMode.EXTEND,
        [_rows(request, 0, 150)],
        [1],
        [0],
        track=([True], [4], [150]),
    )[0]
    assert torch.equal(first, reference[:150])

    harness.cache.conv[0][2].copy_(harness.cache.conv[0][4])
    harness.cache.temporal[2].copy_(harness.cache.temporal[4])
    resumed = harness.forward(
        ForwardMode.EXTEND, [_rows(request, 128, 200)], [2], [128]
    )[0]
    assert torch.equal(resumed, reference[128:])

def test_head_count_outside_the_flashinfer_class_is_refused(kernel):
    """The class's head counts are FlashInfer's; a pool it does not cover never registers."""
    pool = torch.zeros(2, 20, DIM, DIM, device=DEVICE)
    with pytest.raises(ValueError, match=r"\(20 KDA heads per rank\)"):
        kernel.register_state_pool(pool)


def test_decode_equals_prefill_at_the_kimi_k3_head_count(kernel):
    """96 heads (Kimi-K3 at TP 1): mixed-batch decode rows are the request's prefill rows."""
    heads = 96
    try:
        pool = _new_pool(kernel, heads)
    except ValueError as exc:
        # Only FlashInfer's own refusal of this head count is a reason to skip.
        if f"does not support {heads} heads" not in str(exc):
            raise
        pytest.skip(f"this FlashInfer build does not cover {heads} heads: {exc}")
    generator = torch.Generator(device=DEVICE).manual_seed(9)
    gates = dict(
        A_log=torch.randn(heads, device=DEVICE, generator=generator) * 0.2,
        dt_bias=torch.randn(heads * DIM, device=DEVICE, generator=generator) * 0.1,
        lower_bound=-5.0,
    )
    prompts = {1: 5, 2: 16, 3: 37}
    total = 70
    sequences = {slot: _sequence(300 + slot, total, heads) for slot in prompts}
    reference = {}
    for slot, sequence in sequences.items():
        alone = _new_pool(kernel, heads)
        reference[slot] = _extend(kernel, alone, gates, [sequence], [1], [0])[0]

    slots = list(prompts)
    got = {
        slot: [output]
        for slot, output in zip(
            slots,
            _extend(
                kernel,
                pool,
                gates,
                [_rows(sequences[s], 0, prompts[s]) for s in slots],
                slots,
                [0] * len(slots),
            ),
        )
    }
    for step in range(total - max(prompts.values())):
        order = slots[step % len(slots) :] + slots[: step % len(slots)]
        position = {s: prompts[s] + step for s in order}
        output = _decode(
            kernel,
            pool,
            gates,
            [_rows(sequences[s], position[s], position[s] + 1) for s in order],
            order,
        )
        for row, slot in enumerate(order):
            got[slot].append(output[row : row + 1].reshape(1, heads, DIM))
    for slot in slots:
        produced = torch.cat(got[slot])
        assert torch.equal(produced, reference[slot][: produced.shape[0]]), slot


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
