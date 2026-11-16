import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.attention_hook import handle_linear_attn_backend
from sglang.srt.layers.attention.linear.kernels.gdn_batch_invariant_spec import (
    build_extend_state_snapshots,
)
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _resolve(**overrides) -> ServerArgs:
    fields = dict(
        model_path="dummy",
        enable_linear_attn_batch_invariant=True,
        mamba_ssm_dtype="float32",
        disable_radix_cache=True,
    )
    args = ServerArgs(**{**fields, **overrides})
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["Qwen3NextForCausalLM"], linear_attention_layers=[0]
        )
    )
    with (
        patch(
            "sglang.srt.arg_groups.attention_hook.model_config_of",
            return_value=config,
        ),
        override_platform(is_sm100=True),
        override_platform(is_cuda=False),
    ):
        handle_linear_attn_backend(args)
    return args


class TestGDNBatchInvariantArgs(CustomTestCase):
    def test_accepts_fp32_state_without_radix_cache(self):
        args = _resolve()
        # The fp32 pool keeps the FlashInfer bf16-pool decode default away.
        self.assertIsNone(args.linear_attn_decode_backend)

    def test_requires_fp32_state_pool(self):
        with self.assertRaisesRegex(ValueError, "--mamba-ssm-dtype float32"):
            _resolve(mamba_ssm_dtype="bfloat16")

    def test_refuses_paths_that_move_state_off_the_chunk_grid(self):
        """Each of these hands a slot's state over without its open 64-token chunk."""
        cases = (
            (dict(speculative_algorithm="NEXTN"), "speculative decoding"),
            (dict(enable_linear_replayssm=True), "--enable-linear-replayssm"),
            (dict(enable_int8_mamba_checkpoint=True), "int8-mamba-checkpoint"),
            (dict(enable_unified_memory=True), "--enable-unified-memory"),
            (dict(enable_hierarchical_cache=True), "--enable-hierarchical-cache"),
            (dict(disaggregation_mode="decode"), "PD disaggregation"),
            (dict(enable_mis=True), "--enable-mis"),
            (dict(enable_mixed_chunk=True), "--enable-mixed-chunk"),
            (dict(enable_two_batch_overlap=True), "--enable-two-batch-overlap"),
            (dict(enable_streaming_session=True), "--enable-streaming-session"),
            (
                dict(disable_radix_cache=False, mamba_radix_cache_strategy="no_buffer"),
                "extra_buffer",
            ),
            (dict(page_size=48), "divides 64"),
        )
        for overrides, message in cases:
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                _resolve(**overrides)

    def test_accepts_extra_buffer_radix_cache(self):
        _resolve(disable_radix_cache=False, mamba_radix_cache_strategy="extra_buffer")

    def test_snapshot_boundaries_follow_the_scheduler_track_lengths(self):
        """The scheduler names a snapshot by a tracked length: the extend's end
        when that is the snapshot, one past the boundary otherwise (a coarser
        grid or a branch point inside the extend)."""
        snapshots = build_extend_state_snapshots(
            # end on a boundary, end past it, forced boundary + 1, branch + 1, untracked
            track_mask=[True, True, True, True, False],
            track_seqlens=[64 + 192, 128 + 200, 0 + 129, 256 + 65, 0],
            prefix_lens=[64, 128, 0, 256, 0],
            destinations=None,
        )
        self.assertEqual(snapshots.rows, [0, 1, 2, 3])
        self.assertEqual(snapshots.boundaries, [192, 192, 128, 64])


if __name__ == "__main__":
    unittest.main()
