"""Each runner carries its own linear-attn kernel choice.

A target and its draft coexist in one process and can want different kernels:
only eligible GDN/KDA runners get the FlashInfer prefill default,
and the operator's explicit flag applies to the launch. So the choice is a
per-runner stamp read from the runner, the way the full-attention pair already
works (`prefill_attention_backend_str` / `decode_attention_backend_str`), rather
than one process-wide table.

It used to be that table: `attn_backend_wrapper` rebuilt a module-level dict
once per runner, from the handed record plus a local default. Two consequences,
both pinned below -- a second runner could not hold a different choice, and its
rebuild replaced the first runner's (the record never carries the recorded
default, so a runner with no default of its own resolved back to the base
backend).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    resolve_linear_attn_backends,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _Runner:
    """The only thing the readers need from a runner: the stamp."""


class TestLinearAttnBackends(CustomTestCase):
    def _publish(self, **fields):
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args(**fields)
        override.install()
        self.addCleanup(override.restore)

    def test_the_default_applies_when_the_flag_is_unset(self):
        self._publish(linear_attn_backend="triton")
        backends = resolve_linear_attn_backends(prefill_default="flashinfer")
        self.assertEqual(backends.prefill, LinearAttnKernelBackend.FLASHINFER)

    def test_the_default_does_not_reach_the_decode_backend(self):
        self._publish(linear_attn_backend="triton")
        backends = resolve_linear_attn_backends(prefill_default="flashinfer")
        self.assertEqual(backends.decode, LinearAttnKernelBackend.TRITON)

    def test_explicit_flag_wins_over_the_default(self):
        """Precedence lives in the gate, which declines once the flag is set.

        The resolver takes the default as an argument, so the flag has to win
        upstream: `flashinfer_gdn_prefill_default` returns None the moment the
        leaf is set, and that condition is checked before anything touches the
        device -- which is what lets this run anywhere.
        """
        from types import SimpleNamespace

        from sglang.srt.layers.attention.linear.gdn_backend import (
            flashinfer_gdn_prefill_default,
        )
        from sglang.srt.runtime_context import get_server_args

        self._publish(
            linear_attn_backend="triton", linear_attn_prefill_backend="cutedsl"
        )
        runner = SimpleNamespace(server_args=get_server_args())
        self.assertIsNone(flashinfer_gdn_prefill_default(runner))
        self.assertEqual(
            resolve_linear_attn_backends().prefill, LinearAttnKernelBackend.CUTEDSL
        )

    def test_two_runners_hold_different_choices(self):
        """The property the process-wide table could not express.

        The GDN target gets the SM100 default; the draft that is not GDN has no
        default of its own. Both stamps stand, and reading one does not disturb
        the other -- under the old table the draft's rebuild replaced the
        target's choice with the base backend.
        """
        self._publish(linear_attn_backend="triton")

        target, draft = _Runner(), _Runner()
        target.linear_attn_backends = resolve_linear_attn_backends(
            prefill_default="flashinfer"
        )
        draft.linear_attn_backends = resolve_linear_attn_backends()

        self.assertEqual(
            target.linear_attn_backends.prefill, LinearAttnKernelBackend.FLASHINFER
        )
        self.assertEqual(
            draft.linear_attn_backends.prefill, LinearAttnKernelBackend.TRITON
        )

    def test_an_unstamped_runner_raises_rather_than_guessing(self):
        """No default for "nobody stamped this", on the production read path.

        `attn_backend_wrapper` stamps before it builds the backends that read
        the stamp, so a missing one means a backend was built outside that
        path. The runner double below satisfies everything `GDNAttnBackend`
        touches *before* the stamp read, so the `AttributeError` this asserts
        comes from `model_runner.linear_attn_backends` itself -- a default
        stamp on the runner or a restored module-level fallback would turn
        this red-to-green, which is the regression it guards. Silent triton
        fallback would hide the wiring mistake behind a working-but-wrong
        kernel.
        """
        import torch

        from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend

        # The draft-token width is a bag leaf read before the stamp.
        self._publish(speculative_eagle_topk=0)
        runner = SimpleNamespace(
            device="cpu",
            server_args=SimpleNamespace(enable_unified_memory=False),
            is_draft_worker=False,
            req_to_token_pool=SimpleNamespace(
                mamba_pool=SimpleNamespace(
                    mamba_cache=SimpleNamespace(conv=[torch.zeros(1, 1, 4, 7)])
                )
            ),
            token_to_kv_pool=None,
        )
        with self.assertRaises(AttributeError) as caught:
            GDNAttnBackend(runner)
        self.assertIn("linear_attn_backends", str(caught.exception))

    def test_the_per_runner_default_stays_out_of_the_process_config(self):
        """The process-wide config cannot represent one runner's choice.

        Recording the auto-default there is how a second runner used to inherit
        it: the leaf then reads as "the operator named a backend", which is the
        one question the gate asks. So the default lives in the runner's stamp
        and the leaf keeps meaning what was asked for at launch.
        """
        from sglang.srt.runtime_context import get_context, get_exec

        self._publish(linear_attn_backend="triton")
        backends = resolve_linear_attn_backends(prefill_default="flashinfer")

        self.assertEqual(backends.prefill, LinearAttnKernelBackend.FLASHINFER)
        self.assertIsNone(get_exec().mamba.linear_attn_prefill_backend)
        self.assertIsNone(
            get_context().resolved_server_args_dict()["linear_attn_prefill_backend"]
        )


class TestFlashInferKDAPrefillPolicy(CustomTestCase):
    def test_default_preserves_model_hardware_and_operator_choices(self):
        from sglang.srt.configs.bailing_hybrid import BailingHybridConfig
        from sglang.srt.configs.glm5_next import Glm5NextConfig
        from sglang.srt.configs.kimi_k3 import KimiK3Config
        from sglang.srt.configs.kimi_linear import KimiLinearConfig
        from sglang.srt.layers.attention.linear.kda_backend import (
            flashinfer_kda_prefill_default,
        )
        from sglang.srt.model_executor.cuda_graph_config import (
            CudaGraphConfig,
            PhaseConfig,
        )
        from sglang.srt.runtime_context import get_context, get_exec

        linear_config = {"kda_layers": [0], "full_attn_layers": []}
        k3 = KimiK3Config(
            text_config={
                "linear_attn_config": {**linear_config, "gate_lower_bound": -5}
            }
        )
        safe_configs = [
            k3,
            Glm5NextConfig(text_config={"gate_lower_bound": -5}),
            BailingHybridConfig(use_kda=True, kda_safe_gate=True, kda_lower_bound=-5),
        ]
        cases = [(config, (10, 3), {}, "flashinfer") for config in safe_configs]
        cases += [
            (k3, (10, 0), {}, "flashinfer"),
            (k3, (9, 0), {}, "triton"),
            (k3, (12, 0), {}, "triton"),
            (k3, None, {}, "triton"),
            (KimiLinearConfig(linear_attn_config=linear_config), (10, 3), {}, "triton"),
            (BailingHybridConfig(use_kda=True), (10, 3), {}, "triton"),
        ]
        cases += [
            (k3, (10, 3), fields, expected)
            for fields, expected in [
                ({"linear_attn_prefill_backend": "triton"}, "triton"),
                ({"linear_attn_prefill_backend": "helion"}, "helion"),
                ({"linear_attn_backend": "helion"}, "helion"),
                ({"enable_deterministic_inference": True}, "triton"),
                ({"enable_two_batch_overlap": True}, "triton"),
                (
                    {
                        "cuda_graph_config": CudaGraphConfig(
                            prefill=PhaseConfig(backend="full")
                        )
                    },
                    "triton",
                ),
            ]
        ]
        for config, capability, fields, expected in cases:
            runner = SimpleNamespace(
                model_config=SimpleNamespace(hf_config=config, is_draft_model=False)
            )
            with (
                self.subTest(
                    model=config.model_type, capability=capability, fields=fields
                ),
                get_context().override_server_args(
                    **{
                        "linear_attn_backend": "triton",
                        "cuda_graph_config": CudaGraphConfig(
                            prefill=PhaseConfig(backend="disabled")
                        ),
                        **fields,
                    }
                ),
                patch(
                    "sglang.srt.layers.attention.linear.kda_backend.is_cuda",
                    return_value=capability is not None,
                ),
                patch("torch.cuda.get_device_capability", return_value=capability),
            ):
                backends = resolve_linear_attn_backends(
                    prefill_default=flashinfer_kda_prefill_default(runner)
                )
                self.assertEqual(backends.prefill.value, expected)
                self.assertEqual(
                    backends.decode.value, fields.get("linear_attn_backend", "triton")
                )
                self.assertEqual(backends.verify, LinearAttnKernelBackend.TRITON)
                self.assertEqual(
                    get_exec().mamba.linear_attn_prefill_backend,
                    fields.get("linear_attn_prefill_backend"),
                )

    def test_fixed_restrictions_are_rejected_before_forward(self):
        from sglang.srt.layers.attention.linear.kda_backend import (
            _validate_flashinfer_kda_prefill,
        )

        for chunk, tbo, graph, error in [
            (0, False, "disabled", "checkpoint interval"),
            (33, False, "disabled", "checkpoint interval"),
            (64, True, "disabled", "two-batch overlap"),
            (64, False, "full", "eager linear attention"),
            (64, False, "breakable", None),
        ]:
            with self.subTest(chunk=chunk, tbo=tbo, graph=graph):
                kwargs = dict(
                    chunk_size=chunk,
                    enable_two_batch_overlap=tbo,
                    prefill_cuda_graph_backend=graph,
                )
                if error is not None:
                    with self.assertRaisesRegex(ValueError, error):
                        _validate_flashinfer_kda_prefill(**kwargs)
                else:
                    _validate_flashinfer_kda_prefill(**kwargs)

    def test_invalid_checkpoint_metadata_fails_instead_of_falling_back(self):
        import torch

        from sglang.srt.layers.attention.linear.kernels.kda_flashinfer_prefill import (
            build_flashinfer_kda_checkpoint_plan,
        )

        for extend_len, track_len, rows in (
            (128, 1, [0]),
            (128, 193, [0]),
            (130, 130, []),
        ):
            with self.subTest(extend_len=extend_len, track_len=track_len, rows=rows):
                batch = SimpleNamespace(
                    extend_seq_lens_cpu=[extend_len],
                    extend_prefix_lens_cpu=[0],
                    mamba_track_seqlens_cpu=[track_len],
                    mamba_prefill_track_mask_cpu=[True],
                )
                metadata = SimpleNamespace(
                    track_ssm_h_batch_src=torch.tensor(rows, dtype=torch.int32)
                )
                with self.assertRaisesRegex(AssertionError, "KDA checkpoint"):
                    build_flashinfer_kda_checkpoint_plan(batch, metadata, "cpu", 64)


class TestKDABatchInvariantPolicy(CustomTestCase):
    """The batch-invariant kernel keeps a slot's state at its last 16-token
    boundary, so the mode must refuse every feature that moves a slot elsewhere."""

    @staticmethod
    def _config(**overrides):
        from sglang.srt.model_executor.cuda_graph_config import Backend

        config = dict(
            speculative_algorithm=None,
            enable_linear_replayssm=False,
            enable_int8_mamba_checkpoint=False,
            enable_unified_memory=False,
            disaggregation_mode="null",
            enable_hierarchical_cache=False,
            dcp_enabled=False,
            enable_mixed_chunk=False,
            enable_two_batch_overlap=False,
            enable_nvidia_megakernel=False,
            enable_streaming_session=False,
            prefill_cuda_graph_backend=Backend.DISABLED,
            radix_cache_enabled=True,
            enable_mamba_extra_buffer=True,
            mamba_track_interval=256,
            cache_chunk_size=64,
        )
        config.update(overrides)
        return config

    def test_every_state_moving_feature_is_refused(self):
        from sglang.srt.layers.attention.linear.kda_batch_invariant import (
            kda_batch_invariant_refusal,
        )
        from sglang.srt.model_executor.cuda_graph_config import Backend

        self.assertIsNone(kda_batch_invariant_refusal(**self._config()))
        self.assertIsNone(
            kda_batch_invariant_refusal(
                **self._config(
                    radix_cache_enabled=False,
                    enable_mamba_extra_buffer=False,
                    mamba_track_interval=100,
                )
            )
        )
        self.assertIsNone(
            kda_batch_invariant_refusal(
                **self._config(prefill_cuda_graph_backend=Backend.BREAKABLE)
            )
        )
        for overrides in (
            dict(speculative_algorithm="EAGLE"),
            dict(enable_linear_replayssm=True),
            dict(enable_int8_mamba_checkpoint=True),
            dict(enable_unified_memory=True),
            dict(disaggregation_mode="decode"),
            dict(enable_hierarchical_cache=True),
            dict(dcp_enabled=True),
            dict(enable_mixed_chunk=True),
            dict(enable_two_batch_overlap=True),
            dict(enable_nvidia_megakernel=True),
            dict(enable_streaming_session=True),
            dict(prefill_cuda_graph_backend=Backend.FULL),
            # no_buffer caches the live slot at any token.
            dict(enable_mamba_extra_buffer=False),
            dict(mamba_track_interval=100),
            dict(cache_chunk_size=24),
        ):
            with self.subTest(**overrides):
                self.assertIsNotNone(
                    kda_batch_invariant_refusal(**self._config(**overrides))
                )

    def test_models_outside_the_kernel_class_are_named(self):
        import torch

        from sglang.srt.layers.attention.linear.kernels.kda_flashinfer_batch_invariant import (
            kda_batch_invariant_unsupported_reason,
        )

        # Ling-3.0-flash at TP 1 and TP 4, Kimi-K3 at TP 1 and TP 8;
        # the head count is FlashInfer's to accept, at kernel construction.
        ling = dict(
            head_k_dim=128, head_v_dim=128, lower_bound=-5.0, state_dtype=torch.float32
        )
        for heads in (32, 8, 96, 12):
            self.assertIsNone(
                kda_batch_invariant_unsupported_reason(
                    num_k_heads=heads, num_v_heads=heads, **ling
                )
            )
        for overrides in (
            dict(num_k_heads=32, num_v_heads=96),
            # Kimi-Linear has the softplus gate.
            dict(lower_bound=None),
            dict(head_k_dim=32, head_v_dim=32),
            dict(state_dtype=torch.bfloat16),
        ):
            with self.subTest(**overrides):
                config = dict(num_k_heads=32, num_v_heads=32, **ling)
                config.update(overrides)
                self.assertIsNotNone(kda_batch_invariant_unsupported_reason(**config))

    def test_extend_plan_keeps_snapshots_on_the_16_token_grid(self):
        from sglang.srt.layers.attention.linear.kernels.kda_flashinfer_batch_invariant import (
            plan_kda_batch_invariant_extend,
        )

        # Row 0 snapshots at token 32 + 128 and finishes behind it; row 1 resumes
        # inside a chunk; row 2 is empty.
        first, second = plan_kda_batch_invariant_extend(
            extend_lens=[150, 9, 0], prefix_lens=[32, 21, 0], cut_lens=[128, 0, 0]
        )
        self.assertEqual(
            [(s.row, s.start, s.length, s.at_boundary) for s in first],
            [(0, 0, 128, True), (1, 150, 9, False)],
        )
        self.assertEqual(
            [(s.row, s.start, s.length, s.at_boundary) for s in second],
            [(0, 128, 22, True)],
        )
        # Token 21 + 64 is inside a chunk: the slot never holds that state.
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            plan_kda_batch_invariant_extend(
                extend_lens=[100], prefix_lens=[21], cut_lens=[64]
            )

    def test_prefill_cuts_land_on_the_checkpoint_chunk(self):
        """With the prefix cache on, a cut inside a chunk would put the next
        prefix-relative snapshot off the kernel's grid."""
        from sglang.srt.managers.schedule_policy import mamba_prefill_chunk_grid
        from sglang.srt.runtime_context import get_context

        for fields, expected in (
            (dict(enable_linear_attn_batch_invariant=True), 64),
            (dict(enable_linear_attn_batch_invariant=False), 1),
            (
                dict(enable_linear_attn_batch_invariant=True, disable_radix_cache=True),
                1,
            ),
        ):
            with self.subTest(**fields):
                override = get_context().override_server_args(
                    mamba_radix_cache_strategy="extra_buffer",
                    uses_mamba_radix_cache=True,
                    _mamba_cache_chunk_size=64,
                    **fields,
                )
                override.install()
                try:
                    self.assertEqual(mamba_prefill_chunk_grid(), expected)
                finally:
                    override.restore()


if __name__ == "__main__":
    unittest.main()
