import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=400, stage="base-c", runner_config="4-gpu-b200")

QWEN3_NEXT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"


class TestQwen3NextMTPTopkFlashInferTreeVerify(GSM8KMixin, DefaultServerBase):
    # Same tree-MTP configuration as TestQwen3NextMTPTopk (H100), with the GDN
    # target-verify and commit on the FlashInfer ReplaySSM kernels (SM100 only).
    model = QWEN3_NEXT_MODEL
    gsm8k_accuracy_thres = 0.93
    other_args = [
        "--trust-remote-code",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "4",
        "--speculative-num-draft-tokens",
        "8",
        "--mem-fraction-static",
        "0.8",
        "--tp",
        "4",
        "--chunked-prefill-size",
        "2048",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "128",
        "--enable-flashinfer-gdn-tree-verify",
    ]


if __name__ == "__main__":
    unittest.main()
