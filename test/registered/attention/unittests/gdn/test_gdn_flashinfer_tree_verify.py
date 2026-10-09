"""FlashInfer ReplaySSM tree verify + accept-path commit vs the Triton tree verify
+ snapshot scatter, on the tensors the GDN backend hands each path.

Random EAGLE-style trees (parent[i] < i; SGLang writes the root parent as 0,
FlashInfer wants -1) and one random accepted root-to-node path per request.
"""

import unittest

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    flashinfer_gdn_tree_verify_missing,
)
from sglang.srt.speculative.spec_utils import flashinfer_accept_paths
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="4-gpu-b200")

DEVICE = "cuda"
K = V = 128
_sm_major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
_missing = (
    flashinfer_gdn_tree_verify_missing() if torch.cuda.is_available() else "no CUDA"
)


def _random_tree(T, gen):
    parents = [-1]
    for i in range(1, T):
        parents.append(int(torch.randint(0, i, (1,), generator=gen)))
    return parents


def _random_path(parents, gen):
    node = int(torch.randint(0, len(parents), (1,), generator=gen))
    path, cur = [], node
    while cur >= 0:
        path.append(cur)
        cur = parents[cur]
    return path[::-1]


def _make(B, T, HV, HK, seed):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    bf = dict(device=DEVICE, dtype=torch.bfloat16)
    slots = B + 3
    idx = torch.randperm(slots, generator=gen)[:B].to(DEVICE, torch.int32)
    trees = [_random_tree(T, gen) for _ in range(B)]
    paths = [_random_path(t, gen) for t in trees]
    parents_sgl = torch.tensor(trees, device=DEVICE, dtype=torch.int32)
    parents_sgl[:, 0] = 0
    accept_lens = torch.tensor(
        [len(p) for p in paths], device=DEVICE, dtype=torch.int32
    )
    # Global node ids, -1 padded, as EagleVerifyInput emits them.
    accept_index = torch.full((B, T), -1, device=DEVICE, dtype=torch.int32)
    for i, p in enumerate(paths):
        accept_index[i, : len(p)] = (
            torch.tensor(p, device=DEVICE, dtype=torch.int32) + i * T
        )
    return dict(
        q=torch.randn(B, T, HK, K, **bf),
        k=torch.randn(B, T, HK, K, **bf),
        v=torch.randn(B, T, HV, V, **bf),
        a=torch.randn(B, T, HV, **bf) * 0.5,
        b=torch.randn(B, T, HV, **bf),
        A_log=torch.randn(HV, device=DEVICE),
        dt_bias=torch.randn(HV, device=DEVICE) * 0.1,
        pool=torch.randn(slots, HV, V, K, device=DEVICE) * 0.1,
        idx=idx,
        parents_sgl=parents_sgl,
        accept_lens=accept_lens,
        accept_index=accept_index,
        last_step=torch.tensor(
            [p[-1] for p in paths], device=DEVICE, dtype=torch.int64
        ),
        B=B,
        T=T,
        HV=HV,
        HK=HK,
        slots=slots,
    )


def _triton_verify_commit(d):
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_mamba_state_scatter_with_mask,
    )

    pool = d["pool"].clone()
    B, T, HV = d["B"], d["T"], d["HV"]
    snapshots = torch.zeros(B, T, HV, V, K, device=DEVICE)
    out = fused_sigmoid_gating_delta_rule_update(
        A_log=d["A_log"],
        a=d["a"],
        dt_bias=d["dt_bias"],
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=d["q"],
        k=d["k"],
        v=d["v"],
        b=d["b"],
        initial_state_source=pool,
        initial_state_indices=d["idx"],
        use_qk_l2norm_in_kernel=True,
        is_kda=False,
        disable_state_update=True,
        intermediate_states_buffer=snapshots,
        intermediate_state_indices=torch.arange(B, device=DEVICE, dtype=torch.int32),
        cache_steps=T,
        retrieve_parent_token=d["parents_sgl"],
    ).reshape(B, T, HV, V)
    fused_mamba_state_scatter_with_mask(
        pool.unsqueeze(0), snapshots.unsqueeze(0), d["idx"], d["last_step"]
    )
    return out, pool


def _flashinfer_verify_commit(d, tree=True):
    from flashinfer.gdn_decode import (
        gated_delta_rule_mtp,
        gated_delta_rule_replayssm_commit,
    )

    pool = d["pool"].clone()
    S, T, HV, HK = d["slots"], d["T"], d["HV"], d["HK"]
    bf = dict(device=DEVICE, dtype=torch.bfloat16)
    rawv = torch.zeros(S, HV, T, V, **bf)
    rawk = torch.zeros(S, HK, T, K, **bf)
    g = torch.zeros(S, HV, T, device=DEVICE)
    beta = torch.zeros(S, HV, T, device=DEVICE)
    parents = None
    if tree:
        parents = d["parents_sgl"].clone()
        parents[:, 0] = -1
    out, _ = gated_delta_rule_mtp(
        d["q"],
        d["k"],
        d["v"],
        initial_state=pool,
        initial_state_indices=d["idx"],
        A_log=d["A_log"],
        a=d["a"],
        dt_bias=d["dt_bias"],
        b=d["b"],
        disable_state_update=True,
        use_qk_l2norm=True,
        cache_replayssm=True,
        verify_parents=parents,
        replayssm_rawv=rawv,
        replayssm_rawk=rawk,
        replayssm_g=g,
        replayssm_beta=beta,
    )
    gated_delta_rule_replayssm_commit(
        pool.unsqueeze(0),
        rawv.unsqueeze(0),
        rawk.unsqueeze(0),
        g.unsqueeze(0),
        beta.unsqueeze(0),
        d["idx"],
        d["accept_lens"],
        flashinfer_accept_paths(d["accept_index"], d["B"], T),
    )
    return out, pool


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


class TestFlashInferAcceptPaths(CustomTestCase):
    def test_global_to_local_rows_and_padding(self):
        # Global ids stride by draft_token_num; row 1's nodes live at 4..7.
        accept_index = torch.tensor(
            [[0, 2, -1], [4 + 1, 4 + 3, 4 + 0]], dtype=torch.int32
        )
        paths = flashinfer_accept_paths(accept_index, bs=2, draft_token_num=4)
        self.assertEqual(paths.shape, (2, 4))
        self.assertEqual(paths.dtype, torch.int32)
        self.assertEqual(paths[0, :2].tolist(), [0, 2])
        self.assertEqual(paths[1, :3].tolist(), [1, 3, 0])
        # Columns past the tree depth stay padding; the commit never reads them.
        self.assertEqual(paths[:, 3].tolist(), [-1, -1])

    def test_tree_depth_wider_than_draft_is_truncated(self):
        accept_index = torch.arange(10, dtype=torch.int32).reshape(1, 10)
        paths = flashinfer_accept_paths(accept_index, bs=1, draft_token_num=8)
        self.assertEqual(paths.tolist(), [list(range(8))])


@unittest.skipIf(
    _sm_major != 10 or _missing is not None,
    f"needs SM100 and a FlashInfer with ReplaySSM tree verify ({_missing})",
)
class TestFlashInferGDNTreeVerifyMatchesTriton(CustomTestCase):
    # bf16 output floor and the tcgen05 fold's TF32 state, from the kernel A/B.
    OUT_RTOL = 5e-3
    STATE_RTOL = 2e-4

    def _check(self, B, T, HV, HK, seed):
        d = _make(B, T, HV, HK, seed)
        out_t, pool_t = _triton_verify_commit(d)
        out_f, pool_f = _flashinfer_verify_commit(d)
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(out_f).all() and torch.isfinite(pool_f).all())
        self.assertLess(
            _rel(out_f, out_t), self.OUT_RTOL, f"{B=} {T=} {HV=} {HK=} {seed=}"
        )
        live = d["idx"].long()
        self.assertLess(_rel(pool_f[live], pool_t[live]), self.STATE_RTOL)
        untouched = torch.ones(d["slots"], dtype=torch.bool, device=DEVICE)
        untouched[live] = False
        self.assertTrue(torch.equal(pool_f[untouched], d["pool"][untouched]))

    def test_random_trees(self):
        for B, T, HV, HK, seed in [
            (4, 8, 8, 4, 0),
            (16, 8, 32, 16, 1),
            (33, 6, 8, 4, 2),
            (7, 4, 8, 4, 3),
            (8, 8, 8, 4, 4),
        ]:
            with self.subTest(B=B, T=T, HV=HV, HK=HK, seed=seed):
                self._check(B, T, HV, HK, seed)

    def test_chain_parents_match_no_parents(self):
        d = _make(6, 8, 8, 4, seed=7)
        for i in range(d["B"]):
            d["parents_sgl"][i] = torch.arange(-1, d["T"] - 1, device=DEVICE).clamp_min(
                0
            )
            d["accept_index"][i] = torch.arange(d["T"], device=DEVICE) + i * d["T"]
        d["accept_lens"].fill_(d["T"])
        d["last_step"].fill_(d["T"] - 1)
        out_tree, pool_tree = _flashinfer_verify_commit(d, tree=True)
        out_chain, pool_chain = _flashinfer_verify_commit(d, tree=False)
        torch.cuda.synchronize()
        self.assertLess(_rel(out_tree, out_chain), 1e-5)
        self.assertLess(_rel(pool_tree, pool_chain), 1e-5)


if __name__ == "__main__":
    unittest.main()
