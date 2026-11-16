"""Policy of --enable-linear-attn-batch-invariant for KDA: what it refuses and how its kernel is built."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kda_flashinfer_batch_invariant import (
    KDA_BATCH_INVARIANT_CHUNK,
    FlashInferKDABatchInvariantKernel,
    kda_batch_invariant_unsupported_reason,
)
from sglang.srt.model_executor.cuda_graph_config import Backend as CudaGraphBackend
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_memory,
    get_parallel,
    get_schedule,
    get_serving,
    get_spec,
    mamba_cache_chunk_size,
)
from sglang.srt.utils.log_utils import rank0_log

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig


def kda_batch_invariant_refusal(
    *,
    speculative_algorithm: Optional[str],
    enable_linear_replayssm: bool,
    enable_int8_mamba_checkpoint: bool,
    enable_unified_memory: bool,
    disaggregation_mode: str,
    enable_hierarchical_cache: bool,
    dcp_enabled: bool,
    enable_mixed_chunk: bool,
    enable_two_batch_overlap: bool,
    enable_nvidia_megakernel: bool,
    enable_streaming_session: bool,
    prefill_cuda_graph_backend: CudaGraphBackend,
    radix_cache_enabled: bool,
    enable_mamba_extra_buffer: bool,
    mamba_track_interval: int,
    cache_chunk_size: int,
) -> Optional[str]:
    """Why --enable-linear-attn-batch-invariant cannot serve this configuration.

    The kernel keeps a slot's state at its last 16-token boundary and the open
    chunk beside the pool, so every feature that reads, moves or rewrites a
    slot's state at another position, or without that side state, is refused.
    """
    if speculative_algorithm is not None:
        return "speculative decoding (verify rolls the state back per draft token)"
    if enable_linear_replayssm:
        return "--enable-linear-replayssm / --enable-linear-replayssm-spec"
    if enable_int8_mamba_checkpoint:
        return "--enable-int8-mamba-checkpoint (checkpoints are quantized)"
    if enable_unified_memory:
        return "--enable-unified-memory (the state pool is a strided arena view)"
    if disaggregation_mode != "null":
        return "PD disaggregation (the hand-off transfers the state mid-chunk)"
    if enable_hierarchical_cache:
        return "--enable-hierarchical-cache (host copies drop the open chunk)"
    if dcp_enabled:
        return "decode context parallel (not validated)"
    if enable_mixed_chunk:
        return "--enable-mixed-chunk (decode tokens ride in an extend batch)"
    if enable_two_batch_overlap:
        return "--enable-two-batch-overlap"
    if enable_nvidia_megakernel:
        return "--enable-nvidia-megakernel (it runs its own KDA decode on the state)"
    if enable_streaming_session:
        return "--enable-streaming-session (a turn resumes the slot inside a chunk)"
    if prefill_cuda_graph_backend == CudaGraphBackend.FULL:
        return "full prefill CUDA graphs (the prefill plans on the host)"
    if radix_cache_enabled:
        if not enable_mamba_extra_buffer:
            return (
                "--mamba-radix-cache-strategy no_buffer, which caches a slot at "
                "any token; use extra_buffer or --disable-radix-cache"
            )
        for name, value in (
            ("--mamba-track-interval", mamba_track_interval),
            ("the mamba cache chunk size", cache_chunk_size),
        ):
            if value % KDA_BATCH_INVARIANT_CHUNK:
                return f"{name}={value}, which must be a multiple of {KDA_BATCH_INVARIANT_CHUNK}"
    return None


def kda_gate_lower_bound(model_config: ModelConfig) -> Optional[float]:
    """The safe-gate lower bound of a KDA model, None for a softplus gate or a
    config that is not a KDA model."""
    from sglang.srt.configs.bailing_hybrid import BailingHybridConfig
    from sglang.srt.configs.hybrid_arch import hybrid_kda_config

    config = hybrid_kda_config(model_config)
    if config is None:
        return None
    if isinstance(config, BailingHybridConfig):
        return config.kda_lower_bound
    return config.linear_attn_config.get("gate_lower_bound")


def create_kda_batch_invariant_kernel(
    *,
    model_config: ModelConfig,
    state_chunk_size: int,
    temporal: torch.Tensor,
    pool_layers: list[int],
    enable_unified_memory: bool,
) -> FlashInferKDABatchInvariantKernel:
    """Validate the configuration and the model, then build the kernel with
    every KDA layer's state pool registered (outside any CUDA-graph capture)."""
    mamba = get_exec().mamba
    refusal = kda_batch_invariant_refusal(
        speculative_algorithm=get_spec().speculative_algorithm,
        enable_linear_replayssm=mamba.enable_linear_replayssm
        or mamba.enable_linear_replayssm_spec,
        enable_int8_mamba_checkpoint=mamba.enable_int8_mamba_checkpoint,
        enable_unified_memory=enable_unified_memory,
        disaggregation_mode=get_disagg().disaggregation_mode,
        enable_hierarchical_cache=get_memory().enable_hierarchical_cache,
        dcp_enabled=get_parallel().dcp_enabled,
        enable_mixed_chunk=get_schedule().enable_mixed_chunk,
        enable_two_batch_overlap=get_exec().overlap.enable_two_batch_overlap,
        enable_nvidia_megakernel=get_exec().kernel.enable_nvidia_megakernel,
        enable_streaming_session=get_serving().enable_streaming_session,
        prefill_cuda_graph_backend=get_exec().graph.cuda_graph_config.prefill.backend,
        radix_cache_enabled=not get_memory().disable_radix_cache,
        enable_mamba_extra_buffer=mamba.enable_mamba_extra_buffer,
        mamba_track_interval=mamba.mamba_track_interval,
        cache_chunk_size=mamba_cache_chunk_size(),
    )
    if refusal is None:
        from sglang.srt.configs.hybrid_arch import hybrid_kda_config

        _, _, num_heads, head_v_dim, head_k_dim = temporal.shape
        # A mock runner without a KDA config leaves the gate to the kernel's own check.
        if hybrid_kda_config(model_config) is not None:
            refusal = kda_batch_invariant_unsupported_reason(
                # KDA models keep equal key and value head counts.
                num_k_heads=num_heads,
                num_v_heads=num_heads,
                head_k_dim=head_k_dim,
                head_v_dim=head_v_dim,
                lower_bound=kda_gate_lower_bound(model_config),
                state_dtype=temporal.dtype,
            )
    if refusal is not None:
        raise ValueError(
            f"--enable-linear-attn-batch-invariant does not support {refusal}"
        )
    kernel = FlashInferKDABatchInvariantKernel(state_chunk_size=state_chunk_size)
    for pool_layer in pool_layers:
        kernel.register_state_pool(temporal[pool_layer])
    rank0_log(
        "KDA batch-invariant mode: FlashInfer class "
        f"{kernel.fingerprint} serves prefill and decode"
    )
    return kernel
