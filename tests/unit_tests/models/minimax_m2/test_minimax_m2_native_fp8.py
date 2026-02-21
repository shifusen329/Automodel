# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for native FP8 expert support in MiniMax M2 state dict adapter and GroupedExpertsFP8."""

import math
from dataclasses import dataclass

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import (
    BLOCK_SIZE,
    dequantize_from_fp8,
)
from nemo_automodel.components.models.minimax_m2.state_dict_adapter import MiniMaxM2StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsFP8

# Use dimensions that are multiples of BLOCK_SIZE (128) for FP8 block alignment
DIM = 256
INTER_DIM = 128
N_EXPERTS = 4
N_ACTIVATED = 2


@dataclass
class MockConfig:
    hidden_size: int = DIM
    intermediate_size: int = INTER_DIM
    num_local_experts: int = N_EXPERTS
    torch_dtype: str = "bfloat16"


def _make_moe_config(dim=DIM, inter_dim=INTER_DIM, n_experts=N_EXPERTS):
    return MoEConfig(
        dim=dim,
        inter_dim=inter_dim,
        moe_inter_dim=inter_dim,
        n_routed_experts=n_experts,
        n_shared_experts=0,
        n_activated_experts=N_ACTIVATED,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="sigmoid",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        dtype=torch.bfloat16,
        force_e_score_correction_bias=True,
    )


def _make_fp8_backend(**overrides):
    defaults = dict(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch_mm",
        dispatcher="torch",
        native_fp8_experts=True,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
        rope_fusion=False,
    )
    defaults.update(overrides)
    return BackendConfig(**defaults)


def _make_bf16_backend(**overrides):
    defaults = dict(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch_mm",
        dispatcher="torch",
        native_fp8_experts=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
        rope_fusion=False,
    )
    defaults.update(overrides)
    return BackendConfig(**defaults)


def _make_synthetic_fp8_weight(shape, device="cpu"):
    """Create a synthetic FP8 weight with scale_inv."""
    # Create bf16 reference, then quantize to FP8
    bf16_ref = torch.randn(shape, dtype=torch.bfloat16, device=device) * 0.1
    fp8_weight = bf16_ref.to(dtype=torch.float8_e4m3fn)

    # Create block-wise scale_inv
    rows, cols = shape
    scale_rows = math.ceil(rows / BLOCK_SIZE)
    scale_cols = math.ceil(cols / BLOCK_SIZE)
    # Compute actual max-abs scales per block
    scale_inv = torch.ones(scale_rows, scale_cols, dtype=torch.float32, device=device)
    for i in range(scale_rows):
        for j in range(scale_cols):
            r_start = i * BLOCK_SIZE
            r_end = min(r_start + BLOCK_SIZE, rows)
            c_start = j * BLOCK_SIZE
            c_end = min(c_start + BLOCK_SIZE, cols)
            block = bf16_ref[r_start:r_end, c_start:c_end].float()
            amax = block.abs().max()
            if amax > 0:
                # FP8 e4m3fn max is 448.0
                scale = amax / 448.0
                scale_inv[i, j] = scale

    # Re-quantize with proper scaling
    fp8_weight = torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device)
    for i in range(scale_rows):
        for j in range(scale_cols):
            r_start = i * BLOCK_SIZE
            r_end = min(r_start + BLOCK_SIZE, rows)
            c_start = j * BLOCK_SIZE
            c_end = min(c_start + BLOCK_SIZE, cols)
            block = bf16_ref[r_start:r_end, c_start:c_end].float()
            scaled = block / scale_inv[i, j]
            fp8_weight[r_start:r_end, c_start:c_end] = scaled.to(torch.float8_e4m3fn)

    return fp8_weight, scale_inv, bf16_ref


def _make_hf_fp8_state(n_experts, inter_dim, dim, device="cpu"):
    """Create synthetic HF state dict with FP8 expert weights + scale_inv."""
    sd = {}
    bf16_refs = {}
    for e in range(n_experts):
        # gate_proj (w1): [inter_dim, dim]
        fp8_w, scale, ref = _make_synthetic_fp8_weight((inter_dim, dim), device)
        sd[f"model.layers.0.block_sparse_moe.experts.{e}.w1.weight"] = fp8_w
        sd[f"model.layers.0.block_sparse_moe.experts.{e}.w1.weight_scale_inv"] = scale
        bf16_refs[f"w1_{e}"] = ref

        # up_proj (w3): [inter_dim, dim]
        fp8_w, scale, ref = _make_synthetic_fp8_weight((inter_dim, dim), device)
        sd[f"model.layers.0.block_sparse_moe.experts.{e}.w3.weight"] = fp8_w
        sd[f"model.layers.0.block_sparse_moe.experts.{e}.w3.weight_scale_inv"] = scale
        bf16_refs[f"w3_{e}"] = ref

        # down_proj (w2): [dim, inter_dim]
        fp8_w, scale, ref = _make_synthetic_fp8_weight((dim, inter_dim), device)
        sd[f"model.layers.0.block_sparse_moe.experts.{e}.w2.weight"] = fp8_w
        sd[f"model.layers.0.block_sparse_moe.experts.{e}.w2.weight_scale_inv"] = scale
        bf16_refs[f"w2_{e}"] = ref

    # Add a non-expert weight (attention) to test selective dequantization
    fp8_attn, scale_attn, ref_attn = _make_synthetic_fp8_weight((dim, dim), device)
    sd["model.layers.0.self_attn.q_proj.weight"] = fp8_attn
    sd["model.layers.0.self_attn.q_proj.weight_scale_inv"] = scale_attn
    bf16_refs["q_proj"] = ref_attn

    return sd, bf16_refs


class TestFromHFNativeFP8:
    """Test from_hf with native FP8 expert preservation."""

    def test_expert_weights_remain_fp8(self):
        """Expert weights should stay as float8_e4m3fn after from_hf."""
        moe_config = _make_moe_config()
        backend = _make_fp8_backend()
        adapter = MiniMaxM2StateDictAdapter(MockConfig(), moe_config, backend, dtype=torch.bfloat16)

        hf_sd, _ = _make_hf_fp8_state(N_EXPERTS, INTER_DIM, DIM)
        native_sd = adapter.from_hf(hf_sd)

        gate_up = native_sd["model.layers.0.mlp.experts.gate_and_up_projs"]
        down = native_sd["model.layers.0.mlp.experts.down_projs"]

        assert gate_up.dtype == torch.float8_e4m3fn
        assert down.dtype == torch.float8_e4m3fn
        assert gate_up.shape == (N_EXPERTS, DIM, 2 * INTER_DIM)
        assert down.shape == (N_EXPERTS, INTER_DIM, DIM)

    def test_scale_inv_buffers_created(self):
        """Scale_inv buffers should be created alongside FP8 weights."""
        moe_config = _make_moe_config()
        backend = _make_fp8_backend()
        adapter = MiniMaxM2StateDictAdapter(MockConfig(), moe_config, backend, dtype=torch.bfloat16)

        hf_sd, _ = _make_hf_fp8_state(N_EXPERTS, INTER_DIM, DIM)
        native_sd = adapter.from_hf(hf_sd)

        gate_up_scale = native_sd["model.layers.0.mlp.experts.gate_and_up_scale_inv"]
        down_scale = native_sd["model.layers.0.mlp.experts.down_scale_inv"]

        assert gate_up_scale.dtype == torch.float32
        assert down_scale.dtype == torch.float32
        assert gate_up_scale.shape == (N_EXPERTS, DIM // BLOCK_SIZE, 2 * (INTER_DIM // BLOCK_SIZE))
        assert down_scale.shape == (N_EXPERTS, INTER_DIM // BLOCK_SIZE, DIM // BLOCK_SIZE)

    def test_non_expert_weights_dequantized(self):
        """Non-expert FP8 weights (attention) should be dequantized to bf16."""
        moe_config = _make_moe_config()
        backend = _make_fp8_backend()
        adapter = MiniMaxM2StateDictAdapter(MockConfig(), moe_config, backend, dtype=torch.bfloat16)

        hf_sd, _ = _make_hf_fp8_state(N_EXPERTS, INTER_DIM, DIM)
        native_sd = adapter.from_hf(hf_sd)

        q_proj = native_sd["model.layers.0.self_attn.q_proj.weight"]
        assert q_proj.dtype == torch.bfloat16

    def test_scale_merge_correctness(self):
        """Merged FP8 + scale_inv should dequantize to same values as original bf16 reference."""
        moe_config = _make_moe_config()
        backend = _make_fp8_backend()
        adapter = MiniMaxM2StateDictAdapter(MockConfig(), moe_config, backend, dtype=torch.bfloat16)

        hf_sd, _ = _make_hf_fp8_state(N_EXPERTS, INTER_DIM, DIM)
        native_sd = adapter.from_hf(hf_sd)

        gate_up_fp8 = native_sd["model.layers.0.mlp.experts.gate_and_up_projs"]
        gate_up_scale = native_sd["model.layers.0.mlp.experts.gate_and_up_scale_inv"]

        # Dequantize expert 0's gate portion
        expert_0_gate_up = gate_up_fp8[0]  # [dim, 2*inter_dim]
        expert_0_scale = gate_up_scale[0]  # [dim/128, 2*(inter_dim/128)]
        dequantized = dequantize_from_fp8(expert_0_gate_up, expert_0_scale, dtype=torch.bfloat16)

        # Also dequantize the original HF weight directly
        orig_gate_fp8 = hf_sd["model.layers.0.block_sparse_moe.experts.0.w1.weight"]
        orig_gate_scale = hf_sd["model.layers.0.block_sparse_moe.experts.0.w1.weight_scale_inv"]
        orig_dequantized = dequantize_from_fp8(orig_gate_fp8, orig_gate_scale, dtype=torch.bfloat16)

        # The gate portion of dequantized should match orig transposed
        gate_portion = dequantized[:, :INTER_DIM]  # [dim, inter_dim]
        orig_t = orig_dequantized.t()  # [dim, inter_dim]

        torch.testing.assert_close(gate_portion, orig_t, rtol=1e-3, atol=1e-3)


class TestRoundtrip:
    """Test from_hf -> to_hf preserves FP8 weights and scales."""

    def test_roundtrip_preserves_fp8(self):
        """FP8 weights and scale_inv should survive from_hf -> to_hf roundtrip."""
        moe_config = _make_moe_config()
        backend = _make_fp8_backend()
        adapter = MiniMaxM2StateDictAdapter(MockConfig(), moe_config, backend, dtype=torch.bfloat16)

        hf_sd, _ = _make_hf_fp8_state(N_EXPERTS, INTER_DIM, DIM)
        native_sd = adapter.from_hf(hf_sd)
        hf_sd_out = adapter.to_hf(native_sd)

        # Check all expert weights roundtrip
        for e in range(N_EXPERTS):
            for proj in ["w1", "w2", "w3"]:
                key = f"model.layers.0.block_sparse_moe.experts.{e}.{proj}.weight"
                assert key in hf_sd_out, f"Missing key: {key}"
                assert hf_sd_out[key].dtype == torch.float8_e4m3fn, f"{key} should be FP8"

                scale_key = key + "_scale_inv"
                assert scale_key in hf_sd_out, f"Missing scale: {scale_key}"
                assert hf_sd_out[scale_key].dtype == torch.float32

    def test_roundtrip_weight_values(self):
        """FP8 weight values should be identical after roundtrip."""
        moe_config = _make_moe_config()
        backend = _make_fp8_backend()
        adapter = MiniMaxM2StateDictAdapter(MockConfig(), moe_config, backend, dtype=torch.bfloat16)

        hf_sd, _ = _make_hf_fp8_state(N_EXPERTS, INTER_DIM, DIM)
        native_sd = adapter.from_hf(hf_sd)
        hf_sd_out = adapter.to_hf(native_sd)

        for e in range(N_EXPERTS):
            for proj in ["w1", "w2", "w3"]:
                key = f"model.layers.0.block_sparse_moe.experts.{e}.{proj}.weight"
                orig = hf_sd[key]
                roundtripped = hf_sd_out[key]
                # FP8 tensors: compare as float to check bit-exact match
                assert torch.equal(orig.float(), roundtripped.float()), (
                    f"Weight mismatch for {key}"
                )

                scale_key = key + "_scale_inv"
                orig_scale = hf_sd[scale_key]
                roundtripped_scale = hf_sd_out[scale_key]
                torch.testing.assert_close(orig_scale, roundtripped_scale)


class TestGroupedExpertsFP8Forward:
    """Test GroupedExpertsFP8 forward pass produces correct results."""

    def test_forward_grouped_mm(self):
        """GroupedExpertsFP8 with torch_mm should produce output matching bf16 experts."""
        moe_config = _make_moe_config()
        fp8_backend = _make_fp8_backend(experts="torch_mm")
        bf16_backend = _make_bf16_backend(experts="torch_mm")

        fp8_experts = GroupedExpertsFP8(moe_config, backend=fp8_backend)
        bf16_experts = GroupedExperts(moe_config, backend=bf16_backend)

        # Create synthetic weights
        gate_up_shape = (N_EXPERTS, DIM, 2 * INTER_DIM)
        down_shape = (N_EXPERTS, INTER_DIM, DIM)

        bf16_gate_up = torch.randn(gate_up_shape, dtype=torch.bfloat16) * 0.01
        bf16_down = torch.randn(down_shape, dtype=torch.bfloat16) * 0.01

        # Set bf16 experts
        bf16_experts.gate_and_up_projs.data.copy_(bf16_gate_up)
        bf16_experts.down_projs.data.copy_(bf16_down)

        # Set FP8 experts: convert bf16 to FP8 + scale_inv
        fp8_gate_up = bf16_gate_up.to(torch.float8_e4m3fn)
        fp8_down = bf16_down.to(torch.float8_e4m3fn)
        fp8_experts.gate_and_up_projs.data.copy_(fp8_gate_up)
        fp8_experts.down_projs.data.copy_(fp8_down)
        # Use ones for scale_inv (simple case)
        fp8_experts.gate_and_up_scale_inv.fill_(1.0)
        fp8_experts.down_scale_inv.fill_(1.0)

        # Create input
        n_tokens = 8
        x = torch.randn(n_tokens, DIM, dtype=torch.bfloat16)
        token_mask = torch.ones(n_tokens, dtype=torch.bool)
        weights = torch.ones(n_tokens, N_ACTIVATED, dtype=torch.bfloat16) / N_ACTIVATED
        indices = torch.stack([
            torch.arange(n_tokens) % N_EXPERTS,
            (torch.arange(n_tokens) + 1) % N_EXPERTS,
        ], dim=1)

        with torch.no_grad():
            y_fp8 = fp8_experts(x, token_mask, weights, indices)
            y_bf16 = bf16_experts(x, token_mask, weights, indices)

        # FP8 output should be close to bf16 (within FP8 precision tolerance)
        assert y_fp8.shape == y_bf16.shape
        # With unit scale, the only difference is FP8 quantization error
        torch.testing.assert_close(y_fp8.float(), y_bf16.float(), rtol=0.15, atol=0.05)

    def test_forward_loop(self):
        """GroupedExpertsFP8 with per-expert loop should produce valid output."""
        moe_config = _make_moe_config()
        backend = _make_fp8_backend(experts="torch")

        fp8_experts = GroupedExpertsFP8(moe_config, backend=backend)

        # Simple unit-scale weights
        fp8_experts.gate_and_up_projs.data.copy_(
            torch.randn(N_EXPERTS, DIM, 2 * INTER_DIM, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        )
        fp8_experts.down_projs.data.copy_(
            torch.randn(N_EXPERTS, INTER_DIM, DIM, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        )
        fp8_experts.gate_and_up_scale_inv.fill_(1.0)
        fp8_experts.down_scale_inv.fill_(1.0)

        n_tokens = 8
        x = torch.randn(n_tokens, DIM, dtype=torch.bfloat16)
        token_mask = torch.ones(n_tokens, dtype=torch.bool)
        weights = torch.ones(n_tokens, N_ACTIVATED, dtype=torch.bfloat16) / N_ACTIVATED
        indices = torch.stack([
            torch.arange(n_tokens) % N_EXPERTS,
            (torch.arange(n_tokens) + 1) % N_EXPERTS,
        ], dim=1)

        with torch.no_grad():
            y = fp8_experts(x, token_mask, weights, indices)

        assert y.shape == (n_tokens, DIM)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_zero_active_experts(self):
        """GroupedExpertsFP8 handles zero active experts without error."""
        moe_config = _make_moe_config()
        backend = _make_fp8_backend(experts="torch_mm")

        fp8_experts = GroupedExpertsFP8(moe_config, backend=backend)
        fp8_experts.gate_and_up_projs.data.copy_(
            torch.zeros(N_EXPERTS, DIM, 2 * INTER_DIM, dtype=torch.float8_e4m3fn)
        )
        fp8_experts.down_projs.data.copy_(
            torch.zeros(N_EXPERTS, INTER_DIM, DIM, dtype=torch.float8_e4m3fn)
        )
        fp8_experts.gate_and_up_scale_inv.fill_(1.0)
        fp8_experts.down_scale_inv.fill_(1.0)

        n_tokens = 4
        x = torch.randn(n_tokens, DIM, dtype=torch.bfloat16)
        token_mask = torch.zeros(n_tokens, dtype=torch.bool)  # all masked
        weights = torch.ones(n_tokens, N_ACTIVATED, dtype=torch.bfloat16)
        indices = torch.zeros(n_tokens, N_ACTIVATED, dtype=torch.long)

        with torch.no_grad():
            y = fp8_experts(x, token_mask, weights, indices)

        assert y.shape == (n_tokens, DIM)


class TestMemory:
    """Test memory savings with FP8 experts."""

    def test_fp8_parameters_half_memory(self):
        """FP8 expert parameters should use ~50% less memory than bf16."""
        moe_config = _make_moe_config()

        fp8_experts = GroupedExpertsFP8(moe_config, backend=_make_fp8_backend())
        bf16_experts = GroupedExperts(moe_config, backend=_make_bf16_backend())

        fp8_param_bytes = sum(
            p.nelement() * p.element_size()
            for p in fp8_experts.parameters()
        )
        bf16_param_bytes = sum(
            p.nelement() * p.element_size()
            for p in bf16_experts.parameters()
        )

        # FP8 (1 byte) vs bf16 (2 bytes) -> ~50% ratio
        ratio = fp8_param_bytes / bf16_param_bytes
        assert ratio < 0.55, f"FP8/bf16 memory ratio {ratio:.3f} should be ~0.5"
        assert ratio > 0.45, f"FP8/bf16 memory ratio {ratio:.3f} should be ~0.5"

    def test_fp8_weights_frozen(self):
        """FP8 expert weights should have requires_grad=False for LoRA."""
        moe_config = _make_moe_config()
        fp8_experts = GroupedExpertsFP8(moe_config, backend=_make_fp8_backend())

        assert not fp8_experts.gate_and_up_projs.requires_grad
        assert not fp8_experts.down_projs.requires_grad


class TestHasDetection:
    """Test _has_fp8_experts detection."""

    def test_detects_fp8_experts(self):
        moe_config = _make_moe_config()
        backend = _make_fp8_backend()
        adapter = MiniMaxM2StateDictAdapter(MockConfig(), moe_config, backend, dtype=torch.bfloat16)

        hf_sd, _ = _make_hf_fp8_state(N_EXPERTS, INTER_DIM, DIM)
        assert adapter._has_fp8_experts(hf_sd)

    def test_no_fp8_when_bf16(self):
        moe_config = _make_moe_config()
        backend = _make_fp8_backend()
        adapter = MiniMaxM2StateDictAdapter(MockConfig(), moe_config, backend, dtype=torch.bfloat16)

        hf_sd = {}
        for e in range(N_EXPERTS):
            hf_sd[f"model.layers.0.block_sparse_moe.experts.{e}.w1.weight"] = torch.randn(INTER_DIM, DIM)
            hf_sd[f"model.layers.0.block_sparse_moe.experts.{e}.w3.weight"] = torch.randn(INTER_DIM, DIM)
            hf_sd[f"model.layers.0.block_sparse_moe.experts.{e}.w2.weight"] = torch.randn(DIM, INTER_DIM)
        assert not adapter._has_fp8_experts(hf_sd)

    def test_fallback_to_dequantize_when_flag_off(self):
        """When native_fp8_experts=False, FP8 experts should be dequantized."""
        moe_config = _make_moe_config()
        backend = _make_bf16_backend()
        adapter = MiniMaxM2StateDictAdapter(MockConfig(), moe_config, backend, dtype=torch.bfloat16)

        hf_sd, _ = _make_hf_fp8_state(N_EXPERTS, INTER_DIM, DIM)
        native_sd = adapter.from_hf(hf_sd)

        gate_up = native_sd["model.layers.0.mlp.experts.gate_and_up_projs"]
        assert gate_up.dtype == torch.bfloat16, "Should be dequantized when native_fp8_experts=False"
