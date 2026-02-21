# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import (
    BLOCK_SIZE,
    create_scale_inv_for_weight,
    dequantize_from_fp8,
)
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin
from nemo_automodel.components.moe.state_dict_utils import (
    create_dtensor_from_local,
    get_expert_range_for_rank_from_mesh,
    get_submesh,
    should_load_expert_for_rank,
)

logger = logging.getLogger(__name__)

NON_QUANTIZED_KEY_PATTERNS = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "norm.weight",
    "lm_head.weight",
    "embed_tokens.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "block_sparse_moe.gate.weight",
    "mlp.gate.weight",
]


def should_quantize_key(key: str) -> bool:
    if not key.endswith(".weight"):
        return False
    return not any(pattern in key for pattern in NON_QUANTIZED_KEY_PATTERNS)


class MiniMaxM2StateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Convert between MiniMax-M2.1 HF checkpoints and native grouped-expert format."""

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True

    @property
    def _expert_path_segment(self) -> str:
        return "mlp.experts"

    def _dequantize(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        scale_inv_keys = []
        for key, weight in state_dict.items():
            if key.endswith(".weight") and key + "_scale_inv" in state_dict:
                scale_inv = state_dict[key + "_scale_inv"]
                state_dict[key] = dequantize_from_fp8(weight, scale_inv, dtype=self.dtype, name=key)
                scale_inv_keys.append(key + "_scale_inv")

        for key in scale_inv_keys:
            state_dict.pop(key, None)

        return state_dict

    def _has_fp8_experts(self, hf_state_dict: dict[str, Any]) -> bool:
        """Check if HF state dict contains FP8 expert weights."""
        for key, value in hf_state_dict.items():
            if ".block_sparse_moe.experts." in key and key.endswith(".weight"):
                if hasattr(value, "dtype") and value.dtype == torch.float8_e4m3fn:
                    return True
        return False

    def _from_hf_native_fp8(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
    ) -> dict[str, Any]:
        """Convert HF checkpoint to native format preserving FP8 expert weights.

        Expert weights stay as float8_e4m3fn with block-wise scale_inv tensors
        transposed and merged alongside. Non-expert FP8 weights are dequantized
        to self.dtype. Attention weights (small) are always dequantized.

        Requires dim and moe_inter_dim to be multiples of BLOCK_SIZE (128)
        for correct block boundary alignment during transpose and concat.
        """
        n_experts = self.moe_config.n_routed_experts
        dim = self.moe_config.dim
        inter_dim = self.moe_config.moe_inter_dim

        assert dim % BLOCK_SIZE == 0, f"dim={dim} must be divisible by BLOCK_SIZE={BLOCK_SIZE}"
        assert inter_dim % BLOCK_SIZE == 0, f"moe_inter_dim={inter_dim} must be divisible by BLOCK_SIZE={BLOCK_SIZE}"

        # Determine EP range
        if device_mesh is not None:
            start_expert, end_expert = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            expected_experts = end_expert - start_expert
            rank = (
                get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )
        else:
            expected_experts = n_experts
            rank = None

        state_dict: dict[str, Any] = {}

        # Identify expert weight keys and their scale_inv
        expert_re = re.compile(
            r"(?P<pre>.+)\.block_sparse_moe\.experts\.(?P<eid>\d+)\.(?P<proj>w[123])\.weight$"
        )
        expert_weight_keys: set[str] = set()
        expert_scale_keys: set[str] = set()

        for key in hf_state_dict:
            if expert_re.match(key):
                expert_weight_keys.add(key)
                scale_key = key + "_scale_inv"
                if scale_key in hf_state_dict:
                    expert_scale_keys.add(scale_key)

        # Process non-expert keys
        for key, value in hf_state_dict.items():
            if key in expert_weight_keys or key in expert_scale_keys:
                continue
            if key.endswith("_scale_inv"):
                continue  # consumed by non-expert dequantization below

            # Dequantize non-expert FP8 weights
            if key.endswith(".weight") and key + "_scale_inv" in hf_state_dict:
                scale_inv = hf_state_dict[key + "_scale_inv"]
                value = dequantize_from_fp8(value, scale_inv, dtype=self.dtype, name=key)

            native_key = self._hf_key_to_native(key)
            state_dict[native_key] = value

        # Accumulate expert data by layer: {layer_base: {expert_id: {proj: (weight, scale)}}}
        expert_data: dict[str, dict[int, dict[str, tuple]]] = {}

        for key in expert_weight_keys:
            m = expert_re.match(key)
            pre = m.group("pre")  # e.g., "model.layers.0"
            eid = int(m.group("eid"))
            proj = m.group("proj")  # w1, w2, or w3

            if not should_load_expert_for_rank(eid, device_mesh, n_experts):
                continue

            weight = hf_state_dict[key]
            scale = hf_state_dict.get(key + "_scale_inv")

            if pre not in expert_data:
                expert_data[pre] = {}
            if eid not in expert_data[pre]:
                expert_data[pre][eid] = {}
            expert_data[pre][eid][proj] = (weight, scale)

        # Merge expert FP8 weights per layer
        for pre, experts in expert_data.items():
            if len(experts) < expected_experts:
                logger.warning(f"Layer {pre}: only {len(experts)}/{expected_experts} experts, skipping FP8 merge")
                continue

            gate_up_list = []
            gate_up_scale_list = []
            down_list = []
            down_scale_list = []

            for eid in sorted(experts.keys()):
                expert = experts[eid]

                # w1 = gate_proj [inter_dim, dim], w3 = up_proj [inter_dim, dim]
                gate_w, gate_s = expert["w1"]
                up_w, up_s = expert["w3"]
                down_w, down_s = expert["w2"]

                # Transpose FP8 weights: [inter_dim, dim] -> [dim, inter_dim]
                gate_t = gate_w.t().contiguous()
                up_t = up_w.t().contiguous()
                # Concat gate + up along last dim: [dim, 2*inter_dim]
                gate_up_list.append(torch.cat([gate_t, up_t], dim=-1))

                # Transpose and concat scale_inv
                if gate_s is not None and up_s is not None:
                    gate_up_scale_list.append(
                        torch.cat([gate_s.t().contiguous(), up_s.t().contiguous()], dim=1)
                    )
                else:
                    gate_up_scale_list.append(
                        torch.ones(dim // BLOCK_SIZE, 2 * (inter_dim // BLOCK_SIZE), dtype=torch.float32)
                    )

                # Down: [dim, inter_dim] -> [inter_dim, dim]
                down_list.append(down_w.t().contiguous())

                if down_s is not None:
                    down_scale_list.append(down_s.t().contiguous())
                else:
                    down_scale_list.append(
                        torch.ones(inter_dim // BLOCK_SIZE, dim // BLOCK_SIZE, dtype=torch.float32)
                    )

            # Stack across experts
            gate_up_stacked = torch.stack(gate_up_list, dim=0)  # [n, dim, 2*inter_dim] FP8
            gate_up_scale_stacked = torch.stack(gate_up_scale_list, dim=0)  # [n, dim/128, 2*(inter_dim/128)]
            down_stacked = torch.stack(down_list, dim=0)  # [n, inter_dim, dim] FP8
            down_scale_stacked = torch.stack(down_scale_list, dim=0)  # [n, inter_dim/128, dim/128]

            # Wrap weights as DTensors for EP (scale_inv stays replicated as buffers)
            gate_up_stacked = create_dtensor_from_local(gate_up_stacked, device_mesh, rank)
            down_stacked = create_dtensor_from_local(down_stacked, device_mesh, rank)

            native_base = f"{pre}.mlp.experts"
            state_dict[f"{native_base}.gate_and_up_projs"] = gate_up_stacked
            state_dict[f"{native_base}.gate_and_up_scale_inv"] = gate_up_scale_stacked
            state_dict[f"{native_base}.down_projs"] = down_stacked
            state_dict[f"{native_base}.down_scale_inv"] = down_scale_stacked

        return state_dict

    def _convert_fp8_expert_to_hf(
        self,
        fqn: str,
        tensor: torch.Tensor,
        fp8_scale_inv: dict[str, torch.Tensor],
    ) -> Optional[list[tuple[str, torch.Tensor]]]:
        """Split FP8 expert tensor to per-expert HF format with scale_inv.

        Returns HF-format keys directly (block_sparse_moe.experts.{E}.w{1,2,3}).
        """
        n_experts = self.moe_config.n_routed_experts
        inter_dim = self.moe_config.moe_inter_dim
        hf_prefix = self._hf_prefix
        expert_segment = self._expert_path_segment

        if f".{expert_segment}.gate_and_up_projs" in fqn and fqn.endswith(".gate_and_up_projs"):
            layer_num = re.search(r"layers\.(\d+)", fqn).group(1)

            scale_key = fqn.replace(".gate_and_up_projs", ".gate_and_up_scale_inv")
            scale_inv = fp8_scale_inv.get(scale_key)

            splits = self._split_experts_weights(tensor, n_experts)
            expert_ids = self._last_expert_ids

            result = []
            for i, w in enumerate(splits):
                eid = expert_ids[i]
                # w: [dim, 2*inter_dim] FP8 -> split gate [dim, inter_dim] + up [dim, inter_dim]
                w_gate = w[:, :inter_dim].t().contiguous()  # [inter_dim, dim] HF format
                w_up = w[:, inter_dim:].t().contiguous()

                base = f"{hf_prefix}layers.{layer_num}.block_sparse_moe.experts.{eid}"
                result.append((f"{base}.w1.weight", w_gate))
                result.append((f"{base}.w3.weight", w_up))

                if scale_inv is not None:
                    s = scale_inv[eid]
                    s_gate = s[:, : inter_dim // BLOCK_SIZE].t().contiguous()
                    s_up = s[:, inter_dim // BLOCK_SIZE :].t().contiguous()
                    result.append((f"{base}.w1.weight_scale_inv", s_gate))
                    result.append((f"{base}.w3.weight_scale_inv", s_up))

            return result

        elif (
            f".{expert_segment}.down_projs" in fqn
            and fqn.endswith(".down_projs")
            and tensor.ndim == 3
        ):
            layer_num = re.search(r"layers\.(\d+)", fqn).group(1)

            scale_key = fqn.replace(".down_projs", ".down_scale_inv")
            scale_inv = fp8_scale_inv.get(scale_key)

            splits = self._split_experts_weights(tensor, n_experts)
            expert_ids = self._last_expert_ids

            result = []
            for i, w in enumerate(splits):
                eid = expert_ids[i]
                # w: [inter_dim, dim] FP8 -> transpose to [dim, inter_dim] HF format
                w_down = w.t().contiguous()

                base = f"{hf_prefix}layers.{layer_num}.block_sparse_moe.experts.{eid}"
                result.append((f"{base}.w2.weight", w_down))

                if scale_inv is not None:
                    s = scale_inv[eid]
                    s_down = s.t().contiguous()
                    result.append((f"{base}.w2.weight_scale_inv", s_down))

            return result

        return None

    def _hf_key_to_native(self, key: str) -> str:
        key = key.replace(".block_sparse_moe.gate.weight", ".mlp.gate.weight")
        key = key.replace(".block_sparse_moe.e_score_correction_bias", ".mlp.gate.e_score_correction_bias")
        key = re.sub(r"\.block_sparse_moe\.experts\.(\d+)\.w1\.weight$", r".mlp.experts.\1.gate_proj.weight", key)
        key = re.sub(r"\.block_sparse_moe\.experts\.(\d+)\.w3\.weight$", r".mlp.experts.\1.up_proj.weight", key)
        key = re.sub(r"\.block_sparse_moe\.experts\.(\d+)\.w2\.weight$", r".mlp.experts.\1.down_proj.weight", key)
        return key

    def _native_key_to_hf(self, key: str) -> str:
        key = re.sub(r"\.mlp\.experts\.(\d+)\.gate_proj\.weight$", r".block_sparse_moe.experts.\1.w1.weight", key)
        key = re.sub(r"\.mlp\.experts\.(\d+)\.up_proj\.weight$", r".block_sparse_moe.experts.\1.w3.weight", key)
        key = re.sub(r"\.mlp\.experts\.(\d+)\.down_proj\.weight$", r".block_sparse_moe.experts.\1.w2.weight", key)
        key = key.replace(".mlp.gate.weight", ".block_sparse_moe.gate.weight")
        key = key.replace(".mlp.gate.e_score_correction_bias", ".block_sparse_moe.e_score_correction_bias")
        return key

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        hf_state_dict = {}

        # Collect FP8 scale_inv buffers for expert conversion
        fp8_scale_inv: dict[str, torch.Tensor] = {}
        if self.backend.native_fp8_experts:
            for fqn, tensor in state_dict.items():
                if fqn.endswith(".gate_and_up_scale_inv") or fqn.endswith(".down_scale_inv"):
                    fp8_scale_inv[fqn] = tensor

        for fqn, tensor in state_dict.items():
            # Skip scale_inv buffers (consumed by FP8 expert conversion)
            if fqn in fp8_scale_inv:
                continue

            converted_tensors = self.convert_single_tensor_to_hf(
                fqn,
                tensor,
                exclude_key_regex=exclude_key_regex,
                quantization=quantization,
                fp8_scale_inv=fp8_scale_inv,
                **kwargs,
            )
            for key, value in converted_tensors:
                hf_state_dict[key] = value

        return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        # Detect model prefix from key layout.
        for key in hf_state_dict.keys():
            if ".block_sparse_moe.experts." in key and key.endswith(".weight"):
                self._uses_model_prefix = key.startswith("model.")
                break

        if self.backend.native_fp8_experts and self._has_fp8_experts(hf_state_dict):
            return self._from_hf_native_fp8(dict(hf_state_dict), device_mesh)

        dequantized = self._dequantize(dict(hf_state_dict))
        remapped = {self._hf_key_to_native(k): v for k, v in dequantized.items()}
        return self._from_hf_w_merged_experts(remapped, device_mesh)

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        quantization = kwargs.get("quantization", False)
        exclude_key_regex = kwargs.get("exclude_key_regex", None)
        fp8_scale_inv = kwargs.get("fp8_scale_inv", {})

        # FP8 expert weights: output HF keys directly (includes weight + scale_inv)
        if (
            self.backend.native_fp8_experts
            and hasattr(tensor, "dtype")
            and tensor.dtype == torch.float8_e4m3fn
        ):
            fp8_result = self._convert_fp8_expert_to_hf(fqn, tensor, fp8_scale_inv)
            if fp8_result is not None:
                if exclude_key_regex:
                    fp8_result = [(k, v) for k, v in fp8_result if not re.match(exclude_key_regex, k)]
                return fp8_result

        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if expert_result is not None:
            result = [(self._native_key_to_hf(k), v) for k, v in expert_result]
        else:
            result = [(self._native_key_to_hf(fqn), tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        if quantization:
            quantized_result = []
            for key, value in result:
                if should_quantize_key(key):
                    value = value.to(dtype=torch.float8_e4m3fn)
                    weight_scale_inv = create_scale_inv_for_weight(value, block_size=BLOCK_SIZE)
                    quantized_result.append((key, value))
                    quantized_result.append((key + "_scale_inv", weight_scale_inv))
                else:
                    quantized_result.append((key, value))
            return quantized_result

        return result
