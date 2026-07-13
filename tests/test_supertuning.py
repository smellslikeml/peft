# Copyright 2025-present the HuggingFace Inc. team.
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

"""
Tests for Super-Tuning sparse PEFT method.

Paper: https://arxiv.org/abs/2607.09287
"""

import os

import pytest
import torch
from torch import nn

from peft import PeftModel, SupertuningConfig, get_peft_model


class MLP(nn.Module):
    """Simple MLP for testing."""

    def __init__(self, bias=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.lin1 = nn.Linear(20, 40, bias=bias)
        self.lin2 = nn.Linear(40, 30, bias=bias)
        self.lin3 = nn.Linear(30, 10, bias=bias)
        self.sm = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        X = self.lin0(X)
        X = self.relu(X)
        X = self.lin1(X)
        X = self.relu(X)
        X = self.lin2(X)
        X = self.relu(X)
        X = self.lin3(X)
        X = self.sm(X)
        return X


class TestSupertuning:
    """Test cases for Super-Tuning."""

    @pytest.fixture
    def mlp(self):
        torch.manual_seed(0)
        model = MLP()
        return model

    def test_mlp_single_adapter_super(self, mlp):
        """Test basic Super adapter creation and shapes."""
        config = SupertuningConfig(
            sparsity_ratio=0.5,
            regularization_method="magnitude",
            target_modules=["lin1", "lin2"],
        )
        peft_model = get_peft_model(mlp, config)

        # Check that layers are wrapped
        assert hasattr(peft_model.base_model.model.lin1, "sparse_weight")
        assert hasattr(peft_model.base_model.model.lin2, "sparse_weight")

        # Check sparse weight sizes
        base_weight1_size = peft_model.base_model.model.lin1.base_layer.weight.shape
        base_weight2_size = peft_model.base_model.model.lin2.base_layer.weight.shape

        # Check masks have correct shape
        mask1 = peft_model.base_model.model.lin1.mask["default"]
        mask2 = peft_model.base_model.model.lin2.mask["default"]

        assert mask1.shape == base_weight1_size
        assert mask2.shape == base_weight2_size

        # Check sparsity ratio is approximately correct
        actual_sparsity1 = (mask1 == 0).float().mean().item()
        actual_sparsity2 = (mask2 == 0).float().mean().item()

        assert 0.45 <= actual_sparsity1 <= 0.55  # Allow some tolerance
        assert 0.45 <= actual_sparsity2 <= 0.55

    def test_mlp_forward_pass(self, mlp):
        """Test forward pass with Super adapter."""
        config = SupertuningConfig(
            sparsity_ratio=0.5,
            regularization_method="magnitude",
            target_modules=["lin1", "lin2"],
        )
        peft_model = get_peft_model(mlp, config)

        input_tensor = torch.randn(5, 10)
        output = peft_model(input_tensor)

        # Check output shape
        assert output.shape == (5, 10)
        assert not torch.isnan(output).any()

    def test_supra_hybrid_adapter(self, mlp):
        """Test Supra hybrid adapter with LoRA components."""
        config = SupertuningConfig(
            sparsity_ratio=0.5,
            regularization_method="magnitude",
            adapter_type="supra",
            lora_rank=8,
            lora_alpha=16,
            target_modules=["lin1", "lin2"],
        )
        peft_model = get_peft_model(mlp, config)

        # Check that both sparse and LoRA components exist
        assert hasattr(peft_model.base_model.model.lin1, "sparse_weight")
        assert hasattr(peft_model.base_model.model.lin1, "lora_A")
        assert hasattr(peft_model.base_model.model.lin1, "lora_B")

        # Check LoRA shapes
        lora_A = peft_model.base_model.model.lin1.lora_A["default"]
        lora_B = peft_model.base_model.model.lin1.lora_B["default"]

        assert lora_A.shape == (8, 20)  # (rank, in_features)
        assert lora_B.shape == (40, 8)  # (out_features, rank)

    def test_wanda_regularization(self, mlp):
        """Test Wanda-style activation-aware regularization."""
        config = SupertuningConfig(
            sparsity_ratio=0.5,
            regularization_method="wanda",
            num_calibration_samples=16,
            target_modules=["lin1", "lin2"],
        )
        peft_model = get_peft_model(mlp, config)

        # Should create masks with calibration
        assert hasattr(peft_model.base_model.model.lin1, "mask")
        assert hasattr(peft_model.base_model.model.lin2, "mask")

        # Forward pass should work
        input_tensor = torch.randn(5, 10)
        output = peft_model(input_tensor)

        assert output.shape == (5, 10)
        assert not torch.isnan(output).any()

    def test_multiple_adapters(self, mlp):
        """Test multiple Super adapters."""
        config1 = SupertuningConfig(
            sparsity_ratio=0.5,
            regularization_method="magnitude",
            target_modules=["lin1", "lin2"],
        )
        peft_model = get_peft_model(mlp, config1, adapter_name="adapter1")

        config2 = SupertuningConfig(
            sparsity_ratio=0.3,
            regularization_method="magnitude",
            target_modules=["lin1", "lin2", "lin3"],
        )
        peft_model.add_adapter("adapter2", config2)

        # Check both adapters exist
        assert "adapter1" in peft_model.base_model.model.lin1.sparse_weight
        assert "adapter2" in peft_model.base_model.model.lin1.sparse_weight

        # Check different sparsity ratios
        mask1 = peft_model.base_model.model.lin1.mask["adapter1"]
        mask2 = peft_model.base_model.model.lin1.mask["adapter2"]

        sparsity1 = (mask1 == 0).float().mean().item()
        sparsity2 = (mask2 == 0).float().mean().item()

        assert 0.45 <= sparsity1 <= 0.55  # ~0.5
        assert 0.25 <= sparsity2 <= 0.35  # ~0.3

    def test_merge_unmerge(self, mlp):
        """Test merging and unmerging adapters."""
        config = SupertuningConfig(
            sparsity_ratio=0.5,
            regularization_method="magnitude",
            target_modules=["lin1"],
        )
        peft_model = get_peft_model(mlp, config)

        # Get output before merge
        input_tensor = torch.randn(5, 10)
        output_before = peft_model(input_tensor)

        # Merge adapter
        peft_model.base_model.model.lin1.merge()
        output_merged = peft_model(input_tensor)

        # Outputs should be similar (not exact due to numerical precision)
        assert torch.allclose(output_before, output_merged, atol=1e-5, rtol=1e-5)

        # Unmerge
        peft_model.base_model.model.lin1.unmerge()
        output_unmerged = peft_model(input_tensor)

        assert torch.allclose(output_before, output_unmerged, atol=1e-5, rtol=1e-5)

    def test_save_load(self, mlp, tmp_path):
        """Test saving and loading Super adapters."""
        config = SupertuningConfig(
            sparsity_ratio=0.5,
            regularization_method="magnitude",
            target_modules=["lin1", "lin2"],
        )
        peft_model = get_peft_model(mlp, config, adapter_name="default")

        input_tensor = torch.randn(5, 10)
        output_before = peft_model(input_tensor)

        # Save adapter
        save_path = os.path.join(tmp_path, "supertuning")
        peft_model.save_pretrained(save_path)
        assert os.path.exists(os.path.join(save_path, "adapter_config.json"))
        assert os.path.exists(os.path.join(save_path, "adapter_model.bin"))

        # Load adapter
        del peft_model
        torch.manual_seed(0)
        mlp = MLP()
        peft_model = PeftModel.from_pretrained(mlp, save_path)

        output_after = peft_model(input_tensor)

        assert torch.allclose(output_before, output_after)

    def test_trainable_params_count(self, mlp):
        """Test that trainable parameters match sparsity ratio."""
        config = SupertuningConfig(
            sparsity_ratio=0.5,
            regularization_method="magnitude",
            target_modules=["lin1", "lin2"],
        )
        peft_model = get_peft_model(mlp, config)

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)

        # Expected: for each layer, (1 - sparsity_ratio) * num_params
        lin1_params = 20 * 40  # 800
        lin2_params = 40 * 30  # 1200
        expected_trainable = int(0.5 * (lin1_params + lin2_params))

        # Allow some tolerance due to rounding
        assert 0.9 * expected_trainable <= trainable_params <= 1.1 * expected_trainable

    def test_init_weights(self, mlp):
        """Test weight initialization options."""
        # Test with init_weights=True (default)
        config1 = SupertuningConfig(
            sparsity_ratio=0.5,
            init_weights=True,
            target_modules=["lin1"],
        )
        peft_model1 = get_peft_model(mlp, config1)

        sparse_weight1 = peft_model1.base_model.model.lin1.sparse_weight["default"]
        assert torch.allclose(sparse_weight1, torch.zeros_like(sparse_weight1))

        # Test with init_weights=False
        config2 = SupertuningConfig(
            sparsity_ratio=0.5,
            init_weights=False,
            target_modules=["lin2"],
        )
        peft_model2 = get_peft_model(mlp, config2)

        sparse_weight2 = peft_model2.base_model.model.lin2.sparse_weight["default"]
        assert not torch.allclose(sparse_weight2, torch.zeros_like(sparse_weight2))
