# Copyright 2024-present the HuggingFace Inc. team.
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

# This test file is for tests specific to AdaMoLE (Adaptive Mixture of LoRA Experts,
# https://arxiv.org/abs/2405.00361).

import pytest
import torch
from torch import nn

from peft import AdaMoLEConfig, AdaMoLEModel, get_peft_model
from peft.mapping import PEFT_TYPE_TO_TUNER_MAPPING
from peft.tuners.adamole.layer import AdaMoLELinear
from peft.utils import PeftType


class MLP(nn.Module):
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


class TestAdaMoLE:
    @pytest.fixture
    def mlp(self):
        torch.manual_seed(0)
        return MLP()

    def test_adamole_creates_n_experts(self, mlp):
        # Wrapping an nn.Linear with num_experts=4 creates exactly 4 (lora_A, lora_B) pairs, one router and one
        # threshold parameter.
        config = AdaMoLEConfig(r=4, num_experts=4, target_modules=["lin1"])
        peft_model = get_peft_model(mlp, config)

        layer = peft_model.base_model.model.lin1
        assert isinstance(layer, AdaMoLELinear)
        assert len(layer.adamole_A["default"]) == 4
        assert len(layer.adamole_B["default"]) == 4
        # single router producing num_experts logits
        router = layer.adamole_router["default"]
        assert router.out_features == 4
        # exactly one scalar threshold parameter
        assert "default" in layer.adamole_threshold
        assert layer.adamole_threshold["default"].numel() == 1

    def test_adamole_threshold_masks_samples(self):
        # Two samples: one routes decisively to an expert (top gate > threshold -> active), the other routes uniformly
        # (top gate < threshold -> its delta contribution is exactly zero).
        torch.manual_seed(0)
        base = nn.Linear(4, 3, bias=False)
        config = AdaMoLEConfig(r=2, num_experts=4, use_threshold=True)
        layer = AdaMoLELinear(base, "default", config)

        # Give every expert a non-zero delta so a surviving token is clearly non-zero.
        for e in range(4):
            layer.adamole_A["default"][e].weight.data.fill_(1.0)
            layer.adamole_B["default"][e].weight.data.fill_(1.0)

        # Router whose logits depend on the input, so the two samples gate differently. router(x) = x @ weight.T.
        router = layer.adamole_router["default"]
        router.weight.data.zero_()
        if router.bias is not None:
            router.bias.data.zero_()
        router.weight.data[0, 0] = 10.0  # sample 0 -> expert 0 strongly preferred (top gate ~1.0)

        # High threshold: only the near-deterministic sample survives.
        layer.adamole_threshold["default"].data.fill_(0.95)

        x = torch.zeros(2, 4)
        x[0, 0] = 1.0  # decisive routing
        x[1, 1] = 1.0  # uniform routing (top gate = 0.25)

        delta = layer.compute_delta("default", x)
        assert torch.all(delta[1] == 0.0)  # below-threshold sample is exactly zeroed
        assert torch.any(delta[0] != 0.0)  # above-threshold sample contributes

    def test_adamole_backward_updates_router_and_threshold(self):
        # A forward + backward pass must produce gradients on the router weights and on the (straight-through)
        # learnable threshold parameter.
        torch.manual_seed(0)
        base = nn.Linear(4, 3, bias=False)
        config = AdaMoLEConfig(r=2, num_experts=4, use_threshold=True)
        layer = AdaMoLELinear(base, "default", config)

        # Make the experts contribute a clear signal.
        for e in range(4):
            layer.adamole_B["default"][e].weight.data.fill_(1.0)

        x = torch.randn(3, 4)
        layer(x).sum().backward()

        router_weight = layer.adamole_router["default"].weight
        assert router_weight.grad is not None
        assert torch.any(router_weight.grad != 0)

        threshold = layer.adamole_threshold["default"]
        assert threshold.grad is not None
        assert torch.any(threshold.grad != 0)

    def test_adamole_registered_peft_type(self):
        # AdaMoLE is registered and get_peft_model dispatches to AdaMoLEModel, preserving the base output shape.
        assert PeftType.ADAMOLE in PEFT_TYPE_TO_TUNER_MAPPING
        assert PEFT_TYPE_TO_TUNER_MAPPING[PeftType.ADAMOLE] is AdaMoLEModel

        torch.manual_seed(0)
        base = MLP()
        torch.manual_seed(0)
        model_for_peft = MLP()

        config = AdaMoLEConfig(r=4, num_experts=2, target_modules=["lin1", "lin2"])
        peft_model = get_peft_model(model_for_peft, config)
        assert isinstance(peft_model.base_model, AdaMoLEModel)

        x = torch.randn(5, 10)
        with torch.no_grad():
            base_out = base(x)
            peft_out = peft_model(x)
        assert base_out.shape == peft_out.shape
