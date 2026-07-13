# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from peft import PeftModel, SupertuningConfig, get_peft_model
from peft.tuners.supertuning.layer import Linear as SupertuningLinear
from peft.utils import infer_device


class TestSupertuning:
    device = infer_device()

    def test_supertuning_config(self):
        """Test that SupertuningConfig is properly configured."""
        config = SupertuningConfig(
            peft_type="SUPERTUNING",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
            sparsity=0.5,
            scoring_method="wanda",
        )
        assert config.peft_type.value == "SUPERTUNING"
        assert config.sparsity == 0.5
        assert config.scoring_method == "wanda"

    def test_supertuning_config_validation(self):
        """Test that SupertuningConfig validates its parameters."""
        # Invalid sparsity
        with pytest.raises(ValueError, match="sparsity must be between"):
            SupertuningConfig(sparsity=1.5)

        # Invalid scoring method
        with pytest.raises(ValueError, match="scoring_method must be"):
            SupertuningConfig(scoring_method="invalid")

    def test_supertuning_identity_init(self):
        """With zero-initialized values, the sparse update is the identity and does not change the output."""
        torch.manual_seed(0)

        inputs = torch.arange(10).view(-1, 1).to(self.device)
        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        model.eval()
        output_base = model(inputs).logits

        config = SupertuningConfig(
            target_modules=["q_proj", "v_proj"],
            sparsity=0.5,
            init_weights=True,
        )
        model = get_peft_model(model, config)
        model.eval()
        output_peft = model(inputs).logits

        # values start at zero, so the effective weight equals the base weight exactly
        assert torch.allclose(output_base, output_peft, atol=1e-6, rtol=1e-6)

    def test_supertuning_state_dict(self, tmp_path):
        """Test that Supertuning saves only the compact (indices, values) support and round-trips."""
        torch.manual_seed(0)

        inputs = torch.arange(10).view(-1, 1).to(self.device)
        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)

        config = SupertuningConfig(
            target_modules=["q_proj", "v_proj"],
            sparsity=0.5,
            # non-identity update so the round-trip actually exercises the trained values
            init_weights=False,
        )
        model = get_peft_model(model, config)
        model.eval()
        output_peft = model(inputs).logits

        model.save_pretrained(tmp_path)
        del model

        # the adapter checkpoint stores the compact sparse support: 1-D values and indices sized to the trainable
        # count, and nothing resembling a full dense mask.
        state_dict = load_file(tmp_path / "adapter_model.safetensors")
        assert any("supertuning_values" in key for key in state_dict)
        assert any("supertuning_indices" in key for key in state_dict)
        assert not any("sparse_mask" in key for key in state_dict)
        values_keys = [key for key in state_dict if "supertuning_values" in key]
        assert values_keys
        for key in values_keys:
            values = state_dict[key]
            indices = state_dict[key.replace("supertuning_values", "supertuning_indices")]
            # support is a 1-D pair sized to the trainable count (sparse storage, not the full weight numel)
            assert values.ndim == 1
            assert indices.shape == values.shape

        atol, rtol = 1e-5, 1e-8
        # the trained sparse values survive the round-trip
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        model = PeftModel.from_pretrained(model, tmp_path)
        output_loaded = model(inputs).logits
        assert torch.allclose(output_peft, output_loaded, atol=atol, rtol=rtol)

    def test_supertuning_get_peft_model(self):
        """Test that get_peft_model works with Supertuning."""
        torch.manual_seed(0)

        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)

        config = SupertuningConfig(
            target_modules=["q_proj", "v_proj"],
            sparsity=0.5,
        )
        model = get_peft_model(model, config)

        # Check that the model has the adapter
        assert hasattr(model, "peft_config")
        assert "default" in model.peft_config
        assert model.peft_config["default"].peft_type.value == "SUPERTUNING"

    def test_supertuning_trainable_parameters_count(self):
        """Test that trainable parameter count is computed correctly."""
        torch.manual_seed(0)

        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)

        config = SupertuningConfig(
            target_modules=["q_proj", "v_proj"],
            sparsity=0.5,
        )
        model = get_peft_model(model, config)

        # Get trainable parameter count
        if hasattr(model, "get_trainable_parameters_count"):
            counts = model.get_trainable_parameters_count()
            assert "total_parameters" in counts
            assert "trainable_parameters" in counts
            assert "sparsity" in counts
            # Check that sparsity is close to the configured value
            assert 0.4 <= counts["sparsity"] <= 0.6  # Allow some tolerance

    def test_supertuning_magnitude_scoring(self):
        """Test that magnitude-only scoring works."""
        torch.manual_seed(0)

        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)

        config = SupertuningConfig(
            target_modules=["q_proj", "v_proj"],
            sparsity=0.5,
            scoring_method="magnitude",
        )
        model = get_peft_model(model, config)

        assert model.peft_config["default"].scoring_method == "magnitude"

    def test_supertuning_multiple_adapters(self):
        """Test that multiple adapters can be added."""
        torch.manual_seed(0)

        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)

        config1 = SupertuningConfig(
            target_modules=["q_proj", "v_proj"],
            sparsity=0.5,
        )
        model = get_peft_model(model, config1, adapter_name="adapter1")

        config2 = SupertuningConfig(
            target_modules=["q_proj", "v_proj"],
            sparsity=0.3,
        )
        model.add_adapter("adapter2", config2)

        assert "adapter1" in model.peft_config
        assert "adapter2" in model.peft_config
        assert model.peft_config["adapter1"].sparsity == 0.5
        assert model.peft_config["adapter2"].sparsity == 0.3

    def _prepare_trainable_model(self, **config_kwargs):
        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        kwargs = {"target_modules": ["q_proj", "v_proj"], "sparsity": 0.5, "scoring_method": "magnitude"}
        kwargs.update(config_kwargs)
        config = SupertuningConfig(**kwargs)
        model = get_peft_model(model, config)
        return model

    def _supertuning_layers(self, model):
        return [module for module in model.modules() if isinstance(module, SupertuningLinear)]

    def test_supertuning_base_weight_frozen(self):
        """The base weight is frozen; only the sparse ``values`` are trainable."""
        torch.manual_seed(0)
        model = self._prepare_trainable_model()

        saw_layer = False
        for layer in self._supertuning_layers(model):
            saw_layer = True
            weight = layer.get_base_layer().weight
            assert weight.requires_grad is False
            assert layer.supertuning_values["default"].requires_grad is True
        assert saw_layer

    def test_supertuning_gradient_only_reaches_values(self):
        """Backward must not accumulate any gradient on the frozen base weight."""
        torch.manual_seed(0)
        model = self._prepare_trainable_model()
        inputs = torch.arange(10).view(-1, 1).to(self.device)

        model(inputs).logits.float().sum().backward()

        saw_support_signal = False
        for layer in self._supertuning_layers(model):
            weight = layer.get_base_layer().weight
            values = layer.supertuning_values["default"]
            # the frozen weight receives no gradient at all
            assert weight.grad is None
            assert values.grad is not None
            if torch.any(values.grad != 0):
                saw_support_signal = True
        # the support still learns
        assert saw_support_signal

    def test_supertuning_optimizer_step_only_updates_support(self):
        """After an optimizer step, the frozen weight is untouched and only the sparse support changes.

        Uses AdamW, whose stateful update would leak into the frozen entries under the old gradient-masking
        mechanism but cannot here since the base weight receives no gradient.
        """
        torch.manual_seed(0)
        model = self._prepare_trainable_model()
        inputs = torch.arange(10).view(-1, 1).to(self.device)

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1.0)
        layer = self._supertuning_layers(model)[0]
        weight = layer.get_base_layer().weight
        values = layer.supertuning_values["default"]
        weight_before = weight.detach().clone()
        values_before = values.detach().clone()

        model(inputs).logits.float().sum().backward()
        optimizer.step()

        # the frozen base weight is never modified by the optimizer
        assert torch.equal(weight_before, weight.detach())
        # the sparse support (values) is updated
        assert not torch.equal(values_before, values.detach())

    def test_supertuning_forward_applies_sparse_update(self):
        """The forward pass adds the sparse ``values`` at ``indices`` on top of the base weight."""
        torch.manual_seed(0)
        model = self._prepare_trainable_model(init_weights=False)
        inputs = torch.arange(10).view(-1, 1).to(self.device)
        model.eval()

        layer = self._supertuning_layers(model)[0]
        weight = layer.get_base_layer().weight
        indices = layer.supertuning_indices["default"].to(torch.int64)
        values = layer.supertuning_values["default"].to(weight.dtype)

        # reconstruct the effective weight and compare against the module's own linear output
        effective = weight.detach().reshape(-1).scatter_add(0, indices, values.detach()).reshape_as(weight)
        x = torch.randn(3, layer.in_features).to(self.device).to(weight.dtype)
        expected = torch.nn.functional.linear(x, effective, layer.get_base_layer().bias)
        assert torch.allclose(layer(x), expected, atol=1e-5)
        # the update is non-trivial
        assert not torch.equal(effective, weight.detach())
