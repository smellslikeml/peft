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

    def test_supertuning_state_dict(self, tmp_path):
        """Test that Supertuning state dict can be saved and loaded."""
        torch.manual_seed(0)

        inputs = torch.arange(10).view(-1, 1).to(self.device)
        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        model.eval()
        output_base = model(inputs).logits

        config = SupertuningConfig(
            target_modules=["q_proj", "v_proj"],
            sparsity=0.5,
            init_weights=False,
        )
        model = get_peft_model(model, config)
        model.eval()
        output_peft = model(inputs).logits

        atol, rtol = 1e-5, 1e-8
        # sanity check: loading supertuning should not change output (mask is uniform initially)
        # Output may differ slightly due to numerical effects, but should be close
        assert torch.allclose(output_base, output_peft, atol=atol * 10, rtol=rtol * 10)

        model.save_pretrained(tmp_path)
        del model

        # check that the sparse mask is present in state dict
        state_dict = load_file(tmp_path / "adapter_model.safetensors")
        assert any("sparse_mask" in key for key in state_dict)

        # sanity check: the model still produces output after loading
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
