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

"""Integration tests for the DoRA fused kernel merge path.

These tests verify that the PEFT loader boundary composes correctly with the fused Triton kernel
loaded from ``remyxai/dora-factored-kernel`` on HF Hub. When ``USE_FACTORED_DORA_KERNEL`` is enabled,
the merged DoRA weights produced by the fused kernel must match those produced by the existing
dense path within fp32 accumulation tolerance.

Requires the optional ``kernels`` library and a CUDA + Triton runtime; otherwise the fused branch
in ``DoraLinearVariant.merge_safe`` / ``merge_unsafe`` is a no-op and these assertions collapse to
``dense == dense``. The tests are marked ``@pytest.mark.cuda`` and skip cleanly on CPU-only CI.
"""

import pytest
import torch
from torch import nn

import peft.tuners.lora.dora as dora_module
from peft import LoraConfig, get_peft_model


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU"),
]


@pytest.fixture(autouse=True)
def restore_flags():
    # Ensure the global opt-in flags never leak between tests.
    original_norm = dora_module.USE_FACTORED_DORA_NORM
    original_kernel = dora_module.USE_FACTORED_DORA_KERNEL
    yield
    dora_module.USE_FACTORED_DORA_NORM = original_norm
    dora_module.USE_FACTORED_DORA_KERNEL = original_kernel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_dora_kernel_merge_weights_match_dense_path():
    """End-to-end test: the fused kernel merge produces identical weights to the dense path.

    Creates a tiny Linear + LoRA + DoRA model, merges it twice (once with the kernel disabled,
    once with both factored-norm and kernel enabled), and asserts the resulting merged weights
    match within floating-point tolerance.
    """
    # Use both factored norm and kernel for the fast path
    dora_module.USE_FACTORED_DORA_NORM = True
    dora_module.USE_FACTORED_DORA_KERNEL = True

    # Tiny model: d_in=64, d_out=128, r=8
    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 128)

        def forward(self, x):
            return self.linear(x)

    torch.manual_seed(42)
    model = MyModule().eval()
    # init_lora_weights=False makes DoRA a non-trivial transform, so the norm actually matters
    config = LoraConfig(
        target_modules=["linear"],
        use_dora=True,
        init_lora_weights=False,
        r=8,
        lora_alpha=16,
    )
    model = get_peft_model(model, config).eval().to("cuda")

    # First merge: USE_FACTORED_DORA_KERNEL=False (dense path)
    dora_module.USE_FACTORED_DORA_KERNEL = False
    model_dense = get_peft_model(MyModule().eval().to("cuda"), config).eval()
    model_dense.load_state_dict(model.state_dict())
    with torch.inference_mode():
        w_dense = model_dense.merge_and_unload().linear.weight.data.clone()

    # Second merge: USE_FACTORED_DORA_KERNEL=True (fused kernel path)
    dora_module.USE_FACTORED_DORA_KERNEL = True
    model_kernel = get_peft_model(MyModule().eval().to("cuda"), config).eval()
    model_kernel.load_state_dict(model.state_dict())
    with torch.inference_mode():
        w_kernel = model_kernel.merge_and_unload().linear.weight.data.clone()

    # The two paths must produce identical merged weights (within fp accumulation tolerance)
    torch.testing.assert_close(w_kernel, w_dense, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_dora_kernel_merge_with_factored_norm_enabled():
    """Test that the kernel works correctly when both flags are enabled.

    This is the expected production configuration: both factored norm and kernel enabled.
    The test verifies the fused kernel path is numerically equivalent to the dense path.
    """
    dora_module.USE_FACTORED_DORA_NORM = True
    dora_module.USE_FACTORED_DORA_KERNEL = True

    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 64)

        def forward(self, x):
            return self.linear(x)

    torch.manual_seed(123)
    model = MyModule().eval().to("cuda")
    config = LoraConfig(
        target_modules=["linear"],
        use_dora=True,
        init_lora_weights=False,
        r=4,
    )
    model = get_peft_model(model, config).eval()

    # Snapshot adapter state BEFORE any merge — merge_and_unload strips the LoRA/DoRA modules,
    # leaving a bare base layer whose state_dict can't be loaded into a fresh PeftModel.
    state = {k: v.clone() for k, v in model.state_dict().items()}

    # Merge with the fused kernel
    with torch.inference_mode():
        w_kernel = model.merge_and_unload().linear.weight.data.clone()

    # Compare with the dense path (both flags disabled)
    dora_module.USE_FACTORED_DORA_NORM = False
    dora_module.USE_FACTORED_DORA_KERNEL = False
    model_dense = get_peft_model(MyModule().eval().to("cuda"), config).eval()
    model_dense.load_state_dict(state)
    with torch.inference_mode():
        w_dense = model_dense.merge_and_unload().linear.weight.data.clone()

    torch.testing.assert_close(w_kernel, w_dense, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_dora_kernel_merge_multiple_adapters():
    """Test that the kernel merge works correctly with multiple DoRA adapters."""
    dora_module.USE_FACTORED_DORA_NORM = True
    dora_module.USE_FACTORED_DORA_KERNEL = True

    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 48)

        def forward(self, x):
            return self.linear(x)

    torch.manual_seed(456)
    model = MyModule().eval().to("cuda")
    config = LoraConfig(
        target_modules=["linear"],
        use_dora=True,
        init_lora_weights=False,
        r=4,
    )
    model = get_peft_model(model, config).eval()

    # Add a second adapter
    config2 = LoraConfig(
        target_modules=["linear"],
        use_dora=True,
        init_lora_weights=False,
        r=6,
    )
    model.add_adapter("adapter2", config2)

    # Snapshot state BEFORE merging so we can rehydrate the dense-path comparator with the same
    # (multi-adapter) initialization. merge_and_unload strips adapter modules from `model`.
    state = {k: v.clone() for k, v in model.state_dict().items()}

    # Merge the first adapter with the kernel
    with torch.inference_mode():
        w_kernel = model.merge_and_unload().linear.weight.data.clone()

    # Compare with the dense path
    dora_module.USE_FACTORED_DORA_KERNEL = False
    model_dense = get_peft_model(MyModule().eval().to("cuda"), config).eval()
    model_dense.add_adapter("adapter2", config2)
    model_dense.load_state_dict(state)
    with torch.inference_mode():
        w_dense = model_dense.merge_and_unload().linear.weight.data.clone()

    torch.testing.assert_close(w_kernel, w_dense, atol=1e-4, rtol=1e-4)
