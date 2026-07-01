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

"""Tests for the Riemannian preconditioned optimizer and its wiring into the MetaMathQA training utils."""

import pytest


# torch is required for every test in this file; skip cleanly where it is not installed.
torch = pytest.importorskip("torch")

from preconditioned_optimizer import _collect_lora_pairs, create_riemannian_optimizer  # noqa: E402


class TinyLoRALinear(torch.nn.Module):
    """Minimal module exposing lora_A/lora_B weights with the same naming PEFT uses."""

    def __init__(self, in_features: int, out_features: int, r: int) -> None:
        super().__init__()
        self.base = torch.nn.Linear(in_features, out_features, bias=False)
        self.base.weight.requires_grad_(False)
        self.lora_A = torch.nn.Linear(in_features, r, bias=False)  # weight shape (r, in)
        self.lora_B = torch.nn.Linear(r, out_features, bias=False)  # weight shape (out, r)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))


def _make_model(in_features=6, out_features=4, r=2):
    torch.manual_seed(0)
    model = TinyLoRALinear(in_features, out_features, r).double()
    # non-trivial B so that B^T B is not the identity
    with torch.no_grad():
        model.lora_B.weight.copy_(torch.randn_like(model.lora_B.weight))
    return model


def test_collect_lora_pairs_matches_a_and_b():
    model = _make_model()
    pairs = _collect_lora_pairs(model)
    assert len(pairs) == 1
    param_a, param_b = pairs[0]
    assert param_a is model.lora_A.weight
    assert param_b is model.lora_B.weight


def test_create_riemannian_optimizer_is_a_real_optimizer():
    model = _make_model()
    optimizer = create_riemannian_optimizer(model, optimizer_cls=torch.optim.AdamW, lr=1e-3)
    # must remain a genuine AdamW so GradScaler / schedulers keep working
    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(optimizer, torch.optim.Optimizer)
    # frozen base weight must be excluded from the trainable parameter set
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert id(model.base.weight) not in optimized
    assert id(model.lora_A.weight) in optimized
    assert id(model.lora_B.weight) in optimized


def test_create_riemannian_optimizer_requires_lora():
    plain = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match="lora"):
        create_riemannian_optimizer(plain, optimizer_cls=torch.optim.AdamW, lr=1e-3)


def test_step_applies_rxr_preconditioner_to_lora_grads():
    reg = 1e-6
    model = _make_model()
    optimizer = create_riemannian_optimizer(model, optimizer_cls=torch.optim.AdamW, lr=1e-3, reg=reg)

    x = torch.randn(8, model.lora_A.weight.shape[1], dtype=torch.double)
    model(x).pow(2).sum().backward()

    # snapshot Euclidean grads and the factor values used by the preconditioner (taken before the param update)
    grad_a = model.lora_A.weight.grad.clone()
    grad_b = model.lora_B.weight.grad.clone()
    A = model.lora_A.weight.detach().clone()
    B = model.lora_B.weight.detach().clone()
    r = A.shape[0]
    eye = torch.eye(r, dtype=torch.double)

    expected_grad_a = torch.linalg.pinv(B.T @ B + reg * eye) @ grad_a
    expected_grad_b = grad_b @ torch.linalg.pinv(A @ A.T + reg * eye)

    # AdamW.step reads .grad but does not modify it, so after the step .grad holds the preconditioned gradient
    optimizer.step()

    assert torch.allclose(model.lora_A.weight.grad, expected_grad_a, atol=1e-8)
    assert torch.allclose(model.lora_B.weight.grad, expected_grad_b, atol=1e-8)
    # the preconditioned direction genuinely differs from the raw gradient
    assert not torch.allclose(model.lora_A.weight.grad, grad_a, atol=1e-6)


def _import_utils_or_skip():
    # MetaMathQA/utils.py imports heavy deps and requires an accelerator at import time; skip where unavailable.
    try:
        import utils
    except Exception as exc:  # any import-time failure means the harness cannot run here
        pytest.skip(f"MetaMathQA.utils is unavailable in this environment: {exc}")
    return utils


def test_get_optimizer_and_scheduler_wires_riemannian():
    utils = _import_utils_or_skip()
    model = _make_model()

    optimizer, _scheduler = utils.get_optimizer_and_scheduler(
        model,
        optimizer_type="riemannian",
        max_steps=10,
        lr_scheduler_arg=None,
        lr=1e-3,
    )
    assert isinstance(optimizer, torch.optim.AdamW)

    # confirm the optimizer returned through the real call site actually preconditions LoRA gradients
    x = torch.randn(8, model.lora_A.weight.shape[1], dtype=torch.double)
    model(x).pow(2).sum().backward()
    raw_grad_a = model.lora_A.weight.grad.clone()
    optimizer.step()
    assert not torch.allclose(model.lora_A.weight.grad, raw_grad_a, atol=1e-6)
