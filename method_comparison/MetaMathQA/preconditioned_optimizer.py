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
Riemannian preconditioning for LoRA optimizers.

This adds a small ``r x r`` preconditioner to the gradients of the LoRA ``A`` and ``B`` matrices before every base
optimizer step. For a LoRA update ``delta_W = B @ A`` with ``A`` of shape ``(r, in)`` and ``B`` of shape ``(out, r)``,
the Euclidean gradients ``g_A`` and ``g_B`` are replaced with the scaled-gradient / Riemannian directions

    g_A <- (B^T B + reg * I_r)^-1 @ g_A
    g_B <- g_B @ (A A^T + reg * I_r)^-1

Both preconditioners are ``r x r``, so the storage and runtime overhead is tiny (``r`` is the LoRA rank). Applying the
preconditioner before an existing optimizer's step is a drop-in change that stabilizes feature learning and makes
training markedly less sensitive to the learning-rate choice.

Reference:
    - Riemannian Preconditioned LoRA for Fine-Tuning Foundation Models: https://arxiv.org/abs/2402.02347
"""

from collections.abc import Callable
from typing import Optional

import torch
from torch.optim import Optimizer


def _collect_lora_pairs(model: torch.nn.Module) -> list[tuple[torch.nn.Parameter, torch.nn.Parameter]]:
    """Return the list of ``(lora_A, lora_B)`` weight parameter pairs of ``model``.

    Pairs are matched by name: for every parameter whose name contains ``lora_A`` the sibling ``lora_B`` parameter is
    looked up by substituting the substring. Only 2D weight matrices that both require gradients are returned, which is
    exactly what the ``r x r`` preconditioner is defined for.
    """
    params = dict(model.named_parameters())
    pairs = []
    for name, param_a in params.items():
        if "lora_A" not in name:
            continue
        param_b = params.get(name.replace("lora_A", "lora_B"))
        if param_b is None:
            continue
        if param_a.ndim != 2 or param_b.ndim != 2:
            continue
        if not (param_a.requires_grad and param_b.requires_grad):
            continue
        pairs.append((param_a, param_b))
    return pairs


class _RiemannianPreconditioner:
    """Applies the ``r x r`` Riemannian preconditioner in-place to LoRA gradients."""

    def __init__(self, lora_pairs: list[tuple[torch.nn.Parameter, torch.nn.Parameter]], reg: float) -> None:
        self.lora_pairs = lora_pairs
        self.reg = reg

    @staticmethod
    def _inverse(mat: torch.Tensor, reg: float) -> torch.Tensor:
        r = mat.shape[-1]
        eye = torch.eye(r, device=mat.device, dtype=mat.dtype)
        # pinv keeps this well-behaved even if reg is small and the factor is (near) rank-deficient.
        return torch.linalg.pinv(mat + reg * eye)

    @torch.no_grad()
    def step(self) -> None:
        for param_a, param_b in self.lora_pairs:
            grad_a = param_a.grad
            grad_b = param_b.grad
            if grad_a is None and grad_b is None:
                continue

            # Compute the preconditioners in at least float32 for numerical stability (e.g. when training in bf16),
            # while preserving higher precision if the parameters already use it. Cast the result back to the grad
            # dtype afterwards.
            compute_dtype = torch.promote_types(param_a.dtype, torch.float32)
            a = param_a.detach().to(compute_dtype)
            b = param_b.detach().to(compute_dtype)

            if grad_a is not None:
                # g_A <- (B^T B + reg I)^-1 g_A
                precond = self._inverse(b.T @ b, self.reg)
                new_grad_a = precond @ grad_a.to(compute_dtype)
                grad_a.copy_(new_grad_a.to(grad_a.dtype))

            if grad_b is not None:
                # g_B <- g_B (A A^T + reg I)^-1
                precond = self._inverse(a @ a.T, self.reg)
                new_grad_b = grad_b.to(compute_dtype) @ precond
                grad_b.copy_(new_grad_b.to(grad_b.dtype))


def create_riemannian_optimizer(
    model: torch.nn.Module,
    optimizer_cls: type[Optimizer],
    lr: float,
    reg: float = 1e-6,
    **optimizer_kwargs,
) -> Optimizer:
    """Instantiate a Riemannian-preconditioned optimizer for a LoRA model.

    The returned optimizer behaves exactly like ``optimizer_cls`` (e.g. ``torch.optim.AdamW`` or ``torch.optim.SGD``)
    except that, on every ``step``, the gradients of the LoRA ``A`` and ``B`` matrices are first multiplied by the
    ``r x r`` Riemannian preconditioner described in the module docstring. Non-LoRA parameters are updated unchanged.

    Args:
        model (`torch.nn.Module`): The (PEFT) model containing LoRA-adapted parameters.
        optimizer_cls (`type[torch.optim.Optimizer]`): The base optimizer class to wrap, e.g. `torch.optim.AdamW`.
        lr (`float`): Learning rate passed to the base optimizer.
        reg (`float`): Damping added to the ``r x r`` preconditioner for numerical stability (default: 1e-6).
        **optimizer_kwargs: Extra keyword arguments forwarded to the base optimizer (e.g. `weight_decay`, `betas`).

    Returns:
        `torch.optim.Optimizer`: A subclass instance of `optimizer_cls` that preconditions LoRA gradients.
    """
    lora_pairs = _collect_lora_pairs(model)
    if not lora_pairs:
        raise ValueError(
            "create_riemannian_optimizer did not find any trainable lora_A/lora_B parameter pairs on the model. "
            "The Riemannian preconditioner only applies to LoRA-style adapters."
        )

    preconditioner = _RiemannianPreconditioner(lora_pairs, reg=reg)
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    class RiemannianPreconditionedOptimizer(optimizer_cls):
        @torch.no_grad()
        def step(self, closure: Optional[Callable] = None):
            preconditioner.step()
            return super().step(closure)

    optimizer = RiemannianPreconditionedOptimizer(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer
