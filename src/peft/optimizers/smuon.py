# Copyright 2026-present the HuggingFace Inc. team.
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
This module contains the implementation of the sMuon (approximate Muon) LoRA optimizer.
"""

import inspect
from collections.abc import Callable
from typing import Optional

import torch
from torch.optim import Optimizer

from ..peft_model import PeftModel
from .riemannian import _collect_lora_pairs

# Tolerance at which the Newton-Schulz iterations below are considered converged: the residual
# `I - X Y^2` for `_invroot`, the relative change of the iterate for `_msign`.
_NS_TOL = 1e-6


def _invroot(mat: torch.Tensor, steps: int) -> torch.Tensor:
    """Inverse square root of a symmetric positive definite matrix via Newton-Schulz.

    The classical iteration `Y <- Y (3 I - X Y^2) / 2` is a fixed point at `Y = X^-1/2` and
    converges quadratically once `I - X Y^2` is small. Normalising by the mean eigenvalue puts the
    spectrum around 1, which is where the iteration is well behaved.

    At the fixed point the iteration is only neutrally stable, so floating-point noise starts
    growing again once convergence is reached. The loop therefore stops at the first iterate that
    does not improve on the best residual seen so far and returns that best iterate.
    """
    return _invroot_batch(mat.unsqueeze(0), steps).squeeze(0)


def _invroot_batch(mat: torch.Tensor, steps: int) -> torch.Tensor:
    """Batched `_invroot` over a leading batch dimension.

    Members of the batch leave the loop at different iterations, so each keeps iterating until its
    own residual stops improving; the returned iterate is the best one seen for that member.
    """
    r = mat.shape[-1]
    eye = torch.eye(r, device=mat.device, dtype=mat.dtype)
    scale = torch.diagonal(mat, dim1=-2, dim2=-1).sum(-1) / r
    mat = mat / scale.view(-1, 1, 1)
    y = eye.expand_as(mat).clone()
    best, best_residual = y.clone(), torch.full((mat.shape[0],), float("inf"), device=mat.device, dtype=mat.dtype)
    for _ in range(steps):
        residual = (eye - mat @ (y @ y)).flatten(1).norm(dim=1)
        improved = residual < best_residual
        best = torch.where(improved.view(-1, 1, 1), y, best)
        best_residual = torch.where(improved, residual, best_residual)
        # Members whose residual no longer improves hold their iterate; only the rest advance.
        y = torch.where(improved.view(-1, 1, 1), y @ (3 * eye - mat @ (y @ y)) / 2, y)
    return best / torch.sqrt(scale).view(-1, 1, 1)


def _msign(mat: torch.Tensor, steps: int) -> torch.Tensor:
    """Polar factor `U V^T` of `mat` via Newton-Schulz.

    The classical iteration `X <- X (3 I - X^T X) / 2` drives each singular value along
    `sigma -> sigma (3 - sigma^2) / 2`, which converges to 1 for `sigma` in `(0, sqrt(3))`. Dividing
    by the Frobenius norm (an upper bound on the spectral norm) puts the starting values in range.
    Singular values that are exactly 0 stay 0, so the rank is preserved.

    Unlike `_invroot`, `sigma = 1` is a superattracting fixed point: round-off is damped rather than
    amplified, so iterating a fixed number of steps cannot degrade the result and no guard is needed.
    """
    return _msign_batch(mat.unsqueeze(0), steps).squeeze(0)


def _msign_batch(mat: torch.Tensor, steps: int) -> torch.Tensor:
    """Batched `_msign` over a leading batch dimension.

    Adapters of the same shape are stacked into one tensor and run as a single batched iteration,
    which is what keeps the per-step cost close to that of a single adapter's.

    Convergence is measured by the relative change of the iterate rather than by `X^T X - I`: the
    matrices passed here are rank-deficient by construction — the `2r x 2r` core has a zero block
    (Eq. 19) and `Y` / `Z` are projections (Eq. 18) — so their null directions keep
    `max |X^T X - I|` at 1 forever no matter how far the iteration has come.
    """
    cols = mat.shape[-1]
    eye = torch.eye(cols, device=mat.device, dtype=mat.dtype)
    norm = mat.flatten(1).norm(dim=1).clamp_min(_NS_TOL)
    x = mat / norm.view(-1, 1, 1)
    for _ in range(steps):
        updated = x @ (3 * eye - x.transpose(-1, -2) @ x) / 2
        # Members converge at different rates; stop once the slowest has settled. The change is
        # scaled per member, since the members that still iterate carry little of the batch norm.
        change = ((updated - x).flatten(1).norm(dim=1) / x.flatten(1).norm(dim=1).clamp_min(_NS_TOL)).amax()
        x = updated
        if change < _NS_TOL:
            break
    return x


class _SMuonTransform:
    """Applies the sMuon transform in-place to LoRA gradients.

    For each `(lora_A, lora_B)` pair the accumulated momenta are orthogonalized as a whole adapter
    update rather than factor by factor: the low-rank momentum `H = B X + Y A` (Eq. 15) is built
    from the momenta, its polar factor `msign(H)` is computed on a `2r x 2r` core (Eq. 16, 19, 20),
    and the least-squares factor steps of Eq. 21 are written back into the gradients. Everything
    above is matmuls plus the two `r x r` Newton-Schulz iterations — no SVD, QR or eigendecomposition.

    The gradients handed to the base optimizer are the descent directions `delta_A` / `delta_B`,
    already rescaled by `0.2 sqrt(d_in d_out / r)` (Eq. 23) so that the learning rate of the base
    optimizer plays the role of `eta` in the paper's update.
    """

    def __init__(
        self,
        lora_pairs: list[tuple[torch.nn.Parameter, torch.nn.Parameter]],
        beta: float,
        eps: float,
        ns_steps: int,
    ) -> None:
        self.lora_pairs = lora_pairs
        self.beta = beta
        self.eps = eps
        self.ns_steps = ns_steps
        # Group the pairs by adapter shape. Pairs in one group are stacked and transformed as a
        # single batched tensor, which is what keeps the Newton-Schulz loops — the only Python loops
        # in `step` — proportional to the number of distinct shapes rather than to the number of
        # adapters. A model with uniform target shapes collapses this to a single group.
        self.groups: dict[tuple[int, int, int], list[int]] = {}
        for idx, (param_a, param_b) in enumerate(lora_pairs):
            d1, r = param_b.shape
            d2 = param_a.shape[1]
            self.groups.setdefault((d1, d2, r), []).append(idx)
        # Per-pair momenta M_A / M_B, accumulated as `beta M + (1 - beta) G` (App. A.3). The EMA is
        # zero-initialised, so the first step sees `(1 - beta) G`. Buffers are created on first use.
        self.momenta: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = [None] * len(lora_pairs)

    @torch.no_grad()
    def step(self) -> None:
        for indices in self.groups.values():
            self._step_group(indices)

    def _step_group(self, indices: list[int]) -> None:
        # Compute the step in at least float32 for numerical stability when training in bf16,
        # matching the paper's bf16 storage with fp32 step computation. Results are cast back to
        # each gradient's dtype.
        pairs = [self.lora_pairs[i] for i in indices]
        compute_dtype = torch.promote_types(pairs[0][0].dtype, torch.float32)
        # A: r x d2 and B: d1 x r, so that the adapter is dW = B A of shape d1 x d2.
        d1, r = pairs[0][1].shape
        d2 = pairs[0][0].shape[1]
        device = pairs[0][0].device
        eye = torch.eye(r, device=device, dtype=compute_dtype)

        a = torch.stack([p[0].detach().to(compute_dtype) for p in pairs])
        b = torch.stack([p[1].detach().to(compute_dtype) for p in pairs])
        # Pairs whose gradient is absent contribute zeros; a zero matrix is a fixed point of the
        # transform, so those gradients stay untouched rather than being invented.
        grads_a = [p[0].grad if p[0].grad is not None else torch.zeros_like(a[i]) for i, p in enumerate(pairs)]
        grads_b = [p[1].grad if p[1].grad is not None else torch.zeros_like(b[i]) for i, p in enumerate(pairs)]
        g_a = torch.stack([g.to(compute_dtype) for g in grads_a])
        g_b = torch.stack([g.to(compute_dtype) for g in grads_b])

        # App. A.3: autograd already reports the adapter gradients the paper denotes G_A = B^T G
        # and G_B = G A^T as the gradients of the two factors (up to the lora_alpha / r scaling,
        # which msign is invariant to). The EMA is accumulated over the transported buffers below.
        if any(self.momenta[i] is None for i in indices):
            m_a, m_b = torch.zeros_like(a), torch.zeros_like(b)
        else:
            m_a = torch.stack([self.momenta[i][0] for i in indices])  # type: ignore[index]
            m_b = torch.stack([self.momenta[i][1] for i in indices])  # type: ignore[index]
        m_a = self.beta * m_a + (1 - self.beta) * g_a
        m_b = self.beta * m_b + (1 - self.beta) * g_b

        # Alg. 3 lines 1-2: relative jitter, then the two r x r inverse square roots S_B and S_A.
        d_b = b.transpose(-1, -2) @ b
        d_a = a @ a.transpose(-1, -2)
        eps_b = self.eps * torch.clamp(torch.diagonal(d_b, dim1=-2, dim2=-1).sum(-1) / r, min=1.0)
        eps_a = self.eps * torch.clamp(torch.diagonal(d_a, dim1=-2, dim2=-1).sum(-1) / r, min=1.0)
        s_b = _invroot_batch(d_b + eps_b.view(-1, 1, 1) * eye, self.ns_steps)
        s_a = _invroot_batch(d_a + eps_a.view(-1, 1, 1) * eye, self.ns_steps)

        # Alg. 3 lines 3-8: the low-rank momentum factors and orthonormal bases for col(H) and
        # row(H) (Eq. 15, 18). Each basis is projected twice, which the paper credits to Giraud
        # et al. for Gram-Schmidt rounding error.
        x = s_b @ s_b @ m_a
        v1 = _msign_batch(a.transpose(-1, -2), self.ns_steps)

        y_hat = m_b - b @ (s_b @ s_b @ (b.transpose(-1, -2) @ m_b))
        y_hat = y_hat - b @ (s_b @ s_b @ (b.transpose(-1, -2) @ y_hat))
        u2 = _msign_batch(y_hat, self.ns_steps)
        y = y_hat @ s_a @ s_a

        z_t = m_a.transpose(-1, -2) - v1 @ (v1.transpose(-1, -2) @ m_a.transpose(-1, -2))
        z_t = z_t - v1 @ (v1.transpose(-1, -2) @ z_t)
        v2 = _msign_batch(z_t, self.ns_steps)

        # Alg. 3 line 9: the 2r x 2r core U^T H V (Eq. 19) and its polar factor Omega (Eq. 20).
        core = a.new_zeros((len(pairs), 2 * r, 2 * r))
        core[:, :r, :r] = s_b @ (m_a @ v1)
        core[:, :r, r:] = s_b @ (m_a @ v2)
        core[:, r:, :r] = (u2.transpose(-1, -2) @ y) @ (a @ v1)
        omega = _msign_batch(core, self.ns_steps)

        # Alg. 3 line 10: the step directions of Eq. 21, with the symmetric P and T factors.
        p = (a @ v1 + (a @ v1).transpose(-1, -2)) / 2
        t = (p @ s_a @ s_a + (p @ s_a @ s_a).transpose(-1, -2)) / 2
        delta_a = s_b @ (omega[:, :r, :r] @ v1.transpose(-1, -2) + omega[:, :r, r:] @ v2.transpose(-1, -2))
        delta_b = u2 @ omega[:, r:, :r] @ t

        # Eq. 23: rescale so the adapter update's RMS matches Muon's. The base optimizer's learning
        # rate plays the role of eta and its decoupled weight decay that of lambda in Eqs. 24-25, so
        # neither is applied here.
        scale = 0.2 * (d1 * d2 / r) ** 0.5
        for i, (param_a, param_b) in enumerate(pairs):
            if param_a.grad is not None:
                param_a.grad.copy_((scale * delta_a[i]).to(param_a.grad.dtype))
            if param_b.grad is not None:
                param_b.grad.copy_((scale * delta_b[i]).to(param_b.grad.dtype))

        # Alg. 3 lines 14-15 / Eq. 34: transport the momenta into the subspace the update map can
        # express. The paper uses the post-update factors B', A'; the base optimizer owns the
        # parameter update here, so the pre-update factors are used instead, which agrees with
        # Eq. 34 to first order in the step size.
        transported_a = b.transpose(-1, -2) @ b @ x + (b.transpose(-1, -2) @ y) @ a
        transported_b = b @ (x @ a.transpose(-1, -2)) + y @ (a @ a.transpose(-1, -2))
        for i, idx in enumerate(indices):
            self.momenta[idx] = (transported_a[i], transported_b[i])


def create_smuon_optimizer(
    model: PeftModel,
    optimizer_cls: type[Optimizer],
    *,
    lr: float,
    beta: float = 0.9,
    eps: float = 1e-4,
    ns_steps: int = 30,
    **kwargs,
) -> Optimizer:
    """
    Creates an approximate-Muon (sMuon) optimizer for a LoRA-adapted model.

    Approximate Muon with low-rank adapters (Anson, Houghton & Milsom): https://huggingface.co/papers/2608.14492

    The returned optimizer behaves exactly like `optimizer_cls` (e.g. `torch.optim.AdamW` or
    `torch.optim.SGD`) except that, on every `step`, the gradients of the LoRA `A` and `B` matrices
    are first replaced by the sMuon step directions. Non-LoRA parameters are updated unchanged.

    sMuon approximates Muon's spectrally-orthogonalized update for the composite adapter `dW = B A`:
    the low-rank momentum is orthogonalized as one `d1 x d2` object and mapped back onto the two
    factors by least squares (Eq. 11, 21). The orthogonalization runs on a `2r x 2r` core via
    Newton-Schulz iterations (Eq. 16), so no SVD, QR or eigendecomposition is needed and the only
    matrices ever inverted or decomposed are `r x r`.

    Only `nn.Linear`-shaped LoRA layers (`lora_A` / `lora_B`) are transformed. LoRA on embedding
    layers (`lora_embedding_A` / `lora_embedding_B`) and every other trainable parameter — biases,
    DoRA's magnitude vector — is left untouched and updated by `optimizer_cls` directly.

    Notes on hyperparameters:

    - `beta`, `eps` and the weight decay of the base optimizer follow the paper's SFT setup
      (momentum 0.9, weight decay 0.01, jitter `1e-4`).
    - The learning rate of the base optimizer is `eta` in the paper. Gradients are rescaled by
      `0.2 sqrt(d_in d_out / r)` (Eq. 23) so that a learning rate tuned for the paper transfers.
    - sMuon carries its own momentum, so `optimizer_cls` with momentum already applied (e.g. AdamW,
      or SGD with `momentum > 0`) stacks a second moment estimate on top of the spectral direction.
      The paper pairs sMuon with plain momentum only; `SGD(momentum=0)` is the closest match.
    - The paper initializes `A = 0` and `B` orthonormal. PEFT's default LoRA init (`B = 0`) also
      works — `G_A = B^T G` vanishes on the first step, so `lora_A` only starts moving on the second
      one. Factors that are (near) rank-deficient amplify `delta_A`, since the least-squares step
      goes through `B^T B`; the `eps` jitter damps this. Prefer well-conditioned factors.

    Args:
        model (`torch.nn.Module`): The PEFT model containing LoRA-adapted parameters.
        optimizer_cls (`type[torch.optim.Optimizer]`): The base optimizer class to wrap, e.g. `torch.optim.AdamW`.
        lr (`float`): Learning rate passed to the base optimizer; this is `eta` in the paper.
        beta (`float`): Momentum of the sMuon momenta `M_A` / `M_B`, accumulated as `beta M + (1 - beta) G`.
        eps (`float`): Jitter added to the `r x r` Gram matrices before the inverse square root,
            relative to their mean eigenvalue (Alg. 3). Stabilizes factors that are close to
            rank-deficient.
        ns_steps (`int`): Maximum number of Newton-Schulz iterations per `_msign` / `_invroot` call.
            The paper defers the choice of polynomial iteration to the literature and fixes no
            iteration count, so the classical (tuning-free) iteration is used and run to the
            `_NS_TOL` residual or `ns_steps`, whichever comes first.
        kwargs (`dict`): Additional keyword arguments forwarded to the base optimizer (e.g. `weight_decay`).

    Returns:
        `torch.optim.Optimizer`: A subclass instance of `optimizer_cls` that replaces LoRA gradients
        with the sMuon step directions on each `step`. Only `lora_A` / `lora_B` weight matrices are
        transformed; every other trainable parameter is updated by `optimizer_cls` unchanged.
    """
    if not issubclass(optimizer_cls, Optimizer):
        raise TypeError(f"optimizer_cls must be a subclass of torch.optim.Optimizer, got {optimizer_cls!r}.")

    lora_pairs = _collect_lora_pairs(model)
    if not lora_pairs:
        raise ValueError(
            "create_smuon_optimizer did not find any trainable lora_A/lora_B parameter pairs on the model. "
            "The sMuon transform only applies to LoRA-style adapters."
        )

    transform = _SMuonTransform(lora_pairs, beta=beta, eps=eps, ns_steps=ns_steps)
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    # Reject closure-required optimizers (LBFGS) at construction. Even with correct ordering the
    # transform breaks LBFGS's secant-condition curvature estimate. Detect by signature so we don't
    # hardcode a class name.
    step_sig = inspect.signature(optimizer_cls.step)
    closure_param = step_sig.parameters.get("closure")
    if closure_param is not None and closure_param.default is inspect.Parameter.empty:
        raise ValueError(
            f"{optimizer_cls.__name__} requires a closure and re-evaluates the "
            "objective inside step(); the sMuon transform would be recomputed each "
            "iteration, invalidating the optimizer's curvature or line-search state. "
            "Use an optimizer whose step() has an optional closure (SGD, AdamW, etc)."
        )

    class SMuonOptimizer(optimizer_cls):
        @torch.no_grad()
        def step(self, closure: Optional[Callable] = None):
            # Closures typically do ``zero_grad(); loss.backward()``, which torch optimizers invoke
            # at the top of their own ``step()``. That would overwrite the transformed ``.grad``
            # before the base optimizer reads it, silently dropping the sMuon step. Reject at step
            # time rather than support a shape no shipping PEFT consumer uses.
            if closure is not None:
                raise ValueError(
                    "The sMuon optimizer does not support closures. Closures re-evaluate the objective "
                    "inside step(), which would overwrite the transformed gradients before the base "
                    "optimizer consumes them."
                )
            transform.step()
            return super().step()

    optimizer = SMuonOptimizer(trainable_params, lr=lr, **kwargs)
    return optimizer
