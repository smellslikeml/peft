# dora-factored-kernel

A [`huggingface/kernels`](https://github.com/huggingface/kernels)-shaped package for the
**factored-norm DoRA** weight-adaptation operator from
[*Scaling DoRA: High-Rank Adaptation via Factored Norms and Fused Kernels* (arXiv:2603.22276)](https://arxiv.org/abs/2603.22276).

This is **Stage A**: a pure-PyTorch **reference** implementation. It ports the factored-norm
decomposition verbatim from PEFT's merged [`factored_weight_norm`](https://github.com/huggingface/peft/pull/3382)
and exposes it through the single callable that the eventual fused Triton kernel (Stage B) will slot
into — callers keep one import regardless of which path is loaded.

> **Status:** Stage A — PyTorch reference only. **No Triton, no CUDA, not yet Hub-published.** The
> fused fast-path (Stage B) and the PEFT-side loader wiring (Stage C) are separate dispatches.

## What it computes

DoRA rescales the adapted weight by the column-wise L2 norm of `W + s·BA`. The factored decomposition
expands the squared norm into three terms that never materialize the dense `[d_out, d_in]` product:

```
||W_i + s·(BA)_i||² = ||W_i||² + 2s·<W_i, (BA)_i> + s²·||(BA)_i||²
                     = base_i    + 2s·cross_i        + s²·gram_i
```

yielding `O(d_out·r + r²)` intermediates instead of `O(d_out·d_in)`. The public callable returns the
DoRA effective weight:

```python
W_eff = (dora_scale / ||W + s·BA||) ⊙ (W + s·BA)
```

## Install (local, Stage A)

```bash
pip install -e kernels/dora_factored
```

The package is standalone — it depends on `torch` only and does **not** require `peft` or
`transformers`.

## Usage

```python
import torch
from dora_factored import dora_factored_forward

# base_weight: [d_out, d_in], lora_a: [r, d_in], lora_b: [d_out, r]
W_eff = dora_factored_forward(base_weight, lora_a, lora_b, scaling=1.0, dora_scale=magnitude)
# W_eff: [d_out, d_in]
```

`dora_scale` is the DoRA magnitude vector of shape `[d_out]` (one value per output column, matching
PEFT's `DoraLinearLayer.weight`); a scalar is broadcast across columns.

## Eventual Hub install (Stage B–pending)

Once Stage B lands the fused Triton kernel and the package is published to the Hub, it will be
loadable via the `kernels` library — **this does not work yet** (no compiled backend in Stage A):

```python
# PENDING Stage B + Hub publish — left here as the target surface:
from kernels import get_kernel

dora_factored = get_kernel("kernels-community/dora-factored-kernel", version=1)
W_eff = dora_factored.dora_factored_forward(base_weight, lora_a, lora_b, scaling, dora_scale)
```

The `dora_factored_forward(base_weight, lora_a, lora_b, scaling, dora_scale)` signature is the stable
surface Stage B implements behind.

## Tests

CPU-only, no GPU / no PEFT / no Triton required:

```bash
pytest kernels/dora_factored/tests/
```

`tests/test_reference_parity.py` asserts the reference matches a naive dense DoRA baseline
(parametrized over `fp32`/`bf16` and scaling), mirroring the discipline of PEFT's merged
`tests/test_dora_factored_norm.py`.

## Layout

```
kernels/dora_factored/
├── build.toml                 # kernels-library manifest ([general]/[general.hub]; [torch]/[kernel.*] await Stage B)
├── flake.nix                  # reproducible-build spec (structural in Stage A)
├── pyproject.toml             # packaging: name `dora-factored-kernel`, torch-only dep
├── README.md
├── torch-ext/
│   └── dora_factored/
│       ├── __init__.py        # public API: dora_factored_forward(...)
│       └── reference.py       # pure-PyTorch reference (ported from PEFT factored_weight_norm)
└── tests/
    └── test_reference_parity.py
```

## Algorithmic provenance

- **Paper:** [Scaling DoRA — Factored Norms + Fused Kernels (arXiv:2603.22276)](https://arxiv.org/abs/2603.22276)
- **Algorithmic complement (merged upstream):** [huggingface/peft#3382](https://github.com/huggingface/peft/pull/3382) —
  the factored-norm decomposition behind `USE_FACTORED_DORA_NORM` in `src/peft/tuners/lora/dora.py`.
  This package's `reference.py` ports that verbatim; it is the numerical target Stage B must match.
- **PEFT integration invitation:** [sockeye44/dorafactors#1](https://github.com/sockeye44/dorafactors/issues/1)

## Stage progression

- **Stage A (this package):** `kernels`-library-shaped scaffold + PyTorch reference. No Triton.
- **Stage B (later):** fused Triton implementation behind `dora_factored_forward` (human GPU-validates
  numerical parity + perf).
- **Stage C (later):** wire `USE_FACTORED_DORA_NORM`'s fast path in `src/peft/tuners/lora/dora.py` to
  `kernels.get_kernel(...)`; extract via `git subtree split --prefix=kernels/dora_factored` for the
  Hub publish and shepherd upstream to `huggingface/peft`.

## Stage B — fused Triton fast-path (this change)

> Supersedes the *Status* note above for the fast path: `dora_factored_forward(...)` now auto-dispatches
> to a fused Triton kernel on CUDA and still falls back to the Stage A PyTorch reference on CPU / when
> `triton` is absent. The signature is unchanged (no `use_triton=` kwarg); callers keep one import.

The compose (forward) and backward kernels are ported **verbatim** from the paper's reference
implementation at `sockeye44/dorafactors/code/kernelagent_sols/` (`optimize_dora_compose/...` and
`optimize_dora_backward/...`): the `@triton.jit` bodies, `@triton.autotune` configs, and
`@triton.heuristics` predicates are byte-for-byte identical (only the Python launch wrappers are
renamed to fit the package). They are wired behind the public callable through a
`torch.autograd.Function` (`autograd.DoraFactoredFn`) so the fast path composes into a normal
PyTorch graph — the prerequisite for the Stage C PEFT-side wiring.

Two wrapper-level adaptations make the verbatim kernels land behind the Stage A signature:

- The kernels bake a literal `0.7` LoRA coefficient into their bodies. The wrapper folds
  `scaling / 0.7` into the dense LoRA delta so that `0.7` cancels and the result is correct for any
  caller `scaling` — without touching the kernel body.
- The backward treats the magnitude norm as detached (PEFT DoRA §4.3); the forward detaches
  `weight_norm` to match, so the fused backward is exact, not approximate.

New layout under `torch-ext/dora_factored/`:

```
torch-ext/dora_factored/
├── __init__.py            # dora_factored_forward — transparent CUDA dispatch (updated)
├── reference.py           # Stage A PyTorch reference (untouched — parity target)
├── triton_compose.py      # NEW — _fused_dora_compose_kernel (forward), ported verbatim
├── triton_backward.py     # NEW — two-stage backward kernel, ported verbatim
└── autograd.py            # NEW — torch.autograd.Function pairing them
```

### Tests

CPU CI cannot run GPU kernels, so the parity suite is *armed*: `tests/test_triton_parity.py` marks
every test `@pytest.mark.cuda` and skips when CUDA (or `triton`) is absent. On a reviewer's GPU it
asserts forward parity (vs the reference and a dense baseline) and `gradcheck`-style backward parity
(vs the reference's detached-norm autograd), across the autotune tile boundaries
(`BLOCK_N ∈ {128,256,512}`, `BLOCK_M ∈ {8,16,32,64}`, even **and** odd tails) and `fp16`/`bf16`/`fp32`,
at non-`0.7` scalings to exercise the fold. Run on a CUDA host with `triton` installed:

```bash
pytest kernels/dora_factored/tests/ -m cuda
```

(On CPU the whole package still installs and the Stage A reference suite passes unchanged:
`pytest kernels/dora_factored/tests/`.) The `cuda` marker is unregistered here because registering it
would require editing `pyproject.toml`, which is outside this stage's allowed edits; it emits a
harmless `PytestUnknownMarkWarning` and can be registered post-merge.
