# Reproduducible-build spec for the DoRA-factored kernel.
#
# Mirrors the flake.nix shape every kernels-community/* package ships (see
# huggingface/kernels examples/kernels/*/flake.nix). Stage A has no compiled artifact
# to build — only the pure-PyTorch reference — so this flake is structural. The
# `kernel-builder` input resolves once the package is built inside the huggingface/kernels
# monorepo, or after `git subtree split --prefix=kernels/dora_factored` extracts it to a
# standalone repo for the Stage C Hub publish.
{
  description = "Flake for dora-factored kernel (Stage A: PyTorch reference, no Triton yet)";

  inputs.kernel-builder.url = "path:../../..";

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
