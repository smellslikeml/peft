<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# SaMA: Scalable Morpho Adaptation

[SaMA](https://huggingface.co/papers/2611.04462) is a parameter-efficient fine-tuning method built on a
generalization of the Kronecker product. By relaxing the shared-block constraint in the perfect-shuffle
factorization, SaMA spans from low to full rank: its adapter rank grows quadratically with the parameter
budget (up to full rank), and its update merges into the base weights at inference with zero added latency.

# API

## SaMAConfig

[[autodoc]] tuners.sama.config.SaMAConfig

## SaMAModel

[[autodoc]] tuners.sama.model.SaMAModel
