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

"""Path bootstrap so `pytest tests/` works without `pip install -e .`.

The importable package lives under ``torch-ext/`` (kernels-library layout). This conftest puts
``torch-ext`` on ``sys.path`` so ``from dora_factored import dora_factored_forward`` resolves when
running the suite directly. If the package was installed editable, the same import resolves through
the install and this is a harmless no-op. Either way the Stage A tests need no GPU / no PEFT / no
Triton / no `kernels` install.
"""

import sys
from pathlib import Path


_TORCH_EXT = Path(__file__).resolve().parent.parent / "torch-ext"
if str(_TORCH_EXT) not in sys.path:
    sys.path.insert(0, str(_TORCH_EXT))
